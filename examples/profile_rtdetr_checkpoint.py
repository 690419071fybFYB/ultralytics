#!/usr/bin/env python3
"""Layer-wise profiler for RT-DETR checkpoints.

Outputs per-layer latency (ms), parameters, and per-layer GFLOPs (when available),
plus model-level summary statistics.

Example:
    /home/fyb/envs/torch-cuda/bin/python examples/profile_rtdetr_checkpoint.py \
        --model /path/to/your_checkpoint.pt --imgsz 512 --device 0 \
        --warmup 5 --repeats 30 --save-csv runs/profile/rtdetr_profile.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics import RTDETR
from ultralytics.utils.torch_utils import get_flops

try:
    import thop
except Exception:
    thop = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile RT-DETR checkpoint layer-wise.")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt", help="Path to RT-DETR checkpoint.")
    parser.add_argument("--imgsz", type=int, default=512, help="Square input size.")
    parser.add_argument("--device", type=str, default="0", help='Device, e.g. "0", "cuda:0", or "cpu".')
    parser.add_argument("--warmup", type=int, default=5, help="Warmup forward passes before timing.")
    parser.add_argument("--repeats", type=int, default=30, help="Timed forward passes to average.")
    parser.add_argument("--topk", type=int, default=10, help="Show top-k slowest layers.")
    parser.add_argument("--save-csv", type=str, default="", help="Optional CSV output path.")
    parser.add_argument(
        "--query-rerank-mode",
        type=str,
        default="",
        choices=("", "none", "center"),
        help="Optional override for RT-DETR query rerank mode.",
    )
    parser.add_argument("--center-lambda-max", type=float, default=0.25, help="Center rerank lambda max.")
    parser.add_argument("--center-score-norm", type=str, default="zscore_image", help="Center score norm mode.")
    parser.add_argument("--center-score-clip", type=float, default=6.0, help="Center fused-score clip.")
    parser.add_argument("--no-layer-flops", action="store_true", help="Disable per-layer FLOPs estimation.")
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    d = device_str.strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d.startswith("cuda"):
        return torch.device(d)
    if d.isdigit():
        return torch.device(f"cuda:{d}")
    return torch.device(device_str)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def safe_layer_flops(module: nn.Module, x) -> float:
    """Return layer GFLOPs when available, otherwise NaN."""
    if thop is None:
        return float("nan")
    try:
        module_copy = deepcopy(module)
        module_copy.eval()
        x_in = x.copy() if isinstance(x, list) else x
        flops = thop.profile(module_copy, inputs=[x_in], verbose=False)[0]
        return flops / 1e9 * 2
    except Exception:
        return float("nan")


def run_once(net: nn.Module, img: torch.Tensor, device: torch.device, compute_flops: bool = False) -> list[dict]:
    """Run one forward pass using the model's real top-level graph and collect per-layer stats."""
    y, x = [], img
    rows = []
    with torch.inference_mode():
        for m in net.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x_in = x.copy() if isinstance(x, list) else x
            sync(device)
            t0 = time.perf_counter()
            x = m(x)
            sync(device)
            dt_ms = (time.perf_counter() - t0) * 1000
            rows.append(
                {
                    "index": int(getattr(m, "i", -1)),
                    "module": getattr(m, "type", m.__class__.__name__),
                    "params": int(sum(p.numel() for p in m.parameters())),
                    "ms": float(dt_ms),
                    "gflops": safe_layer_flops(m, x_in) if compute_flops else float("nan"),
                }
            )
            y.append(x if getattr(m, "i", -1) in net.save else None)
    return rows


def configure_rtdetr_head(net: nn.Module, args: argparse.Namespace, device: torch.device) -> None:
    """Apply optional center-rerank runtime overrides directly on RT-DETR head."""
    if not hasattr(net, "model") or not net.model:
        return
    head = net.model[-1]
    if not hasattr(head, "query_rerank_mode"):
        return

    if args.query_rerank_mode:
        head.query_rerank_mode = args.query_rerank_mode
    if not hasattr(head, "enc_center_head") and getattr(head, "query_rerank_mode", "none") == "center":
        hidden_dim = getattr(head, "hidden_dim", None)
        if hidden_dim is not None:
            head.enc_center_head = nn.Linear(hidden_dim, 1).to(device)
    head.center_lambda_max = float(args.center_lambda_max)
    head.center_score_norm = str(args.center_score_norm)
    head.center_score_clip = float(args.center_score_clip)


def format_table(rows: list[dict]) -> str:
    header = f"{'idx':>4} {'latency(ms)':>12} {'time%':>8} {'params':>12} {'GFLOPs':>10}  module"
    lines = [header, "-" * len(header)]
    for r in rows:
        gflops = "nan" if math.isnan(r["gflops"]) else f"{r['gflops']:.3f}"
        lines.append(
            f"{r['index']:>4} {r['ms']:>12.3f} {r['time_pct']:>7.2f}% {r['params']:>12,d} {gflops:>10}  {r['module']}"
        )
    return "\n".join(lines)


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "module", "ms", "time_pct", "params", "gflops"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    wrapper = RTDETR(args.model)
    net = wrapper.model
    net.eval().to(device)
    configure_rtdetr_head(net, args, device)

    img = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    for _ in range(max(args.warmup, 0)):
        _ = run_once(net, img, device, compute_flops=False)

    timing_rows = []
    for i in range(max(args.repeats, 1)):
        rows_i = run_once(net, img, device, compute_flops=False)
        if i == 0:
            timing_rows = rows_i
        else:
            for r0, ri in zip(timing_rows, rows_i):
                r0["ms"] += ri["ms"]

    for r in timing_rows:
        r["ms"] /= max(args.repeats, 1)

    flop_rows = run_once(net, img, device, compute_flops=not args.no_layer_flops)
    for r, rf in zip(timing_rows, flop_rows):
        r["gflops"] = rf["gflops"]

    total_ms = sum(r["ms"] for r in timing_rows)
    total_params = sum(r["params"] for r in timing_rows)
    for r in timing_rows:
        r["time_pct"] = 100.0 * r["ms"] / max(total_ms, 1e-9)

    try:
        model_gflops = float(get_flops(net, imgsz=args.imgsz))
    except Exception:
        model_gflops = float("nan")

    print("\nModel summary")
    print(f"- checkpoint: {Path(args.model).resolve()}")
    print(f"- device: {device}")
    print(f"- input: 1x3x{args.imgsz}x{args.imgsz}")
    print(f"- repeats: {max(args.repeats, 1)} (warmup={max(args.warmup, 0)})")
    print(f"- total layers profiled: {len(timing_rows)}")
    print(f"- total params (sum of top-level layers): {total_params:,}")
    if math.isnan(model_gflops):
        print("- model GFLOPs: nan")
    else:
        print(f"- model GFLOPs: {model_gflops:.3f}")
    print(f"- total latency (sum of per-layer avg): {total_ms:.3f} ms")

    print("\nPer-layer profile (index order)")
    print(format_table(timing_rows))

    topk = sorted(timing_rows, key=lambda x: x["ms"], reverse=True)[: max(args.topk, 1)]
    print(f"\nTop-{len(topk)} slowest layers")
    print(format_table(topk))

    if args.save_csv:
        csv_path = Path(args.save_csv).resolve()
        save_csv(csv_path, timing_rows)
        print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    main()
