#!/usr/bin/env python3
"""Run Phase2 quota models on test split and summarize results."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from ultralytics import RTDETR


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EvalResult:
    model_tag: str
    weight_path: str
    test_map50_95: float
    test_map50: float
    test_map75: float
    preprocess_ms: float
    infer_ms: float
    postprocess_ms: float
    latency_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase2 quota test sweep (seed=0).")
    parser.add_argument(
        "--data",
        type=str,
        default="/home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml",
        help="Dataset YAML path.",
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cache", type=str, default="disk")
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default="runs/analysis/phase2_quota_sweep_seed0",
        help="Directory for result artifacts.",
    )
    parser.add_argument(
        "--baseline-weight",
        type=str,
        default="/home/fyb/mydir/ultralytics/runs/detect/runs/train/phase1_center_lam035_seed0_e40/weights/best.pt",
    )
    return parser.parse_args()


def eval_one(weight: Path, model_tag: str, args: argparse.Namespace) -> EvalResult:
    model = RTDETR(str(weight))
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        plots=False,
        save_json=False,
        verbose=False,
    )
    speed = metrics.speed or {}
    preprocess_ms = float(speed.get("preprocess", math.nan))
    infer_ms = float(speed.get("inference", math.nan))
    postprocess_ms = float(speed.get("postprocess", math.nan))
    return EvalResult(
        model_tag=model_tag,
        weight_path=str(weight),
        test_map50_95=float(metrics.box.map),
        test_map50=float(metrics.box.map50),
        test_map75=float(metrics.box.map75),
        preprocess_ms=preprocess_ms,
        infer_ms=infer_ms,
        postprocess_ms=postprocess_ms,
        latency_ms=preprocess_ms + infer_ms + postprocess_ms,
    )


def write_summary(path: Path, rows: list[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model_tag",
                "weight_path",
                "test_map50_95",
                "test_map50",
                "test_map75",
                "preprocess_ms",
                "infer_ms",
                "postprocess_ms",
                "latency_ms",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.model_tag,
                    r.weight_path,
                    f"{r.test_map50_95:.6f}",
                    f"{r.test_map50:.6f}",
                    f"{r.test_map75:.6f}",
                    f"{r.preprocess_ms:.6f}",
                    f"{r.infer_ms:.6f}",
                    f"{r.postprocess_ms:.6f}",
                    f"{r.latency_ms:.6f}",
                ]
            )


def write_md(path: Path, baseline: EvalResult, phase2: list[EvalResult], best: EvalResult) -> None:
    def fmt_pct(v: float) -> str:
        return f"{v * 100:+.3f}%"

    delta_map = best.test_map50_95 - baseline.test_map50_95
    lat_change = (best.latency_ms - baseline.latency_ms) / baseline.latency_ms if baseline.latency_ms > 0 else math.nan
    fps_base = 1000.0 / baseline.latency_ms if baseline.latency_ms > 0 else math.nan
    fps_best = 1000.0 / best.latency_ms if best.latency_ms > 0 else math.nan
    fps_drop = (fps_base - fps_best) / fps_base if fps_base > 0 else math.nan

    lines = []
    lines.append("# Phase2 Test Comparison")
    lines.append("")
    lines.append(f"- Baseline: `{baseline.model_tag}`")
    lines.append(f"- Best Phase2: `{best.model_tag}`")
    lines.append("")
    lines.append("## Test Summary")
    lines.append("")
    lines.append("| model_tag | mAP50-95 | mAP50 | mAP75 | latency(ms/im) |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| {baseline.model_tag} | {baseline.test_map50_95:.6f} | {baseline.test_map50:.6f} | "
        f"{baseline.test_map75:.6f} | {baseline.latency_ms:.6f} |"
    )
    for r in phase2:
        lines.append(
            f"| {r.model_tag} | {r.test_map50_95:.6f} | {r.test_map50:.6f} | {r.test_map75:.6f} | {r.latency_ms:.6f} |"
        )
    lines.append("")
    lines.append("## Best vs Baseline")
    lines.append("")
    lines.append(f"- Delta mAP50-95: `{delta_map:+.6f}` ({delta_map * 100:+.3f} points)")
    lines.append(f"- Latency change: `{fmt_pct(lat_change)}`")
    lines.append(f"- FPS drop: `{fmt_pct(fps_drop)}`")
    lines.append("")
    lines.append("## Gate Check")
    lines.append("")
    lines.append(f"- Accuracy gate (>= +0.3 points): `{delta_map * 100 >= 0.3}`")
    lines.append(f"- Efficiency gate (latency<=+10% or FPS drop<=8%): `{(lat_change <= 0.10) or (fps_drop <= 0.08)}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    analysis_dir = (REPO_ROOT / args.analysis_dir).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    phase2_models: list[tuple[str, Path]] = [
        (
            "phase2_quota_none_lam035_seed0_e40",
            REPO_ROOT / "runs/detect/runs/train/phase2_quota_none_lam035_seed0_e40/weights/best.pt",
        ),
        (
            "phase2_quota_50_30_20_lam035_seed0_e40",
            REPO_ROOT / "runs/detect/runs/train/phase2_quota_50_30_20_lam035_seed0_e40/weights/best.pt",
        ),
        (
            "phase2_quota_60_25_15_lam035_seed0_e40",
            REPO_ROOT / "runs/detect/runs/train/phase2_quota_60_25_15_lam035_seed0_e40/weights/best.pt",
        ),
        (
            "phase2_quota_40_35_25_lam035_seed0_e40",
            REPO_ROOT / "runs/detect/runs/train/phase2_quota_40_35_25_lam035_seed0_e40/weights/best.pt",
        ),
    ]

    baseline_weight = Path(args.baseline_weight).expanduser().resolve()
    if not baseline_weight.exists():
        raise FileNotFoundError(f"Baseline weight not found: {baseline_weight}")

    for tag, w in phase2_models:
        if not w.exists():
            raise FileNotFoundError(f"Phase2 weight not found for {tag}: {w}")

    baseline = eval_one(baseline_weight, "phase1_center_lam035_seed0_e40", args)
    phase2_results = [eval_one(w, tag, args) for tag, w in phase2_models]
    best_phase2 = max(phase2_results, key=lambda x: (x.test_map50_95, x.test_map50, -x.latency_ms))

    write_summary(analysis_dir / "phase2_test_summary.csv", [baseline, *phase2_results])
    write_md(analysis_dir / "comparison_vs_phase1.md", baseline, phase2_results, best_phase2)

    print("Done. Artifacts:")
    print(f"- {analysis_dir / 'phase2_test_summary.csv'}")
    print(f"- {analysis_dir / 'comparison_vs_phase1.md'}")


if __name__ == "__main__":
    main()
