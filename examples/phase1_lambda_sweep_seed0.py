#!/usr/bin/env python3
"""Phase 1 lambda_max sweep for RT-DETR center-aware query reranking (seed=0)."""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ultralytics import RTDETR


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "examples" / "train_rtdetr_dior.py"

OOM_PATTERNS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)

DATALOADER_PATTERNS = (
    "dataloader worker",
    "worker (pid",
    "too many open files",
    "broken pipe",
    "bus error",
)


@dataclass
class SweepRun:
    lambda_max: float
    run_name: str


@dataclass
class TrainResult:
    lambda_max: float
    run_name: str
    run_dir: str
    best_epoch: int | None
    val_map50_95: float | None
    val_map50: float | None
    val_precision: float | None
    val_recall: float | None
    status: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 lambda_max sweep automation (seed=0).")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt")
    parser.add_argument("--data", type=str, default="/home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml")
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="/home/fyb/mydir/ultralytics/runs/detect/runs/train/rtdetr_dior5/weights/best.pt",
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", type=str, default="disk")
    parser.add_argument("--center-loss-weight", type=float, default=0.5)
    parser.add_argument("--center-pos-alpha", type=float, default=4.0)
    parser.add_argument("--center-empty-scale", type=float, default=0.25)
    parser.add_argument("--center-score-norm", type=str, default="zscore_image")
    parser.add_argument("--center-score-clip", type=float, default=6.0)
    parser.add_argument("--center-lambda-warmup-epochs", type=int, default=10)
    parser.add_argument("--lambdas", type=str, default="0.15,0.25,0.35")
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default="runs/analysis/phase1_lambda_sweep_seed0",
        help="Output directory for csv/md reports and logs.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even when results.csv already exists for a lambda run.",
    )
    return parser.parse_args()


def lambda_to_tag(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def parse_lambdas(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("No lambda values were parsed from --lambdas.")
    return values


def get_run_dir(run_name: str) -> Path:
    candidates = [
        REPO_ROOT / "runs" / "detect" / "runs" / "train" / run_name,
        REPO_ROOT / "runs" / "train" / run_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def get_results_csv(run_name: str) -> Path:
    return get_run_dir(run_name) / "results.csv"


def parse_best_metrics(results_csv: Path) -> tuple[int, float, float, float, float]:
    with results_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty results.csv: {results_csv}")

    keys = rows[0].keys()
    epoch_key = next(k for k in keys if k == "epoch")
    map95_key = next(k for k in keys if "metrics/mAP50-95(B)" in k)
    map50_key = next(k for k in keys if "metrics/mAP50(B)" in k)
    p_key = next(k for k in keys if "metrics/precision(B)" in k)
    r_key = next(k for k in keys if "metrics/recall(B)" in k)

    best_row = max(rows, key=lambda row: (float(row[map95_key]), float(row[map50_key])))
    best_epoch = int(float(best_row[epoch_key]))
    best_map95 = float(best_row[map95_key])
    best_map50 = float(best_row[map50_key])
    best_p = float(best_row[p_key])
    best_r = float(best_row[r_key])
    return best_epoch, best_map95, best_map50, best_p, best_r


def classify_failure(log_file: Path) -> str:
    if not log_file.exists():
        return "unknown"
    text = log_file.read_text(encoding="utf-8", errors="ignore").lower()
    if any(pattern in text for pattern in OOM_PATTERNS):
        return "oom"
    if any(pattern in text for pattern in DATALOADER_PATTERNS):
        return "dataloader"
    return "other"


def run_train_once(cmd: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        f.write("$ " + shlex.join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, check=False)
    return proc.returncode


def train_with_retries(
    run: SweepRun,
    args: argparse.Namespace,
    logs_dir: Path,
    force_retrain: bool,
) -> TrainResult:
    run_dir = get_run_dir(run.run_name)
    results_csv = get_results_csv(run.run_name)
    if results_csv.exists() and not force_retrain:
        best_epoch, val_map95, val_map50, val_p, val_r = parse_best_metrics(results_csv)
        return TrainResult(
            lambda_max=run.lambda_max,
            run_name=run.run_name,
            run_dir=str(run_dir),
            best_epoch=best_epoch,
            val_map50_95=val_map95,
            val_map50=val_map50,
            val_precision=val_p,
            val_recall=val_r,
            status="reused",
        )

    batch = args.batch
    workers = args.workers
    attempted_cfgs: set[tuple[int, int]] = set()
    last_reason = "unknown"
    last_error = ""

    while True:
        cfg = (batch, workers)
        if cfg in attempted_cfgs:
            break
        attempted_cfgs.add(cfg)

        log_file = logs_dir / f"{run.run_name}_b{batch}_w{workers}.log"
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--model",
            args.model,
            "--data",
            args.data,
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(batch),
            "--device",
            args.device,
            "--workers",
            str(workers),
            "--cache",
            args.cache,
            "--project",
            "runs/train",
            "--name",
            run.run_name,
            "--patience",
            str(args.patience),
            "--query-rerank-mode",
            "center",
            "--center-lambda-max",
            str(run.lambda_max),
            "--center-lambda-warmup-epochs",
            str(args.center_lambda_warmup_epochs),
            "--center-score-norm",
            args.center_score_norm,
            "--center-score-clip",
            str(args.center_score_clip),
            "--center-loss-weight",
            str(args.center_loss_weight),
            "--center-pos-alpha",
            str(args.center_pos_alpha),
            "--center-empty-scale",
            str(args.center_empty_scale),
            "--center-target",
            "box_centerness",
            "--center-multi-gt-rule",
            "max",
            "--seed",
            str(args.seed),
            "--exist-ok",
        ]

        rc = run_train_once(cmd, log_file)
        if rc == 0 and results_csv.exists():
            best_epoch, val_map95, val_map50, val_p, val_r = parse_best_metrics(results_csv)
            return TrainResult(
                lambda_max=run.lambda_max,
                run_name=run.run_name,
                run_dir=str(get_run_dir(run.run_name)),
                best_epoch=best_epoch,
                val_map50_95=val_map95,
                val_map50=val_map50,
                val_precision=val_p,
                val_recall=val_r,
                status="success",
            )

        last_reason = classify_failure(log_file)
        last_error = f"rc={rc}, reason={last_reason}, log={log_file}"

        if last_reason == "oom":
            if batch == 16:
                batch = 12
                continue
            if batch == 12:
                batch = 8
                continue
            break

        if last_reason == "dataloader" and workers != 0:
            workers = 0
            continue

        break

    return TrainResult(
        lambda_max=run.lambda_max,
        run_name=run.run_name,
        run_dir=str(get_run_dir(run.run_name)),
        best_epoch=None,
        val_map50_95=None,
        val_map50=None,
        val_precision=None,
        val_recall=None,
        status=f"failed_{last_reason}",
        error=last_error,
    )


def write_sweep_summary(path: Path, rows: list[TrainResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "lambda_max",
                "run_name",
                "run_dir",
                "best_epoch",
                "val_map50_95",
                "val_map50",
                "val_precision",
                "val_recall",
                "status",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    f"{r.lambda_max:.2f}",
                    r.run_name,
                    r.run_dir,
                    r.best_epoch if r.best_epoch is not None else "",
                    f"{r.val_map50_95:.6f}" if r.val_map50_95 is not None else "",
                    f"{r.val_map50:.6f}" if r.val_map50 is not None else "",
                    f"{r.val_precision:.6f}" if r.val_precision is not None else "",
                    f"{r.val_recall:.6f}" if r.val_recall is not None else "",
                    r.status,
                ]
            )


def select_best_lambda(results: list[TrainResult]) -> TrainResult | None:
    successful = [r for r in results if r.val_map50_95 is not None and r.val_map50 is not None]
    if not successful:
        return None
    return sorted(successful, key=lambda r: (-r.val_map50_95, -r.val_map50, r.lambda_max))[0]


def eval_model_test(weights: str, args: argparse.Namespace, eval_name: str) -> dict[str, float]:
    model = RTDETR(weights)
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project="runs/eval",
        name=eval_name,
        exist_ok=True,
        plots=False,
        save_json=False,
    )
    speed = metrics.speed if hasattr(metrics, "speed") else {}
    return {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "preprocess_ms": float(speed.get("preprocess", math.nan)),
        "infer_ms": float(speed.get("inference", math.nan)),
        "postprocess_ms": float(speed.get("postprocess", math.nan)),
    }


def write_best_test_summary(path: Path, selected_lambda: float, metrics: dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "selected_lambda",
                "test_map50_95",
                "test_map50",
                "test_map75",
                "preprocess_ms",
                "infer_ms",
                "postprocess_ms",
            ]
        )
        writer.writerow(
            [
                f"{selected_lambda:.2f}",
                f"{metrics['map50_95']:.6f}",
                f"{metrics['map50']:.6f}",
                f"{metrics['map75']:.6f}",
                f"{metrics['preprocess_ms']:.6f}",
                f"{metrics['infer_ms']:.6f}",
                f"{metrics['postprocess_ms']:.6f}",
            ]
        )


def total_latency_ms(metrics: dict[str, float]) -> float:
    return metrics["preprocess_ms"] + metrics["infer_ms"] + metrics["postprocess_ms"]


def write_comparison_markdown(
    path: Path,
    train_results: list[TrainResult],
    best: TrainResult | None,
    best_test: dict[str, float] | None,
    baseline_test: dict[str, float] | None,
) -> None:
    lines = []
    lines.append("# Phase 1 lambda sweep (seed=0)")
    lines.append("")

    success_count = sum(1 for r in train_results if r.val_map50_95 is not None)
    lines.append(f"- Successful lambda runs: {success_count}/{len(train_results)}")

    if best is None or best_test is None or baseline_test is None:
        lines.append("- No valid best-lambda test comparison available.")
        if success_count == 1:
            lines.append("- Confidence note: only one lambda run succeeded, conclusion is low-confidence.")
        lines.append("")
        lines.append("## Recommendation")
        lines.append("- Do not enter Phase 2 yet. Fix failures and rerun this sweep.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    if success_count == 1:
        lines.append("- Confidence note: only one lambda run succeeded, conclusion is low-confidence.")
    lines.append(f"- Selected lambda: `{best.lambda_max:.2f}` from run `{best.run_name}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Baseline | Best-lambda | Delta |")
    lines.append("|---|---:|---:|---:|")

    delta_map95 = best_test["map50_95"] - baseline_test["map50_95"]
    delta_map50 = best_test["map50"] - baseline_test["map50"]
    delta_map75 = best_test["map75"] - baseline_test["map75"]

    lines.append(
        f"| test mAP50-95 | {baseline_test['map50_95']:.6f} | {best_test['map50_95']:.6f} | {delta_map95:+.6f} |"
    )
    lines.append(f"| test mAP50 | {baseline_test['map50']:.6f} | {best_test['map50']:.6f} | {delta_map50:+.6f} |")
    lines.append(f"| test mAP75 | {baseline_test['map75']:.6f} | {best_test['map75']:.6f} | {delta_map75:+.6f} |")

    lat_base = total_latency_ms(baseline_test)
    lat_best = total_latency_ms(best_test)
    lat_change_pct = (lat_best - lat_base) / max(lat_base, 1e-9) * 100.0
    fps_base = 1000.0 / max(lat_base, 1e-9)
    fps_best = 1000.0 / max(lat_best, 1e-9)
    fps_drop_pct = (fps_base - fps_best) / max(fps_base, 1e-9) * 100.0

    lines.append(f"| latency(ms/im) | {lat_base:.6f} | {lat_best:.6f} | {lat_change_pct:+.3f}% |")
    lines.append(f"| FPS | {fps_base:.3f} | {fps_best:.3f} | {(-fps_drop_pct):+.3f}% |")
    lines.append("")

    pass_map = delta_map95 >= 0.003
    pass_eff = lat_change_pct <= 10.0 or fps_drop_pct <= 8.0
    overall = pass_map and pass_eff

    lines.append("## Gate Check")
    lines.append(f"- Delta mAP50-95 >= +0.3 points: `{pass_map}` (delta={delta_map95 * 100:.3f} points)")
    lines.append(
        f"- Efficiency gate (latency<=+10% or FPS drop<=8%): `{pass_eff}` "
        f"(latency={lat_change_pct:+.3f}%, fps_drop={fps_drop_pct:+.3f}%)"
    )
    lines.append(f"- Overall quick gate: `{overall}`")
    lines.append("")
    lines.append("## Recommendation")
    if overall and success_count >= 2:
        lines.append("- Continue to Phase 2 (per-level quota).")
    elif overall:
        lines.append("- Technically passes gate, but confidence is low. Run missing lambda retries before Phase 2.")
    else:
        lines.append("- Do not enter Phase 2 yet. Recheck lambda/loss settings first.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    analysis_dir = (REPO_ROOT / args.analysis_dir).resolve()
    logs_dir = analysis_dir / "logs"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    lambda_values = parse_lambdas(args.lambdas)
    runs = [
        SweepRun(
            lambda_max=value,
            run_name=f"phase1_center_lam{lambda_to_tag(value)}_seed{args.seed}_e{args.epochs}",
        )
        for value in lambda_values
    ]

    train_results = [train_with_retries(run, args, logs_dir, force_retrain=args.force_retrain) for run in runs]
    write_sweep_summary(analysis_dir / "sweep_train_summary.csv", train_results)

    best = select_best_lambda(train_results)
    best_test_metrics = None
    baseline_test_metrics = None

    if best is not None:
        best_weights = get_run_dir(best.run_name) / "weights" / "best.pt"
        if best_weights.exists():
            best_test_metrics = eval_model_test(str(best_weights), args, "phase1_sweep_best_lambda_test")
            write_best_test_summary(analysis_dir / "best_lambda_test_summary.csv", best.lambda_max, best_test_metrics)

        baseline_weights = Path(args.baseline_model)
        if baseline_weights.exists():
            baseline_test_metrics = eval_model_test(str(baseline_weights), args, "phase1_sweep_baseline_test")

    write_comparison_markdown(
        analysis_dir / "comparison_vs_baseline.md",
        train_results,
        best,
        best_test_metrics,
        baseline_test_metrics,
    )

    print(f"Analysis outputs written to: {analysis_dir}")
    print(f"- {analysis_dir / 'sweep_train_summary.csv'}")
    if (analysis_dir / "best_lambda_test_summary.csv").exists():
        print(f"- {analysis_dir / 'best_lambda_test_summary.csv'}")
    print(f"- {analysis_dir / 'comparison_vs_baseline.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
