#!/usr/bin/env python3
"""Train RT-DETR on DIOR with optional history-aware decoder settings."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR


def str2bool(v: str | bool) -> bool:
    """Parse bool-like CLI values."""
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def resolve_path(path_like: str) -> str:
    """Resolve path from CWD first, then repo root."""
    p = Path(path_like)
    if p.exists():
        return str(p.resolve())

    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / path_like
    if candidate.exists():
        return str(candidate.resolve())

    return path_like


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR on DIOR with history-aware decoder options.")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt", help="RT-DETR model path or name.")
    parser.add_argument(
        "--data",
        type=str,
        default="/home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml",
        help="Dataset YAML path.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=512, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", type=str, default="0", help='Device, e.g. "0", "0,1", or "cpu".')
    parser.add_argument("--workers", type=int, default=1, help="Data loader workers.")
    parser.add_argument(
        "--cache",
        type=str,
        default="disk",
        choices=("disk", "ram", "false"),
        help="Cache mode: disk, ram, or false.",
    )
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training data to use.")
    parser.add_argument("--val", type=str2bool, default=True, help="Run validation after each epoch.")
    parser.add_argument("--project", type=str, default="runs/train", help="Output project directory.")
    parser.add_argument("--name", type=str, default="rtdetr_dior_history", help="Experiment name.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow existing project/name directory.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")

    # Keep existing center-rerank options available.
    parser.add_argument(
        "--query-rerank-mode",
        type=str,
        default="center",
        choices=("none", "center"),
        help="RT-DETR query rerank mode.",
    )
    parser.add_argument("--center-lambda-max", type=float, default=0.35, help="Max lambda for center reranking.")
    parser.add_argument("--center-lambda-warmup-epochs", type=int, default=10, help="Warmup epochs for center lambda.")
    parser.add_argument(
        "--center-score-norm",
        type=str,
        default="zscore_image",
        choices=("zscore_image", "none"),
        help="Center score normalization mode.",
    )
    parser.add_argument("--center-score-clip", type=float, default=6.0, help="Clip for fused rerank score.")

    # New history-aware decoder options.
    parser.add_argument("--use-history-fusion", type=str2bool, default=True, help="Enable history-aware decoder fusion.")
    parser.add_argument(
        "--history-window",
        type=int,
        default=3,
        help="Number of recent history states to fuse; <=0 means use all available states.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = False if args.cache == "false" else args.cache
    model = RTDETR(resolve_path(args.model))
    model.train(
        data=resolve_path(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=cache,
        fraction=args.fraction,
        val=args.val,
        project=resolve_path(args.project),
        name=args.name,
        patience=args.patience,
        query_rerank_mode=args.query_rerank_mode,
        center_lambda_max=args.center_lambda_max,
        center_lambda_warmup_epochs=args.center_lambda_warmup_epochs,
        center_score_norm=args.center_score_norm,
        center_score_clip=args.center_score_clip,
        use_history_fusion=args.use_history_fusion,
        history_window=args.history_window,
        exist_ok=args.exist_ok,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
