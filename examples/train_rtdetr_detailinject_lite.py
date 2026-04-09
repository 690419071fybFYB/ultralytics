#!/usr/bin/env python3
"""Train RT-DETR DetailInjectLite variant with configurable Ultralytics arguments."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR


def resolve_path(path_str: str) -> str:
    """Resolve a filesystem path from user input, CWD, or repository root."""
    path = Path(path_str).expanduser()
    if path.exists():
        return str(path.resolve())

    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / path_str
    if candidate.exists():
        return str(candidate.resolve())

    return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for RT-DETR DetailInjectLite training."""
    parser = argparse.ArgumentParser(description="Train RT-DETR DetailInjectLite variant.")
    parser.add_argument(
        "--model",
        type=str,
        default="~/fyb/projects/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l-detailinject-lite.yaml",
        help="Model YAML path.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="/home/gpcvgroup/fyb/projects/ultralytics/rtdetr-l.pt",
        help="Pretrained checkpoint path.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/home/gpcvgroup/fyb/datasets/DIOR_COCO_ULTRA/dior_ultralytics.yaml",
        help="Dataset YAML path.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
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
    parser.add_argument("--project", type=str, default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", type=str, default="detailinject_lite", help="Experiment name.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training set to use.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow existing project/name directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit.")
    parser.add_argument(
        "--query-rerank-mode",
        type=str,
        default="center",
        choices=("none", "center"),
        help="RT-DETR query rerank mode.",
    )
    parser.add_argument("--center-lambda-max", type=float, default=0.35, help="Max lambda for center reranking.")
    parser.add_argument(
        "--center-lambda-warmup-epochs",
        type=int,
        default=10,
        help="Epochs to warm up center lambda.",
    )
    parser.add_argument(
        "--center-score-norm",
        type=str,
        default="zscore_image",
        choices=("zscore_image", "none"),
        help="Center score normalization mode.",
    )
    parser.add_argument("--center-score-clip", type=float, default=6.0, help="Clip for fused rerank score.")
    parser.add_argument("--center-loss-weight", type=float, default=0.5, help="Center loss weight.")
    parser.add_argument("--center-pos-alpha", type=float, default=4.0, help="Positive reweight alpha in center loss.")
    parser.add_argument("--center-empty-scale", type=float, default=0.25, help="Loss scale for images without GT.")
    parser.add_argument(
        "--center-target",
        type=str,
        default="box_centerness",
        choices=("box_centerness",),
        help="Center supervision target type.",
    )
    parser.add_argument(
        "--center-multi-gt-rule",
        type=str,
        default="max",
        choices=("max",),
        help="Multi-GT assignment rule for center target.",
    )
    return parser.parse_args(argv)


def build_train_kwargs(args: argparse.Namespace) -> dict:
    """Build a stable train() kwargs dict for RT-DETR DetailInjectLite runs."""
    cache = False if args.cache == "false" else args.cache
    return {
        "data": resolve_path(args.data),
        "pretrained": resolve_path(args.pretrained),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "cache": cache,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "seed": args.seed,
        "fraction": args.fraction,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
        "query_rerank_mode": args.query_rerank_mode,
        "center_lambda_max": args.center_lambda_max,
        "center_lambda_warmup_epochs": args.center_lambda_warmup_epochs,
        "center_score_norm": args.center_score_norm,
        "center_score_clip": args.center_score_clip,
        "center_loss_weight": args.center_loss_weight,
        "center_pos_alpha": args.center_pos_alpha,
        "center_empty_scale": args.center_empty_scale,
        "center_target": args.center_target,
        "center_multi_gt_rule": args.center_multi_gt_rule,
    }


def main(argv: list[str] | None = None) -> None:
    """Launch RT-DETR DetailInjectLite training."""
    args = parse_args(argv)
    model_path = resolve_path(args.model)
    train_kwargs = build_train_kwargs(args)

    if args.dry_run:
        print(f"[plan] model={model_path}")
        for key, value in train_kwargs.items():
            print(f"[plan] {key}={value}")
        return

    model = RTDETR(model_path)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
