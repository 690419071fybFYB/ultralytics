#!/usr/bin/env python3
"""Train CenterRTDETRDecoder + MGDIFI with configurable Ultralytics arguments."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR


def resolve_data_yaml(data: str) -> str:
    """Resolve dataset YAML path from CWD or repository root."""
    path = Path(data)
    if path.exists():
        return str(path.resolve())

    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / data
    if candidate.exists():
        return str(candidate.resolve())

    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for CenterRTDETRDecoder + MGDIFI training."""
    parser = argparse.ArgumentParser(description="Train RT-DETR center+MGDIFI variant.")
    parser.add_argument(
        "--model",
        type=str,
        default="ultralytics/cfg/models/rt-detr/rtdetr-l-center-mgdifi.yaml",
        help="Model YAML path.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="rtdetr-l.pt",
        help="Pretrained checkpoint or model name.",
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
    parser.add_argument("--project", type=str, default="", help="Output project directory.")
    parser.add_argument("--name", type=str, default="", help="Experiment name. Auto-generated when empty.")
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
    parser.add_argument(
        "--center-fusion-strategy",
        type=str,
        default="geom",
        choices=("add", "geom"),
        help="Fusion strategy for cls rank and center rank.",
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
    parser.add_argument("--center-pos-alpha", type=float, default=4.0, help="Positive reweight alpha.")
    parser.add_argument("--center-empty-scale", type=float, default=0.25, help="Loss scale for empty-GT images.")
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


def default_run_name(args: argparse.Namespace) -> str:
    """Generate a compact run name for the center+MGDIFI variant."""
    fusion = args.center_fusion_strategy
    lam = str(args.center_lambda_max).replace(".", "")
    return f"center_mgdifi_{fusion}_lam{lam}_seed{args.seed}"


def build_train_kwargs(args: argparse.Namespace) -> dict:
    """Build a stable train() kwargs dict for RT-DETR center+MGDIFI runs."""
    cache = False if args.cache == "false" else args.cache
    return {
        "data": resolve_data_yaml(args.data),
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "cache": cache,
        "project": args.project,
        "name": args.name or default_run_name(args),
        "patience": args.patience,
        "seed": args.seed,
        "fraction": args.fraction,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
        "query_rerank_mode": args.query_rerank_mode,
        "center_fusion_strategy": args.center_fusion_strategy,
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
    """Launch CenterRTDETRDecoder + MGDIFI training."""
    args = parse_args(argv)
    train_kwargs = build_train_kwargs(args)

    if args.dry_run:
        print(f"[plan] model={args.model}")
        for key, value in train_kwargs.items():
            print(f"[plan] {key}={value}")
        return

    model = RTDETR(args.model)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
