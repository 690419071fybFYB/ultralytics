#!/usr/bin/env python3
"""Run baseline and AGV2 center-RTDETR training sequentially with matched settings."""

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
    """Parse CLI args for baseline vs AGV2 comparison training."""
    parser = argparse.ArgumentParser(description="Train baseline and AGV2 center RT-DETR variants.")
    parser.add_argument(
        "--data",
        type=str,
        default="/home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml",
        help="Dataset YAML path.",
    )
    parser.add_argument("--pretrained", type=str, default="rtdetr-l.pt", help="Pretrained checkpoint or model name.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=512, help="Training image size.")
    parser.add_argument("--batch", type=int, default=12, help="Batch size.")
    parser.add_argument("--device", type=str, default="0", help='Device, e.g. "0", "0,1", or "cpu".')
    parser.add_argument("--workers", type=int, default=1, help="Data loader workers.")
    parser.add_argument(
        "--cache",
        type=str,
        default="disk",
        choices=("disk", "ram", "false"),
        help="Cache mode: disk, ram, or false.",
    )
    parser.add_argument("--project", type=str, default="runs/train", help="Output project directory.")
    parser.add_argument("--name-prefix", type=str, default="rtdetr_l_center_headcmp", help="Shared run-name prefix.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training set to use.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow existing project/name directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned runs and exit.")

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
        default="add",
        choices=("add", "geom"),
        help="Fusion strategy for cls rank and center rank.",
    )
    parser.add_argument("--center-lambda-max", type=float, default=0.25, help="Max lambda for center reranking.")
    parser.add_argument("--center-lambda-warmup-epochs", type=int, default=10, help="Warmup epochs for center lambda.")
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


def build_train_kwargs(args: argparse.Namespace, variant: str) -> dict:
    """Build train() kwargs shared by baseline and AGV2 runs."""
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
        "name": f"{args.name_prefix}_{variant}",
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


def build_run_configs(args: argparse.Namespace) -> list[dict]:
    """Build the ordered baseline and AGV2 training plan."""
    return [
        {
            "variant": "baseline",
            "model": "ultralytics/cfg/models/rt-detr/rtdetr-l-center.yaml",
            "train_kwargs": build_train_kwargs(args, "baseline"),
        },
        {
            "variant": "agv2",
            "model": "ultralytics/cfg/models/rt-detr/rtdetr-l-center-agv2.yaml",
            "train_kwargs": build_train_kwargs(args, "agv2"),
        },
    ]


def main(argv: list[str] | None = None) -> None:
    """Launch the baseline and AGV2 runs sequentially."""
    args = parse_args(argv)
    runs = build_run_configs(args)

    if args.dry_run:
        for run in runs:
            print(f"[plan] variant={run['variant']} model={run['model']} name={run['train_kwargs']['name']}")
        return

    for run in runs:
        model = RTDETR(run["model"])
        model.train(**run["train_kwargs"])


if __name__ == "__main__":
    main()
