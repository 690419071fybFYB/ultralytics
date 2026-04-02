#!/usr/bin/env python3
"""Train RT-DETR on DIOR with Ultralytics."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR on DIOR dataset.")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt", help="RT-DETR model path or name.")
    parser.add_argument(
        "--data",
        type=str,
        default="/home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml",
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
    parser.add_argument("--project", type=str, default="runs/train", help="Output project directory.")
    parser.add_argument("--name", type=str, default="rtdetr_dior", help="Experiment name.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
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
    parser.add_argument("--center-lambda-max", type=float, default=0.25, help="Max lambda for center reranking.")
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
    parser.add_argument("--exist-ok", action="store_true", help="Allow existing project/name directory.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = False if args.cache == "false" else args.cache
    model = RTDETR(args.model)
    model.train(
        data=resolve_data_yaml(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=cache,
        project=args.project,
        name=args.name,
        patience=args.patience,
        query_rerank_mode=args.query_rerank_mode,
        center_fusion_strategy=args.center_fusion_strategy,
        center_lambda_max=args.center_lambda_max,
        center_lambda_warmup_epochs=args.center_lambda_warmup_epochs,
        center_score_norm=args.center_score_norm,
        center_score_clip=args.center_score_clip,
        center_loss_weight=args.center_loss_weight,
        center_pos_alpha=args.center_pos_alpha,
        center_empty_scale=args.center_empty_scale,
        center_target=args.center_target,
        center_multi_gt_rule=args.center_multi_gt_rule,
        exist_ok=args.exist_ok,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
