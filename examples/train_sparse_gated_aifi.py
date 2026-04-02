#!/usr/bin/env python3
"""Train RT-DETR with Sparse-Gated AIFI on DIOR or another dataset YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR
from ultralytics.nn.modules import SparseGatedAIFI


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
    """Parse command-line arguments for SG-AIFI training."""
    parser = argparse.ArgumentParser(description="Train RT-DETR with Sparse-Gated AIFI.")
    parser.add_argument(
        "--model",
        type=str,
        default="ultralytics/cfg/models/rt-detr/rtdetr-l-sgaifi.yaml",
        help="Model YAML path.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="rtdetr-l.pt",
        help="Pretrained checkpoint or model name used to initialize weights.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml",
        help="Dataset YAML path.",
    )
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
    parser.add_argument("--name", type=str, default="rtdetr_l_sgaifi", help="Experiment name.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training set to use.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP training explicitly.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow existing project/name directory.")

    parser.add_argument("--local-window", type=int, default=3, help="Local attention window size.")
    parser.add_argument("--global-pool", type=int, default=2, help="Spatial pool factor for global tokens.")
    parser.add_argument(
        "--disable-local",
        action="store_true",
        help="Disable the local attention branch for ablation.",
    )
    parser.add_argument(
        "--disable-global",
        action="store_true",
        help="Disable the global attention branch for ablation.",
    )
    parser.add_argument(
        "--disable-gate",
        action="store_true",
        help="Disable token-wise gate and use fixed 0.5 fusion when both branches are enabled.",
    )
    return parser.parse_args()


def configure_sparse_gated_aifi(model: RTDETR, args: argparse.Namespace) -> None:
    """Apply runtime SG-AIFI settings to all SparseGatedAIFI modules in the model."""
    enable_local = not args.disable_local
    enable_global = not args.disable_global
    if not enable_local and not enable_global:
        raise ValueError("At least one of local/global attention branches must remain enabled.")

    found = False
    for module in model.model.modules():
        if isinstance(module, SparseGatedAIFI):
            module.local_window = args.local_window
            module.global_pool = args.global_pool
            module.enable_local = enable_local
            module.enable_global = enable_global
            module.enable_gate = (not args.disable_gate) and enable_local and enable_global
            found = True

    if not found:
        raise RuntimeError("No SparseGatedAIFI module was found in the loaded model.")


def main() -> None:
    """Build the model, inject SG-AIFI runtime settings, and launch training."""
    args = parse_args()
    cache = False if args.cache == "false" else args.cache

    model = RTDETR(args.model)
    configure_sparse_gated_aifi(model, args)

    model.train(
        data=resolve_data_yaml(args.data),
        pretrained=args.pretrained,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=cache,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        fraction=args.fraction,
        amp=args.amp,
        exist_ok=args.exist_ok,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
