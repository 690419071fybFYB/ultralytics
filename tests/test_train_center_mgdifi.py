# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    """Load the MGDIFI center training script as a module."""
    path = Path(__file__).resolve().parents[1] / "examples" / "train_center_mgdifi.py"
    spec = importlib.util.spec_from_file_location("train_center_mgdifi", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_build_train_kwargs_defaults_to_center_mgdifi_yaml():
    """The training script should default to the center+MGDIFI RT-DETR config."""
    module = load_module()
    args = module.parse_args(
        [
            "--data",
            "coco8.yaml",
            "--epochs",
            "5",
            "--imgsz",
            "320",
            "--batch",
            "4",
            "--device",
            "cpu",
        ]
    )

    train_kwargs = module.build_train_kwargs(args)

    assert args.model == "ultralytics/cfg/models/rt-detr/rtdetr-l-center-mgdifi.yaml"
    assert train_kwargs["epochs"] == 5
    assert train_kwargs["imgsz"] == 320
    assert train_kwargs["batch"] == 4
    assert train_kwargs["device"] == "cpu"
    assert train_kwargs["query_rerank_mode"] == "center"
    assert train_kwargs["center_fusion_strategy"] in {"add", "geom"}


def test_default_run_name_mentions_center_mgdifi():
    """Auto-generated run names should make the MGDIFI center variant explicit."""
    module = load_module()
    args = module.parse_args(["--data", "coco8.yaml", "--seed", "7"])

    name = module.default_run_name(args)

    assert "center_mgdifi" in name
    assert name.endswith("_seed7")
