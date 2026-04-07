# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    """Load the comparison training script as a module."""
    path = Path(__file__).resolve().parents[1] / "examples" / "train_center_decoder_agv2_compare.py"
    spec = importlib.util.spec_from_file_location("train_center_decoder_agv2_compare", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_build_run_configs_returns_baseline_and_agv2_variants():
    """The comparison script should produce two runs with locked training settings."""
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
            "--name-prefix",
            "demo",
        ]
    )

    runs = module.build_run_configs(args)

    assert [run["variant"] for run in runs] == ["baseline", "agv2"]
    assert runs[0]["model"] == "ultralytics/cfg/models/rt-detr/rtdetr-l-center.yaml"
    assert runs[1]["model"] == "ultralytics/cfg/models/rt-detr/rtdetr-l-center-agv2.yaml"
    assert runs[0]["train_kwargs"]["epochs"] == 5
    assert runs[0]["train_kwargs"]["imgsz"] == 320
    assert runs[1]["train_kwargs"]["batch"] == 4
    assert runs[0]["train_kwargs"]["name"].startswith("demo_")
    assert runs[1]["train_kwargs"]["name"].startswith("demo_")
