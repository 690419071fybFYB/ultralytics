# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics import YOLO
from ultralytics.utils import ROOT, YAML


def test_ag_modules_forward_shapes():
    """AGDown should halve spatial size and AGUp should restore it."""
    from ultralytics.nn.modules import AGDown, AGUp

    x = torch.randn(1, 16, 32, 32)
    down = AGDown(16, 32)
    up = AGUp(32, 16)

    y = down(x)
    z = up(y)

    assert y.shape == (1, 32, 16, 16)
    assert z.shape == x.shape


def test_ag_modules_yaml_build():
    """The repository YAML using AGDown and AGUp should build and run a forward pass."""
    cfg = ROOT / "cfg" / "models" / "v8" / "yolo-ag.yaml"
    model = YOLO(cfg)
    model.info(verbose=False)

    model.model.eval()
    with torch.no_grad():
        outputs = model.model(torch.randn(1, 3, 64, 64))

    assert outputs is not None


def test_ag_v2_modules_forward_shapes_and_odd_channels():
    """AGDownV2 and AGUpV2 should preserve the planned spatial and channel dimensions."""
    from ultralytics.nn.modules import AGDownV2, AGUpV2

    x = torch.randn(1, 16, 32, 32)
    down = AGDownV2(16, 33)
    up = AGUpV2(33, 17)

    y = down(x)
    z = up(y)

    assert y.shape == (1, 33, 16, 16)
    assert z.shape == (1, 17, 32, 32)


def test_ag_v2_modules_yaml_build(tmp_path):
    """A temporary YAML using AGDownV2 and AGUpV2 should build and run a forward pass."""
    cfg = tmp_path / "yolo-ag-v2-test.yaml"
    YAML.save(
        cfg,
        {
            "nc": 1,
            "depth_multiple": 1.0,
            "width_multiple": 1.0,
            "backbone": [
                [-1, 1, "Conv", [16, 3, 2]],
                [-1, 1, "AGDownV2", [32]],
                [-1, 1, "AGDownV2", [64]],
                [-1, 1, "AGDownV2", [128]],
                [-1, 1, "AGDownV2", [256]],
            ],
            "head": [
                [-1, 1, "AGUpV2", [128]],
                [[-1, 3], 1, "Concat", [1]],
                [-1, 1, "Conv", [128, 3, 1]],
                [-1, 1, "AGUpV2", [64]],
                [[-1, 2], 1, "Concat", [1]],
                [-1, 1, "Conv", [64, 3, 1]],
                [[10, 7, 4], 1, "Detect", ["nc"]],
            ],
        },
    )

    model = YOLO(cfg)
    model.info(verbose=False)

    model.model.eval()
    with torch.no_grad():
        outputs = model.model(torch.randn(1, 3, 64, 64))

    assert outputs is not None


def test_ag_v2_repository_yaml_build():
    """The repository YAML using AGDownV2 and AGUpV2 should build and run a forward pass."""
    cfg = ROOT / "cfg" / "models" / "v8" / "yolo-ag-v2.yaml"
    model = YOLO(cfg)
    model.info(verbose=False)

    model.model.eval()
    with torch.no_grad():
        outputs = model.model(torch.randn(1, 3, 64, 64))

    assert outputs is not None
