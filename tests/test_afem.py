# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import ROOT
from ultralytics.utils.torch_utils import TORCH_1_11


def test_afem_forward_shape_and_finite():
    """AFEM should preserve BCHW shape when input and output channels match."""
    from ultralytics.nn.modules import AFEM

    x = torch.randn(2, 128, 16, 16)
    m = AFEM(128, 128).eval()
    with torch.no_grad():
        y = m(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_afem_forward_changes_channels():
    """AFEM should project to the requested output channels."""
    from ultralytics.nn.modules import AFEM

    x = torch.randn(1, 128, 16, 16)
    m = AFEM(128, 160).eval()
    with torch.no_grad():
        y = m(x)

    assert y.shape == (1, 160, 16, 16)
    assert torch.isfinite(y).all()


def test_afem_optional_args_are_exposed():
    """AFEM should accept explicit dilation and branch width overrides."""
    from ultralytics.nn.modules import AFEM

    x = torch.randn(1, 96, 12, 12)
    m = AFEM(96, 128, dilation=3, branch_channels=64).eval()
    with torch.no_grad():
        y = m(x)

    assert y.shape == (1, 128, 12, 12)
    assert m.dilation == 3
    assert m.branch_channels == 64


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_rtdetr_afem_yaml_parse():
    """RT-DETR YAML with AFEM should build and place AFEM after stage 1."""
    from ultralytics.nn.modules import AFEM

    model = RTDETRDetectionModel("ultralytics/cfg/models/rt-detr/rtdetr-l-afem.yaml", nc=5, ch=3, verbose=False)
    assert isinstance(model.model[2], AFEM)


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_rtdetr_center_afem_yaml_parse():
    """Center RT-DETR YAML with AFEM should preserve the center decoder."""
    from ultralytics.nn.modules import AFEM, CenterRTDETRDecoder

    model = RTDETRDetectionModel("ultralytics/cfg/models/rt-detr/rtdetr-l-center-afem.yaml", nc=5, ch=3, verbose=False)
    assert isinstance(model.model[2], AFEM)
    assert isinstance(model.model[-1], CenterRTDETRDecoder)


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
@pytest.mark.parametrize(
    "cfg_name",
    ("rtdetr-l-afem.yaml", "rtdetr-l-center-afem.yaml"),
)
def test_rtdetr_afem_repository_yaml_forward(cfg_name):
    """The repository RT-DETR AFEM YAMLs should run a dummy forward pass."""
    cfg = ROOT / "cfg" / "models" / "rt-detr" / cfg_name
    model = RTDETRDetectionModel(str(cfg), nc=5, ch=3, verbose=False).eval()

    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 160, 160))

    assert isinstance(outputs, tuple)


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_rtdetr_afem_yaml_parse_with_explicit_optional_args(tmp_path):
    """RT-DETR YAML should allow AFEM optional args to be overridden from YAML."""
    from ultralytics.nn.modules import AFEM

    base_yaml = (ROOT / "cfg" / "models" / "rt-detr" / "rtdetr-l-afem.yaml").read_text()
    tuned_yaml = base_yaml.replace(
        "  - [-1, 1, AFEM, [128]]",
        "  - [-1, 1, AFEM, [128, 3, 64]]",
    )
    yaml_path = tmp_path / "rtdetr-l-afem-tuned.yaml"
    yaml_path.write_text(tuned_yaml)

    model = RTDETRDetectionModel(str(yaml_path), nc=5, ch=3, verbose=False)
    afem = model.model[2]
    assert isinstance(afem, AFEM)
    assert afem.dilation == 3
    assert afem.branch_channels == 64
