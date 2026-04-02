# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import TORCH_1_11


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_sparse_gated_aifi_forward_shape_and_finite():
    """SparseGatedAIFI should preserve BCHW shape and produce finite outputs."""
    from ultralytics.nn.modules import SparseGatedAIFI

    x = torch.randn(2, 256, 8, 8)
    m = SparseGatedAIFI(256, 1024, 8).eval()
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
@pytest.mark.parametrize(
    ("enable_local", "enable_global", "enable_gate"),
    [
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (True, True, True),
    ],
)
def test_sparse_gated_aifi_branch_toggles(enable_local, enable_global, enable_gate):
    """Local/global/gate branch combinations should all run and keep gate in [0, 1]."""
    from ultralytics.nn.modules import SparseGatedAIFI

    x = torch.randn(1, 256, 6, 6)
    m = SparseGatedAIFI(
        256,
        1024,
        8,
        local_window=3,
        global_pool=2,
        gate_hidden_ratio=0.25,
        enable_local=enable_local,
        enable_global=enable_global,
        enable_gate=enable_gate,
    ).eval()
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    if enable_local and enable_global and enable_gate:
        gate = m._compute_gate(x.flatten(2).permute(0, 2, 1))
        assert gate.min() >= 0.0
        assert gate.max() <= 1.0


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_sparse_gated_aifi_invalid_without_any_branch():
    """Module should reject configurations without local or global branch."""
    from ultralytics.nn.modules import SparseGatedAIFI

    with pytest.raises(ValueError, match="at least one attention branch"):
        SparseGatedAIFI(256, 1024, 8, enable_local=False, enable_global=False)


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_sparse_gated_aifi_yaml_parse():
    """RT-DETR YAML with SparseGatedAIFI should build successfully."""
    from ultralytics.nn.modules import SparseGatedAIFI

    model = RTDETRDetectionModel(
        "ultralytics/cfg/models/rt-detr/rtdetr-l-sgaifi.yaml", nc=5, ch=3, verbose=False
    )
    aifi = model.model[11]
    assert isinstance(aifi, SparseGatedAIFI)
