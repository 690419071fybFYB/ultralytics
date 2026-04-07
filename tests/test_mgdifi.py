# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import types
from pathlib import Path

import pytest
import torch

from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import TORCH_1_11


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
@pytest.mark.parametrize("normalize_before", [False, True])
def test_mgdifi_forward_shape_with_external_prior(normalize_before):
    """MGDIFI should accept an external mab prior and preserve BCHW shape in both norm modes."""
    from ultralytics.nn.modules import MGDIFI

    x = torch.randn(2, 256, 20, 20)
    mab_prior = torch.rand(2, 1, 80, 80)
    m = MGDIFI(256, 1024, 8, normalize_before=normalize_before).eval()
    with torch.no_grad():
        y = m(x, mab_prior)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
@pytest.mark.parametrize("normalize_before", [False, True])
def test_mgdifi_forward_shape_without_external_prior(normalize_before):
    """MGDIFI should also work without an explicit mab prior for drop-in AIFI replacement."""
    from ultralytics.nn.modules import MGDIFI

    x = torch.randn(2, 256, 20, 20)
    m = MGDIFI(256, 1024, 8, normalize_before=normalize_before).eval()
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_mgdifi_prior_clamp_and_bounded_scalars():
    """MGDIFI should clamp priors to [0, 1] and bound alpha/beta through sigmoid."""
    from ultralytics.nn.modules import MGDIFI

    x = torch.randn(2, 256, 20, 20)
    mab_prior = torch.randn(2, 1, 80, 80) * 10.0
    m = MGDIFI(256, 1024, 8).eval()

    aligned = m._align_mab_prior(mab_prior, x)
    assert aligned.min() >= 0.0
    assert aligned.max() <= 1.0
    assert torch.allclose(m.alpha.detach(), torch.tensor(0.0, device=m.alpha.device))
    assert torch.allclose(m.beta.detach(), torch.tensor(0.0, device=m.beta.device))

    with torch.no_grad():
        alpha = torch.sigmoid(m.alpha)
        beta = torch.sigmoid(m.beta)
    assert 0.0 < alpha.item() < 1.0
    assert 0.0 < beta.item() < 1.0


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_mgdifi_internal_mab_prior_shape_and_range():
    """MGDIFI should build a soft internal morphology prior when no external prior is provided."""
    from ultralytics.nn.modules import MGDIFI

    x = torch.randn(2, 256, 20, 20)
    m = MGDIFI(256, 1024, 8, mab_cfg={"use_internal_mab": True, "kernel_size": 5, "hidden_ratio": 0.125}).eval()

    with torch.no_grad():
        mab = m._build_internal_mab_prior(x)

    assert mab.shape == (2, 1, 20, 20)
    assert mab.min() >= 0.0
    assert mab.max() <= 1.0
    assert torch.isfinite(mab).all()


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_mgdifi_uses_internal_mab_when_prior_missing():
    """MGDIFI should synthesize an internal prior when forward() is called without mab_prior."""
    from ultralytics.nn.modules import MGDIFI

    x = torch.randn(2, 256, 20, 20)
    m = MGDIFI(256, 1024, 8, mab_cfg={"use_internal_mab": True}).eval()

    seen = {"called": False}
    original_build = m._build_internal_mab_prior

    def wrapped_build(self, x_in):
        seen["called"] = True
        return original_build(x_in)

    m._build_internal_mab_prior = types.MethodType(wrapped_build, m)

    with torch.no_grad():
        y = m(x)

    assert seen["called"] is True
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_mgdifi_yaml_parse():
    """RT-DETR YAML with MGDIFI should build successfully."""
    from ultralytics.nn.modules import MGDIFI

    model = RTDETRDetectionModel("ultralytics/cfg/models/rt-detr/rtdetr-l-mgdifi.yaml", nc=5, ch=3, verbose=False)
    aifi = model.model[11]
    assert isinstance(aifi, MGDIFI)


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_mgdifi_center_yaml_parse():
    """Center RT-DETR YAML with MGDIFI should keep the center decoder while replacing AIFI."""
    from ultralytics.nn.modules import CenterRTDETRDecoder, MGDIFI

    model = RTDETRDetectionModel("ultralytics/cfg/models/rt-detr/rtdetr-l-center-mgdifi.yaml", nc=5, ch=3, verbose=False)
    assert isinstance(model.model[11], MGDIFI)
    assert isinstance(model.model[-1], CenterRTDETRDecoder)


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_mgdifi_yaml_parse_with_explicit_mab_cfg(tmp_path):
    """RT-DETR YAML should be able to configure internal MAB behavior for MGDIFI."""
    from ultralytics.nn.modules import MGDIFI

    base_yaml = Path("ultralytics/cfg/models/rt-detr/rtdetr-l-mgdifi.yaml").read_text()
    tuned_yaml = base_yaml.replace(
        "  - [-1, 1, MGDIFI, [1024, 8]]",
        (
            "  - [-1, 1, MGDIFI, [1024, 8, 0.0, null, false, "
            "{use_internal_mab: true, kernel_size: 7, hidden_ratio: 0.125}]]"
        ),
    )
    yaml_path = tmp_path / "rtdetr-l-mgdifi-tuned.yaml"
    yaml_path.write_text(tuned_yaml)

    model = RTDETRDetectionModel(str(yaml_path), nc=5, ch=3, verbose=False)
    mgdifi = model.model[11]
    assert isinstance(mgdifi, MGDIFI)
    assert mgdifi.use_internal_mab is True
    assert mgdifi.mab_kernel_size == 7


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_rtdetr_model_forwards_mab_prior_to_mgdifi():
    """RT-DETR should accept model(x, mab_prior=prior) and forward the prior into MGDIFI."""
    from ultralytics.nn.modules import MGDIFI

    model = RTDETRDetectionModel("ultralytics/cfg/models/rt-detr/rtdetr-l-mgdifi.yaml", nc=5, ch=3, verbose=False).eval()
    mgdifi = model.model[11]
    assert isinstance(mgdifi, MGDIFI)

    seen = {"prior": "unset"}
    original_forward = mgdifi.forward

    def wrapped_forward(self, x, mab_prior=None):
        seen["prior"] = mab_prior
        return original_forward(x, mab_prior=mab_prior)

    mgdifi.forward = types.MethodType(wrapped_forward, mgdifi)

    x = torch.randn(1, 3, 160, 160)
    prior = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        y = model(x, mab_prior=prior)

    assert seen["prior"] is not None
    assert torch.allclose(seen["prior"], prior)
    assert isinstance(y, tuple)


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_rtdetr_model_forward_without_mab_prior_stays_compatible():
    """RT-DETR should still allow model(x) and pass None into MGDIFI by default."""
    from ultralytics.nn.modules import MGDIFI

    model = RTDETRDetectionModel("ultralytics/cfg/models/rt-detr/rtdetr-l-mgdifi.yaml", nc=5, ch=3, verbose=False).eval()
    mgdifi = model.model[11]
    assert isinstance(mgdifi, MGDIFI)

    seen = {"prior": "unset"}
    original_forward = mgdifi.forward

    def wrapped_forward(self, x, mab_prior=None):
        seen["prior"] = mab_prior
        return original_forward(x, mab_prior=mab_prior)

    mgdifi.forward = types.MethodType(wrapped_forward, mgdifi)

    x = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        y = model(x)

    assert seen["prior"] is None
    assert isinstance(y, tuple)


@pytest.mark.skipif(not TORCH_1_11, reason="RT-DETR requires torch>=1.11")
def test_rtdetr_model_uses_internal_mab_without_external_prior():
    """RT-DETR should let YAML-configured MGDIFI synthesize its own prior when model(x) is used."""
    from ultralytics.nn.modules import MGDIFI

    model = RTDETRDetectionModel("ultralytics/cfg/models/rt-detr/rtdetr-l-mgdifi.yaml", nc=5, ch=3, verbose=False).eval()
    mgdifi = model.model[11]
    assert isinstance(mgdifi, MGDIFI)

    seen = {"called": False}
    original_build = mgdifi._build_internal_mab_prior

    def wrapped_build(self, x_in):
        seen["called"] = True
        return original_build(x_in)

    mgdifi._build_internal_mab_prior = types.MethodType(wrapped_build, mgdifi)

    x = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        y = model(x)

    assert seen["called"] is True
    assert isinstance(y, tuple)
