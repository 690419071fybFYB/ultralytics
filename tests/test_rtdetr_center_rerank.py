# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn.modules import DetailInject, DetailInjectLite
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.utils.torch_utils import TORCH_1_11


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
def test_rtdetr_zscore_image():
    """Per-image z-score should be zero-mean, unit-variance and robust to constant inputs."""
    scores = torch.randn(3, 97)
    z = RTDETRDecoder._zscore_image(scores)
    assert torch.allclose(z.mean(dim=1), torch.zeros(3), atol=1e-5)
    assert torch.allclose(z.var(dim=1, unbiased=False), torch.ones(3), atol=1e-4)

    const = torch.full((2, 31), 5.0)
    z_const = RTDETRDecoder._zscore_image(const)
    assert torch.equal(z_const, torch.zeros_like(const))


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
def test_centerness_multi_gt_uses_max_rule():
    """Overlapping GTs should use max centerness target per token."""
    loss_fn = RTDETRDetectionLoss(nc=2, center_target="box_centerness", center_multi_gt_rule="max")
    points = torch.tensor(
        [
            [0.50, 0.50],  # inside both
            [0.80, 0.50],  # mostly outside the small box
            [0.10, 0.10],  # outside both
        ],
        dtype=torch.float32,
    )
    gt1 = torch.tensor([[0.50, 0.50, 0.80, 0.80]], dtype=torch.float32)
    gt2 = torch.tensor([[0.55, 0.50, 0.20, 0.20]], dtype=torch.float32)

    combined = loss_fn._build_box_centerness_targets(points, torch.cat([gt1, gt2], dim=0), [2])[0]
    target1 = loss_fn._build_box_centerness_targets(points, gt1, [1])[0]
    target2 = loss_fn._build_box_centerness_targets(points, gt2, [1])[0]
    expected = torch.maximum(target1, target2)
    assert torch.allclose(combined, expected, atol=1e-6)


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
def test_center_loss_empty_gt_is_finite():
    """Center loss for images without GT should remain finite and non-negative."""
    loss_fn = RTDETRDetectionLoss(
        nc=2,
        center_loss_weight=0.5,
        center_pos_alpha=4.0,
        center_empty_scale=0.25,
        center_target="box_centerness",
        center_multi_gt_rule="max",
    )

    center_logits = torch.randn(2, 64)
    center_points = torch.rand(64, 2)
    center_valid_mask = torch.ones(64, dtype=torch.bool)
    gt_bboxes = torch.zeros((0, 4), dtype=torch.float32)
    gt_groups = [0, 0]

    center_loss = loss_fn._get_center_loss(center_logits, center_points, center_valid_mask, gt_bboxes, gt_groups)
    assert torch.isfinite(center_loss)
    assert float(center_loss) >= 0.0


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
def test_rtdetr_decoder_center_rerank_forward_shapes():
    """Both rerank modes should run forward and return consistent output shapes."""
    x = [
        torch.randn(1, 32, 8, 8),
        torch.randn(1, 32, 4, 4),
        torch.randn(1, 32, 2, 2),
    ]

    decoder_none = RTDETRDecoder(
        nc=5, ch=(32, 32, 32), hd=32, nq=20, ndp=2, nh=4, ndl=1, d_ffn=64, nd=0, query_rerank_mode="none"
    ).eval()
    decoder_center = RTDETRDecoder(
        nc=5, ch=(32, 32, 32), hd=32, nq=20, ndp=2, nh=4, ndl=1, d_ffn=64, nd=0, query_rerank_mode="center"
    ).eval()
    decoder_center.load_state_dict(decoder_none.state_dict(), strict=False)

    with torch.no_grad():
        y_none, aux_none = decoder_none(x)
        y_center, aux_center = decoder_center(x)

    assert y_none.shape == y_center.shape == (1, 20, 9)
    assert len(aux_none) == 8 and len(aux_center) == 8


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
def test_rtdetr_decoder_none_matches_center_when_lambda_zero():
    """With lambda=0 and no score normalization, center rerank should match baseline none mode."""
    x = [
        torch.randn(1, 32, 8, 8),
        torch.randn(1, 32, 4, 4),
        torch.randn(1, 32, 2, 2),
    ]
    decoder_none = RTDETRDecoder(
        nc=5, ch=(32, 32, 32), hd=32, nq=20, ndp=2, nh=4, ndl=1, d_ffn=64, nd=0, query_rerank_mode="none"
    ).eval()
    decoder_center_zero = RTDETRDecoder(
        nc=5,
        ch=(32, 32, 32),
        hd=32,
        nq=20,
        ndp=2,
        nh=4,
        ndl=1,
        d_ffn=64,
        nd=0,
        query_rerank_mode="center",
        center_lambda_max=0.0,
        center_score_norm="none",
        center_score_clip=0.0,
    ).eval()
    decoder_center_zero.load_state_dict(decoder_none.state_dict(), strict=False)

    with torch.no_grad():
        y_none, aux_none = decoder_none(x)
        y_zero, aux_zero = decoder_center_zero(x)

    assert torch.allclose(y_none, y_zero, atol=1e-6, rtol=1e-6)
    for a, b in zip(aux_none[:4], aux_zero[:4]):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
def test_detail_inject_shape_and_gate_range():
    """DetailInject should align stage1 features, preserve semantic shape, and produce bounded gates."""
    module = DetailInject(c_sem=256, c_detail=128, c_out=256).eval()
    semantic = torch.randn(2, 256, 16, 16)
    detail = torch.randn(2, 128, 128, 128)

    with torch.no_grad():
        aligned_detail = module._align_detail(detail, semantic.shape[2:])
        gate = module.gate(torch.cat((semantic, aligned_detail), 1))
        fused = module((semantic, detail))

    assert aligned_detail.shape == semantic.shape == fused.shape == (2, 256, 16, 16)
    assert torch.isfinite(fused).all()
    assert float(gate.min()) >= 0.0
    assert float(gate.max()) <= 1.0


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
def test_rtdetr_l_with_detail_inject_builds_and_runs_forward():
    """The RT-DETR-L YAML with DetailInject should build and complete a forward pass."""
    model = RTDETRDetectionModel("ultralytics/cfg/models/rt-detr/rtdetr-l.yaml", ch=3, nc=80, verbose=False).eval()
    assert any(isinstance(m, DetailInject) for m in model.model)

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y, aux = model.predict(x)

    assert y.ndim == 3
    assert y.shape[0] == 1
    assert y.shape[2] == 84  # 4 box coords + 80 classes
    assert len(aux) == 8


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
def test_detail_inject_lite_shape_gate_and_alpha():
    """DetailInjectLite should preserve shape, produce bounded gates, and expose alpha."""
    module = DetailInjectLite(c_sem=256, c_detail=128, c_out=256).eval()
    semantic = torch.randn(2, 256, 16, 16)
    detail = torch.randn(2, 128, 128, 128)

    with torch.no_grad():
        aligned_detail = module._align_detail(detail, semantic.shape[2:])
        channel_gate = module._channel_gate(semantic, aligned_detail)
        detail_refined = aligned_detail * channel_gate
        spatial_gate = module._spatial_gate(semantic + detail_refined)
        fused = module((semantic, detail))

    assert aligned_detail.shape == semantic.shape == fused.shape == (2, 256, 16, 16)
    assert torch.isfinite(fused).all()
    assert float(channel_gate.min()) >= 0.0
    assert float(channel_gate.max()) <= 1.0
    assert float(spatial_gate.min()) >= 0.0
    assert float(spatial_gate.max()) <= 1.0
    assert isinstance(module.alpha, torch.nn.Parameter)
    assert torch.allclose(module.alpha.detach(), torch.tensor(0.1), atol=1e-6)


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
def test_rtdetr_l_with_detail_inject_lite_builds_and_runs_forward():
    """The RT-DETR-L YAML with DetailInjectLite should build and complete a forward pass."""
    model = RTDETRDetectionModel(
        "ultralytics/cfg/models/rt-detr/rtdetr-l-detailinject-lite.yaml", ch=3, nc=80, verbose=False
    ).eval()
    assert any(isinstance(m, DetailInjectLite) for m in model.model)

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y, aux = model.predict(x)

    assert y.ndim == 3
    assert y.shape[0] == 1
    assert y.shape[2] == 84
    assert len(aux) == 8
