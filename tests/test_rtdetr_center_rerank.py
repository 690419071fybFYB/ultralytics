# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.nn.modules import CenterRTDETRDecoder
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.utils.torch_utils import TORCH_1_11


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
def test_rtdetr_zscore_image():
    """Per-image z-score should be zero-mean, unit-variance and robust to constant inputs."""
    scores = torch.randn(3, 97)
    z = CenterRTDETRDecoder._zscore_image(scores)
    assert torch.allclose(z.mean(dim=1), torch.zeros(3), atol=1e-5)
    assert torch.allclose(z.var(dim=1, unbiased=False), torch.ones(3), atol=1e-4)

    const = torch.full((2, 31), 5.0)
    z_const = CenterRTDETRDecoder._zscore_image(const)
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
        nc=5, ch=(32, 32, 32), hd=32, nq=20, ndp=2, nh=4, ndl=1, d_ffn=64, nd=0
    ).eval()
    decoder_center = CenterRTDETRDecoder(
        nc=5, ch=(32, 32, 32), hd=32, nq=20, ndp=2, nh=4, ndl=1, d_ffn=64, nd=0, query_rerank_mode="center"
    ).eval()
    decoder_center.load_state_dict(decoder_none.state_dict(), strict=False)

    with torch.no_grad():
        y_none, aux_none = decoder_none(x)
        y_center, aux_center = decoder_center(x)

    assert y_none.shape == y_center.shape == (1, 20, 9)
    assert len(aux_none) == 5
    assert len(aux_center) == 8


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
def test_rtdetr_decoder_none_matches_center_when_lambda_zero():
    """With lambda=0 and no score normalization, center rerank should match baseline none mode."""
    x = [
        torch.randn(1, 32, 8, 8),
        torch.randn(1, 32, 4, 4),
        torch.randn(1, 32, 2, 2),
    ]
    decoder_none = RTDETRDecoder(
        nc=5, ch=(32, 32, 32), hd=32, nq=20, ndp=2, nh=4, ndl=1, d_ffn=64, nd=0
    ).eval()
    decoder_center_zero = CenterRTDETRDecoder(
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
def test_center_decoder_yaml_parse():
    """Center decoder model YAML should instantiate the configured subclass."""
    model = RTDETRDetectionModel(
        "ultralytics/cfg/models/rt-detr/rtdetr-l-center.yaml", nc=5, ch=3, verbose=False
    )
    head = model.model[-1]
    assert isinstance(head, CenterRTDETRDecoder)
