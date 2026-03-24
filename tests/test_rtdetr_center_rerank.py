# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch
import torch.nn as nn

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.modules.transformer import DeformableTransformerDecoder, DeformableTransformerDecoderLayer, HistoryFusion
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
def test_history_fusion_softmax_and_window():
    """History fusion should produce normalized layer-wise weights and respect history window."""
    fusion = HistoryFusion(max_history=4)
    with torch.no_grad():
        fusion.logits.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))

    history = [torch.full((1, 2, 8), float(i)) for i in range(4)]
    fused_all, alpha_all = fusion(history, history_window=None)
    expected_alpha_all = torch.softmax(torch.tensor([0.0, 1.0, 2.0, 3.0]), dim=0)
    expected_all = sum(expected_alpha_all[i] * history[i] for i in range(4))
    assert torch.allclose(alpha_all, expected_alpha_all, atol=1e-6)
    assert torch.allclose(fused_all, expected_all, atol=1e-6)

    fused_win, alpha_win = fusion(history, history_window=2)
    expected_alpha_win = torch.softmax(torch.tensor([2.0, 3.0]), dim=0)
    expected_win = expected_alpha_win[0] * history[2] + expected_alpha_win[1] * history[3]
    assert alpha_win.shape == (2,)
    assert torch.allclose(alpha_win, expected_alpha_win, atol=1e-6)
    assert torch.allclose(fused_win, expected_win, atol=1e-6)


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
def test_deformable_decoder_history_fusion_forward_shapes():
    """History-aware decoder should keep bbox/score output shapes unchanged."""
    layer = DeformableTransformerDecoderLayer(d_model=16, n_heads=4, d_ffn=32, n_levels=2, n_points=2)
    decoder = DeformableTransformerDecoder(
        hidden_dim=16, decoder_layer=layer, num_layers=2, eval_idx=-1, use_history_fusion=True, history_window=2
    ).train()

    embed = torch.randn(1, 5, 16)
    refer_bbox = torch.randn(1, 5, 4)
    feats = torch.randn(1, 5, 16)
    shapes = [[2, 2], [1, 1]]
    bbox_head = nn.ModuleList([nn.Linear(16, 4), nn.Linear(16, 4)])
    score_head = nn.ModuleList([nn.Linear(16, 3), nn.Linear(16, 3)])
    pos_mlp = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 16))

    dec_bboxes, dec_scores = decoder(embed, refer_bbox, feats, shapes, bbox_head, score_head, pos_mlp)
    assert dec_bboxes.shape == (2, 1, 5, 4)
    assert dec_scores.shape == (2, 1, 5, 3)
