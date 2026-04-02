# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.init import constant_

from .head import RTDETRDecoder


class CenterRTDETRDecoder(RTDETRDecoder):
    """RT-DETR decoder variant with center-aware query reranking."""

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
        query_rerank_mode: str = "center",
        center_fusion_strategy: str = "add",
        center_lambda_max: float = 0.25,
        center_lambda_warmup_epochs: int = 10,
        center_score_norm: str = "zscore_image",
        center_score_clip: float = 6.0,
    ):
        """Initialize the center-aware RT-DETR decoder."""
        super().__init__(
            nc=nc,
            ch=ch,
            hd=hd,
            nq=nq,
            ndp=ndp,
            nh=nh,
            ndl=ndl,
            d_ffn=d_ffn,
            dropout=dropout,
            act=act,
            eval_idx=eval_idx,
            nd=nd,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learnt_init_query=learnt_init_query,
        )
        # 这几个属性控制“分类分数 + 中心性分数”的联合排序行为。
        self.query_rerank_mode = query_rerank_mode
        self.center_fusion_strategy = center_fusion_strategy
        self.center_lambda_max = center_lambda_max
        self.center_lambda_warmup_epochs = center_lambda_warmup_epochs
        self.center_score_norm = center_score_norm
        self.center_score_clip = center_score_clip
        # baseline decoder 没有这个分支；center 版本额外预测每个 encoder token 的中心性。
        self.enc_center_head = nn.Linear(self.hidden_dim, 1).to(self.enc_score_head.weight.device)
        constant_(self.enc_center_head.bias, 0.0)

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """Run RT-DETR forward and additionally return center-aware encoder outputs."""
        from ultralytics.models.utils.ops import get_cdn_group

        feats, shapes = self._get_encoder_input(x)
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        # 和 baseline 的主要区别在这里：除了原始 encoder top-k 信息，还返回 center 监督所需的附加量。
        embed, refer_bbox, enc_bboxes, enc_scores, enc_center_logits, center_points, center_valid_mask = (
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox, batch=batch)
        )

        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta, enc_center_logits, center_points, center_valid_mask
        if self.training:
            return x
        # 推理主输出仍保持和 baseline 一致，便于复用现有后处理；center 相关信息只放在 aux 里。
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Generate decoder inputs and center-aware query selection metadata."""
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        if self.training:
            if hasattr(self.valid_mask, "is_inference") and self.valid_mask.is_inference():
                self.valid_mask = self.valid_mask.clone()
            if hasattr(self.anchors, "is_inference") and self.anchors.is_inference():
                self.anchors = self.anchors.clone()

        features = self.enc_output(self.valid_mask * feats)
        enc_outputs_scores = self.enc_score_head(features)

        cls_scores = enc_outputs_scores.max(-1).values
        query_rerank_mode = str(getattr(self, "query_rerank_mode", "none")).lower()
        if query_rerank_mode == "center":
            # center 分支对所有 encoder token 打分，再和分类分数融合后做 top-k。
            enc_outputs_center = self.enc_center_head(features).squeeze(-1)
            center_score_norm = str(getattr(self, "center_score_norm", "zscore_image")).lower()
            if center_score_norm == "zscore_image":
                # 两类分数量纲不同，先做逐图归一化再融合，避免某一分支数值主导排序。
                cls_rank = self._zscore_image(cls_scores)
                center_rank = self._zscore_image(enc_outputs_center)
            else:
                cls_rank = cls_scores
                center_rank = enc_outputs_center
            fused_scores = self._fuse_query_scores(cls_rank, center_rank, batch=batch)
            query_scores = fused_scores
            center_logits = enc_outputs_center
        else:
            query_scores = cls_scores
            center_logits = None

        # 这里才是真正影响 decoder 初始化的地方：按融合后的 query_scores 重新选 top-k。
        topk_ind = torch.topk(query_scores, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        # center loss 用的是所有 encoder token 的几何中心点，而不是 top-k query 的中心点。
        center_points = self.anchors[..., :2].sigmoid().squeeze(0)
        center_valid_mask = self.valid_mask.squeeze(0).squeeze(-1)
        return embeddings, refer_bbox, enc_bboxes, enc_scores, center_logits, center_points, center_valid_mask

    @staticmethod
    def _zscore_image(scores: torch.Tensor, eps: float = 1e-6, var_eps: float = 1e-8) -> torch.Tensor:
        """Apply per-image z-score normalization to scores with near-zero variance protection."""
        mean = scores.mean(dim=1, keepdim=True)
        var = scores.var(dim=1, unbiased=False, keepdim=True)
        z = (scores - mean) / torch.sqrt(var + eps)
        return torch.where(var > var_eps, z, torch.zeros_like(z))

    def _center_lambda(self, batch: dict | None) -> float:
        """Return the center score fusion weight, with warmup during training."""
        lam = max(float(getattr(self, "center_lambda_max", 0.25)), 0.0)
        if not self.training:
            return lam

        warmup_epochs = max(float(getattr(self, "center_lambda_warmup_epochs", 10)), 0.0)
        if warmup_epochs == 0.0:
            return lam
        if batch is None or "epoch" not in batch:
            return lam
        # 训练前期先弱化 center 分支，避免它在分类分支尚未稳定时过早扰动 query 排序。
        epoch = float(batch["epoch"])
        ratio = min(max(epoch / warmup_epochs, 0.0), 1.0)
        return lam * ratio

    def _fuse_query_scores(
        self,
        cls_rank: torch.Tensor,
        center_rank: torch.Tensor,
        batch: dict | None = None,
    ) -> torch.Tensor:
        """Fuse class and center ranks for query selection.

        `add` keeps the current additive baseline.
        `geom` maps both ranks into (0, 1) and applies geometric-mean fusion, so tokens score highest only when
        class and center responses are both high.
        """
        lam = self._center_lambda(batch)
        strategy = str(getattr(self, "center_fusion_strategy", "add")).lower()
        if strategy == "geom":
            cls_prob = torch.sigmoid(cls_rank)
            center_prob = torch.sigmoid(lam * center_rank)
            return torch.sqrt(torch.clamp(cls_prob * center_prob, min=0.0))
        if strategy != "add":
            raise ValueError(f"Unsupported center_fusion_strategy: {strategy}")

        fused_scores = cls_rank + lam * center_rank
        score_clip = float(getattr(self, "center_score_clip", 6.0))
        if score_clip > 0:
            fused_scores = fused_scores.clamp(-score_clip, score_clip)
        return fused_scores
