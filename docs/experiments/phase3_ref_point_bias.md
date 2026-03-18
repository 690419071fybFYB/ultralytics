# Phase 3 - Reference Point Bias (xy)

## Phase Goal
- 目标: 在 Phase 1 最优配置基础上，验证仅对 reference point 的 `xy` 做轻量偏置是否带来稳定收益。
- 当前状态: **Code Ready**（2026-03-17）。
- 基线固定:
  - `query_rerank_mode=center`
  - `center_lambda_max=0.35`
  - `query_quota_mode=none`

## Code Changes
- `/home/fyb/mydir/ultralytics/ultralytics/nn/modules/head.py`
  - 新增 `enc_ref_bias_head`，实现 `reference_point_bias_mode=xy` 的参考点中心偏置。
  - 新增 warmup 调度与中心门控开关。
  - 前向新增输出：`selected_query_points`、`biased_query_xy`、`selected_query_valid_mask`。
- `/home/fyb/mydir/ultralytics/ultralytics/nn/tasks.py`
  - runtime args 同步 Phase 3 参数。
  - loss 透传新增输出到 criterion。
  - 初始化损失时注入 `reference_point_bias_loss_weight/reference_point_bias_empty_scale`。
- `/home/fyb/mydir/ultralytics/ultralytics/models/utils/loss.py`
  - 新增 query->GT 中心分配与 centerness 加权 L1 损失 `loss_ref_bias`。
- `/home/fyb/mydir/ultralytics/ultralytics/cfg/default.yaml`
  - 新增 Phase 3 默认参数。
- `/home/fyb/mydir/ultralytics/examples/train_rtdetr_dior.py`
  - 新增 CLI 参数并透传到 `model.train(...)`。

## Experiment Matrix
- 固定项:
  - model/data/seed/epochs/patience/imgsz/batch/workers/cache/device:
    - `rtdetr-l.pt`
    - `/home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml`
    - `0/40/8/512/16/1/disk/0`
  - `query_quota_mode=none`
  - `query_rerank_mode=center`
  - `center_lambda_max=0.35`
- Stage A (`reference_point_bias_loss_weight=0.2`):
  - `phase3_refbias_s003_lw02_seed0_e40`
  - `phase3_refbias_s006_lw02_seed0_e40`
  - `phase3_refbias_s010_lw02_seed0_e40`
- Stage B (L1 消融，基于 Stage A 最优 shift):
  - `reference_point_bias_loss_weight=0.0`
  - `reference_point_bias_loss_weight=0.2`

## Run Log
- 待执行（代码已就绪，尚未启动 Phase 3 训练）。

## Results
- 待填充。

## Comparison
- 待填充（与 Phase 1 best 同口径 test 对比）。

## Gate Decision
- 当前判定: **Pending**（未进入训练评估阶段）。

## Transition to Next Phase
- 仅当满足以下条件才推进下一阶段:
  - `Δtest mAP50-95 >= +0.3`
  - latency 增幅 `<= 10%` 或 FPS 下降 `<= 8%`
  - 同一 shift 下 `loss_weight=0.2` 不劣于 `loss_weight=0.0`

## Risks & Open Questions
- 若增益较小，需要补 3-seed 配对统计确认稳定性。
- 需关注 `AP_S` 与延迟之间的 trade-off。
