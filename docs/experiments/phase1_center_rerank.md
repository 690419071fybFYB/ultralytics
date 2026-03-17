# Phase 1 - Center-aware Query Reranking

## Phase Goal

- 目标: 在不引入 proposal-style 分支的前提下，仅通过中心先验重排（query reranking）提升 RT-DETR 的 query 初始化质量。
- 成功门槛（快速筛选轮）:
    - `Δtest mAP50-95 >= +0.3` points（相对 baseline）。
    - latency 劣化不超过 `+10%`（或 FPS 下降不超过 `8%`）。
- 失败判定: 任一硬门槛不满足即判 Fail，不进入 Phase 2。

## Code Changes

### 1) Center-aware reranking（核心逻辑）

- `/home/fyb/mydir/ultralytics/ultralytics/nn/modules/head.py:1716`
    - 增加 `query_rerank_mode=center` 分支。
- `/home/fyb/mydir/ultralytics/ultralytics/nn/modules/head.py:1723`
    - 增加 per-image z-score：`_zscore_image`。
- `/home/fyb/mydir/ultralytics/ultralytics/nn/modules/head.py:1729`
    - 融合分数 `fused_scores = cls_rank + lambda * center_rank`。
- `/home/fyb/mydir/ultralytics/ultralytics/nn/modules/head.py:1780`
    - 增加 `center_lambda_warmup_epochs` 调度。

### 2) Center supervision loss

- `/home/fyb/mydir/ultralytics/ultralytics/models/utils/loss.py:400`
    - 增加 `center_loss_weight/center_pos_alpha/center_empty_scale/center_target/center_multi_gt_rule`。
- `/home/fyb/mydir/ultralytics/ultralytics/models/utils/loss.py:418`
    - 实现 box-centerness target 构造（多 GT 使用 `max` 规则）。
- `/home/fyb/mydir/ultralytics/ultralytics/models/utils/loss.py:461`
    - 增加逐 token 的 BCEWithLogits center loss。
- `/home/fyb/mydir/ultralytics/ultralytics/models/utils/loss.py:555`
    - 将 `loss_center` 汇入总 loss 字典。

### 3) 训练/推理参数与运行时对齐

- `/home/fyb/mydir/ultralytics/ultralytics/nn/tasks.py:763`
    - 增加 RT-DETR head runtime args 同步（保证 train/infer 一致）。
- `/home/fyb/mydir/ultralytics/ultralytics/nn/tasks.py:786`
    - criterion 注入 center loss 相关参数。
- `/home/fyb/mydir/ultralytics/ultralytics/nn/tasks.py:835`
    - loss 前向中传递 `center_logits/center_points/center_valid_mask`。

### 4) 配置与训练入口

- `/home/fyb/mydir/ultralytics/ultralytics/cfg/default.yaml:41`
    - 新增 rerank、lambda、score norm/clip 配置。
- `/home/fyb/mydir/ultralytics/ultralytics/cfg/default.yaml:137`
    - 新增 center loss 配置项。
- `/home/fyb/mydir/ultralytics/examples/train_rtdetr_dior.py:49`
    - 增加 CLI 参数和参数透传。
- `/home/fyb/mydir/ultralytics/examples/phase1_lambda_sweep_seed0.py:57`
    - 增加 λ 扫描脚本（seed=0, 自动汇总报告）。

## Experiment Matrix

### 固定配置

- model: `rtdetr-l.pt`
- data: `/home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml`
- seed: `0`
- epochs: `40`
- patience: `8`
- imgsz/batch/workers/cache/device: `512/16/1/disk/0`
- rerank mode: `center`
- warmup: `center_lambda_warmup_epochs=10`
- loss: `center_loss_weight=0.5`, `center_pos_alpha=4.0`, `center_empty_scale=0.25`

### 变量（单变量扫描）

| exp_id | lambda_max | run_name                         |
| ------ | ---------: | -------------------------------- |
| P1-E1  |       0.15 | `phase1_center_lam015_seed0_e40` |
| P1-E2  |       0.25 | `phase1_center_lam025_seed0_e40` |
| P1-E3  |       0.35 | `phase1_center_lam035_seed0_e40` |

## Run Log

| exp_id | run_name                       | run_dir                                                                             | seed | key_hparams   | best_epoch | val_map50_95 | test_map50_95 | latency_ms | status  | notes                                   |
| ------ | ------------------------------ | ----------------------------------------------------------------------------------- | ---: | ------------- | ---------: | -----------: | ------------: | ---------: | ------- | --------------------------------------- |
| P1-E1  | phase1_center_lam015_seed0_e40 | `/home/fyb/mydir/ultralytics/runs/detect/runs/train/phase1_center_lam015_seed0_e40` |    0 | `lambda=0.15` |         39 |     0.650380 |             - |          - | success | 40 epochs 完成（8.261h）                |
| P1-E2  | phase1_center_lam025_seed0_e40 | `/home/fyb/mydir/ultralytics/runs/detect/runs/train/phase1_center_lam025_seed0_e40` |    0 | `lambda=0.25` |         40 |     0.653040 |             - |          - | success | 40 epochs 完成（8.329h）                |
| P1-E3  | phase1_center_lam035_seed0_e40 | `/home/fyb/mydir/ultralytics/runs/detect/runs/train/phase1_center_lam035_seed0_e40` |    0 | `lambda=0.35` |         40 |     0.656640 |      0.553459 |  16.185679 | success | 40 epochs 完成（8.763h），被选为 best λ |

数据来源:

- `runs/analysis/phase1_lambda_sweep_seed0/sweep_train_summary.csv`
- `runs/analysis/phase1_lambda_sweep_seed0/best_lambda_test_summary.csv`
- `runs/analysis/phase1_lambda_sweep_seed0/logs/*.log`

## Results

### Sweep Val 指标

| lambda | best_epoch | val mAP50-95 | val mAP50 | precision |   recall |
| -----: | ---------: | -----------: | --------: | --------: | -------: |
|   0.15 |         39 |     0.650380 |  0.832730 |  0.893180 | 0.790490 |
|   0.25 |         40 |     0.653040 |  0.837020 |  0.905210 | 0.789890 |
|   0.35 |         40 |     0.656640 |  0.840280 |  0.896550 | 0.794680 |

### Best λ=0.35 的 Test 指标

| metric            |     value |
| ----------------- | --------: |
| test mAP50-95     |  0.553459 |
| test mAP50        |  0.754819 |
| test mAP75        |  0.595093 |
| preprocess ms/im  |  0.451157 |
| infer ms/im       | 14.230122 |
| postprocess ms/im |  1.504399 |

## Comparison

### Baseline vs Phase1 Best (λ=0.35)

| Metric         |  Baseline | Phase1 Best |     Delta |
| -------------- | --------: | ----------: | --------: |
| test mAP50-95  |  0.529117 |    0.553459 | +0.024342 |
| test mAP50     |  0.729857 |    0.754819 | +0.024962 |
| test mAP75     |  0.567229 |    0.595093 | +0.027864 |
| latency(ms/im) | 16.241847 |   16.185679 |   -0.346% |
| FPS            |    61.569 |      61.783 |   +0.347% |

来源: `runs/analysis/phase1_lambda_sweep_seed0/comparison_vs_baseline.md`

## Gate Decision

- `ΔmAP50-95 >= +0.3 points`: **Pass**（+2.434 points）。
- `latency <= +10%` 或 `FPS drop <= 8%`: **Pass**（latency -0.346%，FPS +0.347%）。
- Overall: **Phase 1 Pass**。

证据文件:

- `runs/analysis/phase1_lambda_sweep_seed0/comparison_vs_baseline.md`
- `runs/analysis/phase1_lambda_sweep_seed0/best_lambda_test_summary.csv`

## Transition to Next Phase

- 进入依据: Phase 1 门槛通过，且收益明确来自 `center-aware rerank`（不依赖 proposal 分支）。
- 下一阶段唯一新增变量: `query quota policy`（`query_quota_mode` / `query_level_ratios`）。
- 保持不变: model/data/seed/epochs/imgsz/batch/workers/cache/lambda/loss 权重。

## Risks & Open Questions

- 当前 Phase 1 只做了 seed=0 快筛，统计置信度有限。
- AP 增益是否可跨 seed 稳定复现尚未验证。
- 下一阶段需重点观察小目标 AP 与效率变化是否出现 trade-off。
