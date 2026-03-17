# Phase 2 - Per-level Query Quota

## Phase Goal
- 目标: 在保留 Phase 1 center rerank 的前提下，验证“按特征层配额选 query”是否进一步提升多尺度覆盖与最终精度。
- 当前状态: **Running**（2026-03-16）。
- 本阶段门槛（沿用快速门槛）:
  - `Δtest mAP50-95 >= +0.3` points（相对 baseline）。
  - latency 劣化不超过 `+10%`（或 FPS 下降不超过 `8%`）。

## Code Changes

### 1) Query quota 核心逻辑
- `/home/fyb/mydir/ultralytics/ultralytics/nn/modules/head.py:1795`
  - `_parse_level_ratios`：解析并归一化每层比例。
- `/home/fyb/mydir/ultralytics/ultralytics/nn/modules/head.py:1822`
  - `_allocate_level_quotas`：按比例分配整数配额并处理余数/容量。
- `/home/fyb/mydir/ultralytics/ultralytics/nn/modules/head.py:1876`
  - `_select_query_indices`：支持 `none`（全局 top-k）与 `fixed`（分层配额 + 全局补齐）。
- `/home/fyb/mydir/ultralytics/ultralytics/nn/modules/head.py:1741`
  - 解码前 query 选择切换到配额策略。

### 2) 配置与参数透传
- `/home/fyb/mydir/ultralytics/ultralytics/cfg/default.yaml:47`
  - 增加 `query_quota_mode/query_level_ratios/query_quota_min_per_level` 默认项。
- `/home/fyb/mydir/ultralytics/ultralytics/nn/tasks.py:778`
  - runtime args 同步配额参数到 head。
- `/home/fyb/mydir/ultralytics/examples/train_rtdetr_dior.py:72`
  - CLI 增加配额参数。
- `/home/fyb/mydir/ultralytics/examples/train_rtdetr_dior.py:133`
  - `model.train(...)` 透传配额参数。

## Experiment Matrix

### 固定配置
- model/data/seed/epochs/patience/imgsz/batch/workers/cache/device:
  - `rtdetr-l.pt`
  - `/home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml`
  - `0/40/8/512/16/1/disk/0`
- Phase 1 固定参数保持不变:
  - `query_rerank_mode=center`
  - `center_lambda_max=0.35`
  - `center_lambda_warmup_epochs=10`
  - `center_score_norm=zscore_image`
  - `center_score_clip=6.0`
  - `center_loss_weight=0.5`
  - `center_pos_alpha=4.0`
  - `center_empty_scale=0.25`

### 变量（quota）
| exp_id | run_name | query_quota_mode | query_level_ratios |
|---|---|---|---|
| P2-E0 | `phase2_quota_none_lam035_seed0_e40` | none | `-` |
| P2-E1 | `phase2_quota_50_30_20_lam035_seed0_e40` | fixed | `0.5,0.3,0.2` |
| P2-E2 | `phase2_quota_60_25_15_lam035_seed0_e40` | fixed | `0.6,0.25,0.15` |
| P2-E3 | `phase2_quota_40_35_25_lam035_seed0_e40` | fixed | `0.4,0.35,0.25` |

## Run Log

> Snapshot time: 2026-03-16（本地日志检查时）

| exp_id | run_name | run_dir | seed | key_hparams | best_epoch | val_map50_95 | test_map50_95 | latency_ms | status | notes |
|---|---|---|---:|---|---:|---:|---:|---:|---|---|
| P2-E0 | phase2_quota_none_lam035_seed0_e40 | `/home/fyb/mydir/ultralytics/runs/detect/runs/train/phase2_quota_none_lam035_seed0_e40` | 0 | `quota=none, lambda=0.35` | - | 0.56367 (epoch6) | - | - | running | 当前进行到 epoch 6，`results.csv` 持续写入 |
| P2-E1 | phase2_quota_50_30_20_lam035_seed0_e40 | `/home/fyb/mydir/ultralytics/runs/detect/runs/train/phase2_quota_50_30_20_lam035_seed0_e40` | 0 | `quota=fixed, ratios=0.5,0.3,0.2` | - | - | - | - | queued | 串行调度，等待 P2-E0 完成后启动 |
| P2-E2 | phase2_quota_60_25_15_lam035_seed0_e40 | `/home/fyb/mydir/ultralytics/runs/detect/runs/train/phase2_quota_60_25_15_lam035_seed0_e40` | 0 | `quota=fixed, ratios=0.6,0.25,0.15` | - | - | - | - | queued | 串行调度 |
| P2-E3 | phase2_quota_40_35_25_lam035_seed0_e40 | `/home/fyb/mydir/ultralytics/runs/detect/runs/train/phase2_quota_40_35_25_lam035_seed0_e40` | 0 | `quota=fixed, ratios=0.4,0.35,0.25` | - | - | - | - | queued | 串行调度 |

证据:
- PID: `runs/launch_logs/phase2_quota_sweep_seed0_20260316-110432.pid`
- Launch Log: `runs/launch_logs/phase2_quota_sweep_seed0_20260316-110432.log`
- Current CSV: `runs/detect/runs/train/phase2_quota_none_lam035_seed0_e40/results.csv`

## Results

### 当前可见中间指标（P2-E0）
| epoch | val mAP50-95 | val mAP50 | precision | recall |
|---:|---:|---:|---:|---:|
| 1 | 0.34721 | 0.45828 | 0.53099 | 0.46870 |
| 2 | 0.47516 | 0.64030 | 0.69708 | 0.62411 |
| 3 | 0.51821 | 0.68663 | 0.73845 | 0.64939 |
| 4 | 0.52600 | 0.70140 | 0.80104 | 0.65828 |
| 5 | 0.56252 | 0.74215 | 0.81670 | 0.68776 |
| 6 | 0.56367 | 0.74238 | 0.83281 | 0.70166 |

状态结论: 仍在训练早中期，尚不能做最终优选与 test 对比结论。

## Comparison
- 与 baseline 的正式对比：**Pending**（需 4 个 run 结束并统一 test 评估）。
- 与 Phase 1 best（`lambda=0.35`）对比：**Pending**（当前仅有 P2-E0 的中间 val 指标）。

## Gate Decision
- 当前判定: **Pending**。
- 原因:
  - 仅 P2-E0 在跑，P2-E1/E2/E3 尚未启动。
  - 尚无最佳 quota 的 test 指标与效率数据。

## Transition to Next Phase
- 当前不满足进入下一 Phase 的前置条件。
- 进入 Phase 3 的条件:
  - Phase 2 完成全部实验并输出统一对比。
  - 门槛判定为 Pass。

## Risks & Open Questions
- 串行 sweep 总耗时较长，可能受中途环境负载影响。
- 不同 quota 比例对小目标和大目标 AP 的影响方向可能不同，需要分指标分析（如 `AP_S/AP_M/AP_L`）。
- 仍为单 seed 快筛口径，后续若收益边界小，需要补多 seed 统计。
