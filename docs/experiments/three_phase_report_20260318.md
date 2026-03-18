# RT-DETR 三阶段实验汇报（截至 2026-03-18）

## 0. 任务背景
- 任务: DIOR 遥感目标检测（Ultralytics RT-DETR）。
- 主指标: `test mAP50-95(B)`。
- 基线主线: 在 RT-DETR 里优化 `query` 的初始化质量，优先走轻改动、可归因路线。

---

## 1. Phase 1: Center-aware Query Reranking

## 1.1 初衷
RT-DETR 在 decoder 前要从 encoder token 中选 top-k query。  
原始选择主要看分类得分，可能会漏掉“位置上更像目标中心”的 token。  
所以在排序阶段加入中心先验，目标是让初始 query 质量更高、收敛更快。

## 1.2 方法（center 机制 + lambda 的作用）
- 对每个 encoder token，取:
  - `S_cls`: 分类分数（类别 logits 最大值）。
  - `S_ctr`: 中心分数（额外 center head 输出）。
- 做每图标准化（z-score）后融合:
  - `S_fuse = Z_cls + lambda * Z_ctr`
- 用 `S_fuse` 做 top-k query 选择。

`lambda` 的准确含义:
- 它是“中心先验在排序里的权重系数”。
- `lambda` 越大: 更偏向中心先验。
- `lambda` 越小: 更接近原始分类排序。
- 训练前期用 warmup 从 0 线性升到 `lambda_max`，避免早期 center head 不稳定时干扰过大。

一句话通俗解释:
- `center` 不是直接出框，而是给每个 token 一个“像不像目标中心”的分。
- `lambda` 决定这个“像中心”在 query 排名里有多大话语权。

## 1.3 代码落点
- `ultralytics/nn/modules/head.py`
  - 计算 `center logits`，做 z-score 融合，得到 `fused_scores` 并 top-k。
- `ultralytics/models/utils/loss.py`
  - `box-centerness` 监督 center 分支。
- `ultralytics/nn/tasks.py`
  - 训练参数同步与 loss 透传。

## 1.4 实验设置
- 扫描 `lambda_max`: `0.15 / 0.25 / 0.35`，`seed=0`，`epochs=40`。
- 其余配置固定。

## 1.5 结果

### val（lambda 扫描）
| lambda_max | best val mAP50-95 | best val mAP50 |
|---:|---:|---:|
| 0.15 | 0.65038 | 0.83273 |
| 0.25 | 0.65304 | 0.83702 |
| 0.35 | 0.65664 | 0.84028 |

### test（最佳 lambda=0.35）
- `test mAP50-95 = 0.553459`
- 相对 baseline `0.529117`，提升 `+2.434` points。
- 延迟无劣化（历史记录显示约持平，略优）。

## 1.6 分析
- Phase 1 成功验证: “中心先验参与 query rerank”是有效方向。
- 最优点在 `lambda=0.35`，说明 DIOR 场景里中心先验信息是有价值的。

---

## 2. Phase 2: Per-level Query Quota

## 2.1 初衷
多尺度特征层 token 数量不均，担心全局 top-k 会偏向某一层。  
因此尝试固定每层配额（quota），强制多尺度 query 覆盖。

## 2.2 方法
- 在 query 选择阶段加入分层配额:
  - `none`: 原始全局 top-k。
  - `fixed`: 按比例为各层分配 query 数，再补齐。
- 比较比例:
  - `0.5,0.3,0.2`
  - `0.6,0.25,0.15`
  - `0.4,0.35,0.25`

## 2.3 结果（test，已完成）
| model_tag | test mAP50-95 | latency(ms/im) |
|---|---:|---:|
| phase1_center_lam035_seed0_e40 (baseline) | 0.553459 | 16.429015 |
| phase2_quota_none_lam035_seed0_e40 | 0.550640 | 16.479024 |
| phase2_quota_50_30_20_lam035_seed0_e40 | 0.548423 | 19.652493 |
| phase2_quota_60_25_15_lam035_seed0_e40 | 0.550756 | 19.404202 |
| phase2_quota_40_35_25_lam035_seed0_e40 (best of phase2) | 0.551984 | 19.399900 |

与 baseline 对比（phase2 最优 vs phase1 baseline）:
- `ΔmAP50-95 = -0.148` points
- latency `+18.083%`
- FPS drop `+15.314%`

## 2.4 为什么失败
- 想法本身合理，但在本任务中“强制每层拿固定名额”限制了排序自由度。
- 原本高质量 token 被配额约束稀释，精度没有提升。
- 同时引入了额外选择逻辑，推理开销增加，效率门槛也没过。

一句话通俗解释:
- quota 像“每层都要分座位”，公平了，但不一定让最强选手坐进去更多，最终成绩反而没涨。

---

## 3. Phase 3: Reference Point Bias（进行中）

## 3.1 初衷
Phase 1 证明“排序更好”有效。  
Phase 3 进一步做“位置更准”: 在 decoder 前，轻量修正 reference point 的 `xy`，让初始参考点更靠近真实目标中心。

## 3.2 方法（refbias 是什么变化）
- 原始 reference point: `r0 = bbox_head(topk_features) + anchors`
- 只改 `xy`（不改 `wh`）:
  - `xy0 = sigmoid(r0_xy)`
  - `delta = tanh(ref_bias_head(topk_features))`
  - `gate = sigmoid(selected_center_logits)`（可 detach）
  - `xy1 = clip(xy0 + scale(t) * gate * delta)`
  - `r_xy <- logit(xy1)`
- 增加辅助监督 `loss_ref_bias`:
  - 用 selected query points 与 GT 做 max-centerness 分配；
  - 对 `xy1` 和分配到的 `gt center` 做加权 L1。

一句话通俗解释:
- Phase 1 是“挑人更准”；
- Phase 3 是“人选好了，再把起跑点往目标中心挪一小步”。

## 3.3 Phase3 分阶段计划与进度（seed=0 快筛）
- Stage A（偏置幅度筛选）:
  - 目的: 找到合适的 `reference_point_bias_max_shift`。
  - 计划: 固定其余参数，仅比较 `0.03/0.06/0.10` 三组。
  - 当前: `s003` 已完成，`s006` 进行中，`s010` 未启动。
- Stage B（机制归因消融）:
  - 目的: 验证收益是否来自 `loss_ref_bias`。
  - 计划: 在 Stage A 最优 shift 上做两组对比:
    - `reference_point_bias_loss_weight=0.0`
    - `reference_point_bias_loss_weight=0.2`
  - 当前: 未启动，等待 Stage A 完成后执行。
- Stage C（最终 test 门槛判定）:
  - 目的: 决定 Phase3 是否通过并进入下一阶段。
  - 计划: 用 Stage B 最优模型跑 test，与 Phase1 baseline 同口径对比。
  - 门槛:
    - `Δtest mAP50-95 >= +0.3 points`
    - latency `<= +10%` 或 FPS 下降 `<= 8%`
  - 当前: 未启动。

`s003/s006/s010` 的含义:
- 它们对应 `reference_point_bias_max_shift` 的取值，分别是 `0.03/0.06/0.10`。
- 该值是相对整张图的归一化比例（`xy` 坐标范围 `[0,1]`），不是像素绝对值。
- 以 `imgsz=512` 粗略换算: `0.03≈15px`，`0.06≈31px`，`0.10≈51px`。

### 当前可见 val 指标
| run_name | status | best_epoch | best val mAP50-95 | best val mAP50 |
|---|---|---:|---:|---:|
| phase3_refbias_s003_lw02_seed0_e40 | completed | 39 | 0.65225 | 0.83837 |
| phase3_refbias_s006_lw02_seed0_e40 | running | 20 (current best) | 0.62796 | 0.81222 |

对照（Phase 1 best val）:
- `phase1_center_lam035_seed0_e40`: `0.65664`

## 3.4 初步分析
- 目前已完成的 `s003` 在 val 上尚未超过 Phase 1 best（差约 `-0.439` points）。
- 但 Phase 3 还未跑完整网格与 test，对结论仍需保守。
- 下一步必须按计划完成 Stage A + Stage B（L1消融）后再判定。

---

## 4. 三阶段总览

| Phase | 初衷 | 关键改动 | 当前结论 |
|---|---|---|---|
| Phase 1 | 用中心先验改进 query 排序 | `S_fuse = Z_cls + lambda * Z_ctr` | 成功，test 显著提升 |
| Phase 2 | 保证多尺度 query 覆盖 | 固定每层 quota 分配 | 失败，精度和效率都未达标 |
| Phase 3 | 修正 reference point 初始位置 | `xy` bias + `loss_ref_bias` | 进行中，需完成剩余实验 |

---

## 5. 组会可直接汇报的结论
1. 研究主线是正确的: Query 初始化相关改动确实有收益（Phase 1 已证实）。  
2. 配额机制不是当前数据/模型的有效方向（Phase 2 已被证伪）。  
3. 现阶段应集中验证 RefBias 是否能在不牺牲效率的前提下带来额外增益（Phase 3 正在进行）。  
4. 最终是否进入下一阶段，必须以完整 test 结果和门槛判定为准。

---

## 6. 结果来源
- Phase1:
  - `runs/analysis/phase1_lambda_sweep_seed0/sweep_train_summary.csv`
  - `runs/analysis/phase1_lambda_sweep_seed0/best_lambda_test_summary.csv`
- Phase2:
  - `runs/analysis/phase2_quota_sweep_seed0/phase2_test_summary.csv`
  - `runs/analysis/phase2_quota_sweep_seed0/comparison_vs_phase1.md`
- Phase3:
  - `runs/detect/runs/train/phase3_refbias_s003_lw02_seed0_e40/results.csv`
  - `runs/detect/runs/train/phase3_refbias_s006_lw02_seed0_e40/results.csv`
