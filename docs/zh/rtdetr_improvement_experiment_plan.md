# RT-DETR 双方向联动改进实验计划 v2（DIOR / rtdetr-l / 延迟退化<=5%）

## 0. 本版修订点

1. 锁定 MUQS 的分数标度规则，避免三项分数量纲不一致。  
2. 明确 `mu_heuristic` 与 `mu_learned` 的监督与实现边界。  
3. 在两阶段之间增加“小复核”环节，降低单 seed 误筛风险。  
4. 固定延迟测试口径，保证“<=5%”判定可复现。  

---

## 1. 目标与约束

- 目标：在不破坏 RT-DETR 实时性的前提下，联合验证 `SSM-AIFI` 与 `MUQS` 的精度与稳定性增益。  
- 主数据集：`DIOR`  
- 主模型：`rtdetr-l`  
- 硬约束：单卡 `batch=1` 端到端延迟退化 `<=5%`。  
- 推进策略：两阶段淘汰 + 中间复核。  

---

## 2. 基线定义（必须先锁定）

为避免结论漂移，保留两个基线：

1. `E0a`：原始 `rtdetr-l`（官方路径）  
2. `E0b`：当前你项目内最强稳定基线（若已验证优于 E0a，则后续门槛以 E0b 为主比较对象）  

说明：后续所有 `ΔmAP` 默认相对 `E0b` 计算；报告中同时保留对 `E0a` 的差值列。  

---

## 3. 改进方向与实现范围

### 3.1 编码器方向：SSM-AIFI 分层策略

- `S5`：`AIFI + SSM` 双通路（主改动）
- `S4`：仅加 `SSM-Lite`
- `S3`：默认关闭，仅在延迟预算富余时尝试 `SSM-tiny`

首轮默认超参：

- `S5 ssm_ratio = 0.33`
- `S4 ssm_ratio = 0.25`
- `S3 tiny ssm_ratio = 0.125`（默认关闭）

实现约束：

- `S5` 先落地，`S4` 作为第二步增量，不允许首轮同时打开 `S3`。  
- 任一 SSM 变体若单独引入即导致延迟退化 `>5%`，直接淘汰。  

### 3.2 Query 选择方向：MUQS（统一标度后融合）

在 query top-k 前定义：

- `S_cls(i)`：token 的类别置信分（如 `max_c logit_cls(i,c)`）  
- `S_match(i)`：token 的可匹配性分  
- `S_unc(i)`：token 的不确定性分  

每张图内做 `z-score` 标准化：

- `Z(x) = (x-mean(x))/sqrt(var(x)+1e-6)`，若 `var<1e-8` 则置零。  

融合分数：

- `S_fuse = Z(S_match) - lambda * Z(S_unc) + beta * Z(S_cls)`  
- 默认：`lambda=0.2, beta=1.0`  
- 可选 `clip(S_fuse, [-6,6])` 防极值抢占。  

模式开关：

1. `maxcls`：`S_fuse = Z(S_cls)`（基线）  
2. `mu_heuristic`：无新增监督，快速筛选  
3. `mu_learned`：新增监督（完整方案）  

### 3.3 MUQS 监督定义（仅 mu_learned）

新增两个 token 级头：

- `match_head -> S_match`  
- `uncert_head -> S_unc`

监督目标（连续值）：

- `y_match(i) = max_j IoU(pred_box_i, gt_box_j)`  
- `y_unc(i) = 1 - y_match(i)`  

损失：

- `loss_match = BCEWithLogits(S_match, y_match)`  
- `loss_uncert = BCEWithLogits(S_unc, y_unc)`  
- 默认权重：`w_match=0.5, w_uncert=0.25`

训练稳定策略：

1. 前 `5` epoch 可冻结 `uncert_head`（可选开关）  
2. 出现抖动或 NaN：先降 `w_uncert`，再冻结 `uncert_head`  

---

## 4. 三段式实验设计

### 4.1 阶段 A：筛选（30 epoch，seed=0）

实验矩阵：

1. `E0a` 原始基线  
2. `E0b` 当前最强基线  
3. `E1` MUQS-Heuristic  
4. `E2` MUQS-Learned  
5. `E3` SSM-AIFI-S5  
6. `E4` SSM-AIFI-S5+S4Lite  
7. `E5` Combo（`E2` + `E3/E4` 中最优者）

阶段 A 晋级门槛：

- `ΔmAP50-95 >= +0.3`（对 E0b）
- 延迟退化 `<=5%`
- 无训练异常（NaN / loss 爆炸 / 分支塌缩）

### 4.2 阶段 A'：小复核（20 epoch，seed=1）

- 仅对阶段 A 前 2 名方案复跑  
- 要求：相对 E0b 的提升方向一致（不允许符号反转）  
- 若方向反转，降级为“低置信候选”，不得直入阶段 B  

### 4.3 阶段 B：复验（100 epoch，seed=0/1）

- 仅保留通过 A' 的前 2 名  
- 报告均值与方差  
- 若两者差异 `<0.2 mAP`，优先延迟更低者  
- 若仍超延迟红线，回退到 `MUQS-only` 主线  

---

## 5. 验收门槛（最终判定）

候选主线必须同时满足：

1. 精度：阶段 B 相对 E0b，`ΔmAP50-95 >= +0.5`  
2. 小目标保护：`ΔAPs >= -0.1`  
3. 效率：延迟退化 `<=5%`  
4. 稳定性：无持续分支塌缩、无部署阻塞  

任一硬门槛不满足即判定 fail。  

---

## 6. 评测与记录口径（固定）

### 6.1 指标

- 精度：`mAP50-95`, `APs`, `APm`, `APl`  
- 性能：`latency(ms)`, `FPS`, 峰值显存  
- 可靠性：`ECE`（至少覆盖 `E0b`, `E2`, `E5`）  

### 6.2 延迟测试口径

- 环境固定：同 GPU、同驱动、同精度模式、同输入尺寸  
- `warmup=100` 次，`measure=300` 次  
- 每次测量前后 `torch.cuda.synchronize()`  
- 报告 `preprocess / infer / postprocess / total`  

### 6.3 每个实验的三件套

1. 配置与训练日志  
2. 指标汇总表（val/test）  
3. 延迟报告（含测试脚本与口径）  

---

## 7. 风险与回滚

1. 风险：理论复杂度降了但真实延迟升了  
   - 对策：以真实端到端延迟为准，超线即淘汰  
2. 风险：MUQS-Learned 不稳定  
   - 对策：降 `w_uncert`，必要时冻结 `uncert_head`  
3. 风险：SSM 分支收益不稳  
   - 对策：先 `S5-only`，再增量 `S4`，不跨级联动  
4. 风险：部署链路失败  
   - 对策：每个晋级候选必须做一次导出+推理 smoke  

---

## 8. 推荐执行顺序（务实版）

1. 先做 `MUQS-Heuristic`（低风险拿信号）  
2. 再做 `S5-only`（先看延迟边界）  
3. 再上 `MUQS-Learned`（补完整机制）  
4. 最后做 `Combo`（避免一开始变量过多）  

这套顺序能最大化“可解释性 + 成功率 + 工程可控性”。  
