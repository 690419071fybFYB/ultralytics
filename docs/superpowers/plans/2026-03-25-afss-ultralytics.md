# AFSS for Ultralytics YOLO Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前 `ultralytics` 仓库中实现论文 AFSS（Anti-Forgetting Sampling Strategy），让 YOLO 训练在不改模型结构的前提下支持按 epoch 动态采样训练图像。

**Architecture:** 采用“自定义 Trainer + 样本状态管理器 + 可变训练子集”方案。核心不改检测头和 loss，而是在训练调度层维护每张图像的 `precision`、`recall`、`last_used_epoch`，每个 epoch 依据论文规则重建 train dataloader；严格复现版本额外加入周期性的样本级评估，以更新全部训练图像的 sufficiency state。

**Tech Stack:** Python 3, PyTorch, Ultralytics YOLO trainer/dataloader stack, existing validator matching utilities

---

## Chunk 1: Design Lock

### Task 1: 固化 AFSS 论文约束和工程边界

**Files:**
- Create: `docs/superpowers/plans/2026-03-25-afss-ultralytics.md`
- Create: `docs/afss-notes.md`

- [ ] **Step 1: 记录论文里必须保真的规则**

写入以下固定规则，后续实现不得擅自改动：

```text
learning_sufficiency(i) = min(precision_i, recall_i)
easy:     sufficiency > 0.85
moderate: 0.55 <= sufficiency <= 0.85
hard:     sufficiency < 0.55

easy sampling:
- 每个 epoch 仅采样 easy 集的 2%
- forced review 子集不超过 easy 集 2% 的一半
- 长期未使用判定间隔默认 10 epochs

moderate sampling:
- 每个 epoch 采样 moderate 集的 40%
- 若样本连续 2 个 epoch 未被使用，则强制纳入
- 短期覆盖窗口默认 3 epochs

hard sampling:
- 每个 epoch 全量参与

state update:
- warmup 后每 5 epochs 刷新 precision/recall
```

- [ ] **Step 2: 定义两条实现路径**

写入以下实现分层：

```text
Phase A / MVP:
- 先打通 AFSS 调度与变长子集训练
- precision/recall 仅对“当期参与训练的样本”增量更新
- 未参与样本沿用旧状态
- 目标：验证代码结构、采样逻辑、训练可跑

Phase B / Paper-faithful:
- 周期性对整个 train set 做样本级评估
- 严格更新全部样本的 precision/recall
- 目标：尽量复现论文速度/精度曲线
```

- [ ] **Step 3: 明确首版不做的事情**

写入以下非目标：

```text
- 不修改 YOLO backbone/head/loss
- 不把 AFSS 强塞进所有 task，一版先覆盖 detect
- 不在第一阶段支持 classification
- 不追求一步到位复现 OBB/seg/pose
```

- [ ] **Step 4: 设计验收标准**

```text
1. 能通过 trainer 开关启用/关闭 AFSS
2. 日志可打印 easy/moderate/hard 数量与本 epoch 实际采样数
3. train loader 每个 epoch 的样本量可变化且训练不崩
4. DDP/single-GPU 均不因 sampler 长度变化而死锁
5. 关闭 AFSS 时行为与当前仓库一致
```


## Chunk 2: Sample Identity Plumbing

### Task 2: 给训练批次补充稳定的样本身份

**Files:**
- Modify: `ultralytics/data/base.py`
- Modify: `ultralytics/data/dataset.py`
- Test: `tests/data/test_afss_dataset_identity.py`

- [ ] **Step 1: 在 dataset label 中加入稳定 sample id**

计划修改 `BaseDataset.get_image_and_label()`，在每个样本字典里补：

```python
label["sample_idx"] = index
label["sample_key"] = label["im_file"]
```

要求：
- `sample_idx` 在单个 dataset 实例内稳定
- `sample_key` 优先使用 `im_file`，方便持久化状态
- 不破坏现有 transforms

- [ ] **Step 2: 更新 collate_fn 让 batch 保留身份字段**

计划修改 `YOLODataset.collate_fn()`，让下列字段原样保留：

```python
if k in {"im_file", "sample_key"}:
    new_batch[k] = list(value)
elif k == "sample_idx":
    new_batch[k] = torch.as_tensor(value, dtype=torch.long)
```

- [ ] **Step 3: 写失败测试验证 batch 身份可追踪**

```python
def test_afss_batch_contains_sample_identity():
    batch = next(iter(loader))
    assert "sample_idx" in batch
    assert "sample_key" in batch
    assert len(batch["sample_key"]) == batch["img"].shape[0]
```

- [ ] **Step 4: 运行测试确认只新增能力不改旧语义**

Run: `pytest tests/data/test_afss_dataset_identity.py -v`
Expected: PASS


## Chunk 3: AFSS State And Selector

### Task 3: 实现样本状态表与论文采样规则

**Files:**
- Create: `ultralytics/data/afss.py`
- Test: `tests/data/test_afss_selector.py`

- [ ] **Step 1: 定义状态数据结构**

计划新建：

```python
from dataclasses import dataclass

@dataclass
class AFSSSampleState:
    sample_key: str
    precision: float = 0.0
    recall: float = 0.0
    last_used_epoch: int = -1

    @property
    def sufficiency(self) -> float:
        return min(self.precision, self.recall)
```

- [ ] **Step 2: 定义配置对象**

```python
@dataclass
class AFSSConfig:
    enabled: bool = False
    warmup_epochs: int = 10
    easy_threshold: float = 0.85
    hard_threshold: float = 0.55
    easy_ratio: float = 0.02
    easy_forced_max_ratio: float = 0.5
    easy_review_gap: int = 10
    moderate_ratio: float = 0.40
    moderate_gap: int = 3
    state_update_interval: int = 5
    seed: int = 0
```

- [ ] **Step 3: 实现 difficulty partition**

```python
def split_samples(states):
    easy = [s for s in states if s.sufficiency > 0.85]
    moderate = [s for s in states if 0.55 <= s.sufficiency <= 0.85]
    hard = [s for s in states if s.sufficiency < 0.55]
    return easy, moderate, hard
```

- [ ] **Step 4: 实现 easy 组采样规则**

```python
target_easy = round(len(easy) * 0.02)
forced = [s for s in easy if epoch - s.last_used_epoch >= 10]
forced = random_sample(forced, min(len(forced), target_easy // 2))
random_easy = random_sample([s for s in easy if s not in forced], target_easy - len(forced))
```

- [ ] **Step 5: 实现 moderate 组采样规则**

```python
forced = [s for s in moderate if epoch - s.last_used_epoch >= 3]
target_moderate = round(len(moderate) * 0.40)
random_moderate = random_sample(
    [s for s in moderate if s not in forced],
    max(target_moderate - len(forced), 0),
)
```

- [ ] **Step 6: 实现 hard 组全量采样**

```python
selected_hard = hard
selected = forced_easy + random_easy + forced_moderate + random_moderate + selected_hard
```

- [ ] **Step 7: 写 selector 单测**

```python
def test_easy_sampling_caps_at_two_percent(): ...
def test_moderate_sampling_targets_forty_percent(): ...
def test_hard_samples_are_always_included(): ...
def test_long_unseen_easy_samples_get_priority(): ...
def test_stale_moderate_samples_get_forced_coverage(): ...
```

- [ ] **Step 8: 运行测试**

Run: `pytest tests/data/test_afss_selector.py -v`
Expected: PASS


## Chunk 4: Dynamic Train Loader

### Task 4: 让 Trainer 支持每个 epoch 变更训练子集

**Files:**
- Create: `ultralytics/models/yolo/detect/afss_trainer.py`
- Modify: `ultralytics/models/yolo/detect/__init__.py`
- Test: `tests/models/yolo/test_afss_trainer_loader.py`

- [ ] **Step 1: 新建 AFSSDetectionTrainer 继承 DetectionTrainer**

计划不要大改 `BaseTrainer`，而是新增独立 trainer：

```python
class AFSSDetectionTrainer(DetectionTrainer):
    def __init__(...):
        super().__init__(...)
        self.afss = AFSSManager(...)
```

- [ ] **Step 2: 给 dataset 增加 active_indices 视图**

首选方案：

```python
class AFSSSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, active_indices):
        self.base_dataset = base_dataset
        self.active_indices = list(active_indices)

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, i):
        return self.base_dataset[self.active_indices[i]]
```

原因：
- 不直接改 `BaseDataset.__len__`
- 不污染普通训练路径
- 可以按 epoch 重建 dataloader

- [ ] **Step 3: 在 epoch start 重建 train loader**

计划在 `on_train_epoch_start` 等效位置插入：

```python
selected_indices = self.afss.select_indices(epoch=self.epoch)
self.train_loader = self.build_afss_train_loader(selected_indices)
```

注意：
- 不能调用 `_build_train_pipeline()`，否则会重建 optimizer/scheduler
- 只重建 `train_loader`
- DDP 下每个 rank 必须收到同一组 `selected_indices`

- [ ] **Step 4: 写 loader 级 smoke test**

```python
def test_afss_trainer_rebuilds_train_loader_per_epoch():
    trainer = AFSSDetectionTrainer(...)
    first = trainer.afss.select_indices(epoch=0)
    second = trainer.afss.select_indices(epoch=1)
    assert first != second or len(first) == 0
```

- [ ] **Step 5: 运行测试**

Run: `pytest tests/models/yolo/test_afss_trainer_loader.py -v`
Expected: PASS


## Chunk 5: State Update Pipeline

### Task 5: 为训练样本维护 precision/recall 与 last_used_epoch

**Files:**
- Modify: `ultralytics/models/yolo/detect/afss_trainer.py`
- Create: `ultralytics/models/yolo/detect/afss_metrics.py`
- Test: `tests/models/yolo/test_afss_state_update.py`

- [ ] **Step 1: 先做 MVP 版状态更新**

在 `on_train_batch_end` 或 trainer batch loop 周边累积当期 batch 的样本使用情况：

```python
for sample_key in batch["sample_key"]:
    states[sample_key].last_used_epoch = current_epoch
```

并在可获得局部匹配结果时更新当前样本的 `precision/recall`。

- [ ] **Step 2: 锁定论文严格版的评估策略**

严格复现不应只更新本 epoch 被采样的样本，而应在以下时机对全部 train images 刷新：

```python
if epoch >= afss.warmup_epochs and (epoch - afss.warmup_epochs) % 5 == 0:
    refresh_all_train_sample_metrics()
```

- [ ] **Step 3: 复用 validator/matching 逻辑做样本级 Prec/Rec**

计划新建独立 helper：

```python
def evaluate_train_samples(model, dataset, device) -> dict[str, tuple[float, float]]:
    # 对每张图像单独统计 TP / FP / FN
    # precision = TP / (TP + FP + eps)
    # recall = TP / (TP + FN + eps)
```

要求：
- 指标必须是 image-level，不是 dataset aggregate
- 和 task=detect 当前 IoU/matching 规则一致
- 支持空预测与空标签边界

- [ ] **Step 4: 写状态更新测试**

```python
def test_last_used_epoch_updates_for_selected_samples(): ...
def test_sufficiency_is_min_precision_recall(): ...
def test_periodic_full_refresh_respects_interval(): ...
```

- [ ] **Step 5: 运行测试**

Run: `pytest tests/models/yolo/test_afss_state_update.py -v`
Expected: PASS


## Chunk 6: Config Surface And UX

### Task 6: 暴露 AFSS 配置开关，保证默认关闭时零影响

**Files:**
- Modify: `ultralytics/cfg/default.yaml`
- Create: `examples/train_yolo_afss.py`
- Test: `tests/cfg/test_afss_cfg.py`

- [ ] **Step 1: 在默认配置里加入 AFSS 参数**

```yaml
afss: false
afss_warmup_epochs: 10
afss_easy_threshold: 0.85
afss_hard_threshold: 0.55
afss_easy_ratio: 0.02
afss_easy_forced_max_ratio: 0.5
afss_easy_review_gap: 10
afss_moderate_ratio: 0.40
afss_moderate_gap: 3
afss_state_update_interval: 5
```

- [ ] **Step 2: 提供最小使用示例**

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect.afss_trainer import AFSSDetectionTrainer

model = YOLO("yolo11n.pt")
model.train(
    data="coco8.yaml",
    epochs=10,
    trainer=AFSSDetectionTrainer,
    afss=True,
)
```

- [ ] **Step 3: 配置测试**

```python
def test_afss_defaults_are_disabled(): ...
def test_afss_custom_args_are_parsed(): ...
```

- [ ] **Step 4: 运行测试**

Run: `pytest tests/cfg/test_afss_cfg.py -v`
Expected: PASS


## Chunk 7: End-to-End Verification

### Task 7: 做最小数据集验证和性能 sanity check

**Files:**
- Test: `tests/integration/test_afss_train_smoke.py`
- Create: `docs/afss-notes.md`

- [ ] **Step 1: 在小数据集上跑 1-3 epoch smoke**

Run:

```bash
pytest tests/integration/test_afss_train_smoke.py -v
```

Expected:

```text
- baseline trainer can still train
- AFSS trainer can train
- AFSS logs selected subset size each epoch
```

- [ ] **Step 2: 手工做一次最小训练命令**

Run:

```bash
python examples/train_yolo_afss.py
```

Expected:

```text
- 第一个 warmup 阶段接近全量训练
- warmup 后每个 epoch 的 selected images 数开始下降
- hard sample count 始终被纳入
```

- [ ] **Step 3: 记录观测结果**

在 `docs/afss-notes.md` 记录：

```text
- 单卡/多卡是否稳定
- 样本级评估耗时占比
- 与 baseline 相比每个 epoch 实际节省多少样本
- 是否需要把全量 state refresh 改为独立 no_grad pass
```


## Recommended Execution Order

1. 先做 Chunk 2 和 Chunk 3，打通 sample identity 与 selector。
2. 再做 Chunk 4，让 detect trainer 跑通动态子集训练。
3. 先用 Chunk 5 的 MVP 版状态更新做 smoke，确认结构成立。
4. 最后补 Chunk 5 的严格版全量样本评估，追论文复现度。
5. detect 稳定后，再决定是否扩展到 `ultralytics/models/yolo/obb/train.py`。

## Main Risks

- 最大风险不是采样逻辑，而是“样本级 precision/recall 怎么高效算”。论文的严格版需要周期性刷新全训练集状态，这一步很可能需要单独的 `no_grad` train-set eval pass。
- 次大风险是每个 epoch 动态重建 train loader 对 DDP 的影响。实现时必须保证各 rank 使用完全一致的 `selected_indices`。
- 如果直接在现有 dataset 上改 `__len__`/`__getitem__` 行为，容易破坏普通训练路径；因此更推荐独立 `AFSSSubsetDataset` 包装器。
- 论文声称“几乎无额外开销”是方法层表述；在仓库实现里，真实开销取决于样本级指标刷新方式，不能先假设它免费。

## Decision

这篇 AFSS **可以代码实现**，而且和当前 Ultralytics 框架是相容的，因为它本质上是训练调度层策略，不要求改网络结构。

但如果目标是“先做出能跑版本”，建议按下面顺序推进：

```text
MVP:
- detect only
- 自定义 trainer
- 动态 train subset
- 仅增量更新当前参与样本状态

paper-faithful:
- 周期性全 train set 样本级评估
- 严格按论文 2% / 40% / all 规则与 10 / 3 / 5 interval 刷新
- 再扩展到 OBB
```
