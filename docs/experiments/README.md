# Experiment Notebook

## Project Context
- Task: DIOR 遥感目标检测，基线模型为 RT-DETR（Ultralytics 实现）。
- Dataset YAML: `/home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml`。
- Baseline run: `/home/fyb/mydir/ultralytics/runs/detect/runs/train/rtdetr_dior5`。
- Primary metric: `test mAP50-95(B)`。
- Secondary metrics: `test mAP50(B)`, `test mAP75(B)`, latency(ms/im), FPS。
- Decision rule: 仅当当前 Phase 门槛通过，才进入下一 Phase。

## Phase Timeline

| Phase | Topic | Status | Start Date | End Date | Notes |
|---|---|---|---|---|---|
| Phase 1 | Center-aware Query Reranking | Pass | 2026-03-13 | 2026-03-15 | λ 扫描完成，门槛通过 |
| Phase 2 | Per-level Query Quota | Running | 2026-03-16 | - | 当前在跑 `phase2_quota_none_lam035_seed0_e40` |
| Phase 3 | Reference Point Bias (planned) | Planned | - | - | 仅在 Phase 2 通过后启动 |

## Phase Summary Table

| Phase | Goal | Core Change | Best Result | Gate Decision | Detail |
|---|---|---|---|---|---|
| Phase 1 | 验证 `center_lambda_max` 是否提升中心先验重排效果 | `S_fuse = z(cls) + λ*z(center)` | test mAP50-95: `0.553459` (`λ=0.35`) | Pass | [phase1_center_rerank.md](./phase1_center_rerank.md) |
| Phase 2 | 验证固定配额的多尺度 query 选择是否进一步提升 | `query_quota_mode=fixed` + `query_level_ratios` | 进行中（首个对照 run 已到 epoch 6） | Pending | [phase2_quota.md](./phase2_quota.md) |

## Repro Commands

### Phase 1 (lambda sweep, seed=0)
```bash
cd /home/fyb/mydir/ultralytics
/home/fyb/envs/torch-cuda/bin/python examples/phase1_lambda_sweep_seed0.py \
  --model rtdetr-l.pt \
  --data /home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml
```

### Phase 2 (quota sweep, seed=0, sequential)
```bash
cd /home/fyb/mydir/ultralytics
/home/fyb/envs/torch-cuda/bin/python -u examples/train_rtdetr_dior.py \
  --model rtdetr-l.pt --data /home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml \
  --epochs 40 --imgsz 512 --batch 16 --device 0 --workers 1 --cache disk \
  --project runs/train --name phase2_quota_none_lam035_seed0_e40 --patience 8 \
  --query-rerank-mode center --center-lambda-max 0.35 --center-lambda-warmup-epochs 10 \
  --center-score-norm zscore_image --center-score-clip 6.0 --query-quota-mode none \
  --center-loss-weight 0.5 --center-pos-alpha 4.0 --center-empty-scale 0.25 \
  --center-target box_centerness --center-multi-gt-rule max --seed 0 --exist-ok
```

## Quality Check Commands
```bash
# 1) 检查关键记录文件是否存在
ls -la docs/experiments

# 2) 检查 Phase 1 引用结果文件
ls -la runs/analysis/phase1_lambda_sweep_seed0

# 3) 检查 Phase 2 当前 run 指标写入
tail -n 5 runs/detect/runs/train/phase2_quota_none_lam035_seed0_e40/results.csv

# 4) 检查 sweep 主进程是否仍在运行
pgrep -af "phase2_quota"
```

## Links
- Phase 1 Detail: [phase1_center_rerank.md](./phase1_center_rerank.md)
- Phase 2 Detail: [phase2_quota.md](./phase2_quota.md)
- Generic Phase Template: [templates/phase_template.md](./templates/phase_template.md)
- Experiment Entry Template: [templates/experiment_entry_template.md](./templates/experiment_entry_template.md)
- Phase 1 sweep CSV: [`runs/analysis/phase1_lambda_sweep_seed0/sweep_train_summary.csv`](../../runs/analysis/phase1_lambda_sweep_seed0/sweep_train_summary.csv)
- Phase 1 best test CSV: [`runs/analysis/phase1_lambda_sweep_seed0/best_lambda_test_summary.csv`](../../runs/analysis/phase1_lambda_sweep_seed0/best_lambda_test_summary.csv)
- Phase 1 comparison MD: [`runs/analysis/phase1_lambda_sweep_seed0/comparison_vs_baseline.md`](../../runs/analysis/phase1_lambda_sweep_seed0/comparison_vs_baseline.md)
- Phase 2 launch log: [`runs/launch_logs/phase2_quota_sweep_seed0_20260316-110432.log`](../../runs/launch_logs/phase2_quota_sweep_seed0_20260316-110432.log)
