# Experiment Entry Template

## Required Fields

| field | required | description |
|---|---|---|
| exp_id | yes | 实验编号，如 `P1-E1` |
| run_name | yes | 训练命名，与目录一致 |
| run_dir | yes | 绝对路径 |
| seed | yes | 随机种子 |
| key_hparams | yes | 本实验关键变量 |
| best_epoch | yes | 最优 epoch（若未完成填 `-`） |
| val_map50_95 | yes | 验证主指标 |
| test_map50_95 | yes | 测试主指标（无则 `-`） |
| latency_ms | yes | 时延（无则 `-`） |
| status | yes | `success/running/failed/queued/reused` |
| notes | yes | 异常、重试、备注 |

## Single Entry Snippet

| exp_id | run_name | run_dir | seed | key_hparams | best_epoch | val_map50_95 | test_map50_95 | latency_ms | status | notes |
|---|---|---|---:|---|---:|---:|---:|---:|---|---|
| PX-E1 | my_run | /abs/path/to/run | 0 | lambda=0.35 | 40 | 0.6566 | 0.5534 | 16.18 | success | - |

## Failure Entry Snippet

| exp_id | run_name | run_dir | seed | key_hparams | best_epoch | val_map50_95 | test_map50_95 | latency_ms | status | notes |
|---|---|---|---:|---|---:|---:|---:|---:|---|---|
| PX-E2 | my_failed_run | /abs/path/to/run | 0 | quota=0.6,0.25,0.15 | - | - | - | - | failed | OOM after epoch 2, retried batch=12 then failed |
