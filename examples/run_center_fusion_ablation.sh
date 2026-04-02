#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/fyb/envs/torch-cuda/bin/python}"
SCRIPT="examples/train_center_decoder_ablation.py"

DATA="${DATA:-/home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml}"
MODEL="${MODEL:-ultralytics/cfg/models/rt-detr/rtdetr-l-center.yaml}"
PRETRAINED="${PRETRAINED:-rtdetr-l.pt}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-512}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-1}"
CACHE="${CACHE:-disk}"
PROJECT="${PROJECT:-runs/train}"
SEED="${SEED:-0}"
FRACTION="${FRACTION:-1.0}"

FUSIONS=(${FUSIONS:-add geom})
LAMBDAS=(${LAMBDAS:-0.15 0.25 0.35 0.45 0.55 0.75 1.0})

for fusion in "${FUSIONS[@]}"; do
  for lam in "${LAMBDAS[@]}"; do
    lam_tag="${lam/./}"
    name="center_${fusion}_lam${lam_tag}_seed${SEED}"
    echo "Running ${name}"
    "${PYTHON_BIN}" "${SCRIPT}" \
      --model "${MODEL}" \
      --pretrained "${PRETRAINED}" \
      --data "${DATA}" \
      --epochs "${EPOCHS}" \
      --imgsz "${IMGSZ}" \
      --batch "${BATCH}" \
      --device "${DEVICE}" \
      --workers "${WORKERS}" \
      --cache "${CACHE}" \
      --project "${PROJECT}" \
      --seed "${SEED}" \
      --fraction "${FRACTION}" \
      --center-fusion-strategy "${fusion}" \
      --center-lambda-max "${lam}" \
      --name "${name}" \
      --exist-ok
  done
done
