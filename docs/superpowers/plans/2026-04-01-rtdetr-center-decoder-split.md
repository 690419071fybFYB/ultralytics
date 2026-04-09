# RT-DETR center decoder split plan

## Goal
Keep `RTDETRDecoder` as clean baseline and move center-aware behavior into a new subclass that can be selected from YAML.

## Files
- `tests/test_rtdetr_center_rerank.py`: adjust tests to distinguish baseline and center decoder behavior.
- `ultralytics/nn/modules/rtdetr_center_decoder.py`: new center-aware decoder subclass.
- `ultralytics/nn/modules/head.py`: restore baseline `RTDETRDecoder` implementation.
- `ultralytics/nn/modules/__init__.py`: export new decoder class.
- `ultralytics/nn/tasks.py`: import/register new decoder and keep center runtime/loss handling compatible.

## Steps
1. Write failing tests for baseline vs center decoder outputs and helpers.
2. Run focused pytest and verify failure.
3. Add new center decoder subclass file.
4. Restore baseline decoder in `head.py`.
5. Export/import/register the new class.
6. Run focused pytest until green.
