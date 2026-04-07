from ultralytics import RTDETR

model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-l-center-agv2.yaml")
model.train(
    data="/home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml",
    pretrained="rtdetr-l.pt",
    epochs=100,
    imgsz=512,
    batch=12,
    device=0,
    workers=1,
    cache="disk",
    seed=0,
    patience=10,
    fraction=1,
    query_rerank_mode="center",
    center_fusion_strategy="add",
    center_lambda_max=0.25,
    center_lambda_warmup_epochs=10,
    center_score_norm="zscore_image",
    center_score_clip=6.0,
    center_loss_weight=0.5,
    center_pos_alpha=4.0,
    center_empty_scale=0.25,
    center_target="box_centerness",
    center_multi_gt_rule="max",
    name="rtdetr_l_center_agv2",
)

