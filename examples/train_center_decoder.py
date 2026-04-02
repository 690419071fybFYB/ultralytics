from ultralytics import RTDETR

model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-l-center.yaml")

model.train(
    data="/home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml",
    pretrained="/home/fyb/mydir/ultralytics/rtdetr-l.pt",
    epochs=100,
    imgsz=512,
    batch=16,
    device=0,
    workers=1,
    cache="disk",
    seed=0,
    patience=10,
    query_rerank_mode="center",
    center_fusion_strategy="geom",
    center_lambda_max=0.35,
    center_lambda_warmup_epochs=10,
    center_score_norm="zscore_image",
    center_score_clip=6.0,
    center_loss_weight=0.5,
    center_pos_alpha=4.0,
    center_empty_scale=0.25,
    center_target="box_centerness",
    center_multi_gt_rule="max",
    name="rtdetr_l_center",
)
