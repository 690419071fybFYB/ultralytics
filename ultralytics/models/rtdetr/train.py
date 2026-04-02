# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy

import torch.nn as nn

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import CenterRTDETRDecoder
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr

from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):
    """Trainer class for the RT-DETR model developed by Baidu for real-time object detection.

    This class extends the DetectionTrainer class for YOLO to adapt to the specific features and architecture of
    RT-DETR. The model leverages Vision Transformers and has capabilities like IoU-aware query selection and adaptable
    inference speed.

    Attributes:
        loss_names (tuple): Names of the loss components used for training.
        data (dict): Dataset configuration containing class count and other parameters.
        args (dict): Training arguments and hyperparameters.
        save_dir (Path): Directory to save training results.
        test_loader (DataLoader): DataLoader for validation/testing data.

    Methods:
        get_model: Initialize and return an RT-DETR model for object detection tasks.
        build_dataset: Build and return an RT-DETR dataset for training or validation.
        get_validator: Return a DetectionValidator suitable for RT-DETR model validation.

    Examples:
        >>> from ultralytics.models.rtdetr.train import RTDETRTrainer
        >>> args = dict(model="rtdetr-l.yaml", data="coco8.yaml", imgsz=640, epochs=3)
        >>> trainer = RTDETRTrainer(overrides=args)
        >>> trainer.train()

    Notes:
        - F.grid_sample used in RT-DETR does not support the `deterministic=True` argument.
        - AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.
    """

    def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
        """Initialize and return an RT-DETR model for object detection tasks.

        Args:
            cfg (dict, optional): Model configuration.
            weights (str, optional): Path to pre-trained model weights.
            verbose (bool): Verbose logging if True.

        Returns:
            (RTDETRDetectionModel): Initialized model.
        """
        model = RTDETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """Set model and RT-DETR-head attributes from dataset and train args."""
        super().set_model_attributes()
        head = self.model.model[-1]
        if not isinstance(head, CenterRTDETRDecoder):
            return
        # Force-set for backward compatibility with old checkpoints that may not contain new attrs.
        head.query_rerank_mode = self.args.query_rerank_mode
        head.center_fusion_strategy = self.args.center_fusion_strategy
        head.center_lambda_max = self.args.center_lambda_max
        head.center_lambda_warmup_epochs = self.args.center_lambda_warmup_epochs
        head.center_score_norm = self.args.center_score_norm
        head.center_score_clip = self.args.center_score_clip
        if not hasattr(head, "enc_center_head"):
            head.enc_center_head = nn.Linear(head.hidden_dim, 1).to(head.enc_score_head.weight.device)

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build and return an RT-DETR dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, either 'train' or 'val'.
            batch (int, optional): Batch size for rectangle training.

        Returns:
            (RTDETRDataset): Dataset object for the specific mode.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def preprocess_batch(self, batch):
        """Preprocess batch and append epoch metadata for head-side lambda warmup."""
        batch = super().preprocess_batch(batch)
        batch["epoch"] = self.epoch
        batch["num_epochs"] = self.epochs
        return batch

    def get_validator(self):
        """Return an RTDETRValidator suitable for RT-DETR model validation."""
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
