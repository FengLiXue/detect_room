from __future__ import annotations

from typing import Iterable

import numpy as np

from src.common.types import Detection


class YOLODetector:
    def __init__(self, model_path: str = "yolo11n-pose.pt", confidence: float = 0.35, image_size: int = 640, device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("未安装 ultralytics，请先执行 pip install -r requirements.txt") from exc

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def __call__(self, image: np.ndarray) -> list[Detection]:
        kwargs = {
            "source": image,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "classes": [0],  # COCO person
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        result = self.model.predict(**kwargs)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confs = result.boxes.conf.detach().cpu().numpy().astype(float)
        keypoints = self._keypoints(result, len(boxes))
        return [
            Detection(bbox=box, confidence=float(score), class_id=0, keypoints=points)
            for box, score, points in zip(boxes, confs, keypoints)
        ]

    @staticmethod
    def _keypoints(result, count: int) -> list[np.ndarray | None]:
        raw = getattr(result, "keypoints", None)
        if raw is None or raw.xy is None or len(raw.xy) != count:
            return [None] * count
        xy = raw.xy.detach().cpu().numpy().astype(np.float32)
        confidence = getattr(raw, "conf", None)
        if confidence is None:
            scores = np.ones(xy.shape[:2], dtype=np.float32)
        else:
            scores = confidence.detach().cpu().numpy().astype(np.float32)
        return [np.concatenate([xy[index], scores[index, :, None]], axis=1) for index in range(count)]
