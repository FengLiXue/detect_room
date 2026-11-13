from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

HEAD_KEYPOINTS = (0, 1, 2, 3, 4)  # nose, left eye, right eye, left ear, right ear
ANKLE_KEYPOINTS = (15, 16)


@dataclass
class Detection:
    bbox: np.ndarray  # xyxy, float32
    confidence: float
    class_id: int = 0
    keypoints: np.ndarray | None = None  # (K, 3) = x, y, confidence


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # xyxy, float32
    confidence: float
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    state: str = "confirmed"
    metadata: dict[str, Any] = field(default_factory=dict)
    keypoints: np.ndarray | None = None  # (K, 3) = x, y, confidence

    def head_box(self, min_confidence: float = 0.3) -> np.ndarray:
        """用头部关键点围一个头框。关键点缺失时，改用人体框顶部。"""
        x1, y1, x2, y2 = (float(value) for value in self.bbox)
        body_height = max(y2 - y1, 1.0)
        body_width = max(x2 - x1, 1.0)
        points: list[tuple[float, float]] = []
        if self.keypoints is not None and len(self.keypoints) > HEAD_KEYPOINTS[-1]:
            for index in HEAD_KEYPOINTS:
                x, y, confidence = (float(value) for value in self.keypoints[index])
                if confidence >= min_confidence:
                    points.append((x, y))
        if points:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            span = max(max(xs) - min(xs), max(ys) - min(ys), body_height * 0.12, 8.0)
            pad = span * 0.85
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            return np.array([center_x - pad, center_y - pad, center_x + pad, center_y + pad], dtype=np.float32)
        head_height = max(body_height * 0.22, 8.0)
        head_width = min(body_width * 0.6, head_height)
        center_x = (x1 + x2) * 0.5
        return np.array([center_x - head_width / 2.0, y1, center_x + head_width / 2.0, y1 + head_height], dtype=np.float32)

    def head_point(self, min_confidence: float = 0.3) -> np.ndarray:
        """有鼻子用鼻子，否则用头框中心。"""
        if self.keypoints is not None and len(self.keypoints) > 0:
            x, y, confidence = (float(value) for value in self.keypoints[0])
            if confidence >= min_confidence:
                return np.array([x, y], dtype=np.float32)
        box = self.head_box(min_confidence)
        return np.array([(float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5], dtype=np.float32)

    def ankle_points(self, min_confidence: float = 0.3) -> list[tuple[float, float]]:
        """置信度够的脚踝。没检测到时返回空列表。"""
        if self.keypoints is None or len(self.keypoints) <= ANKLE_KEYPOINTS[-1]:
            return []
        points: list[tuple[float, float]] = []
        for index in ANKLE_KEYPOINTS:
            x, y, confidence = (float(value) for value in self.keypoints[index])
            if confidence >= min_confidence:
                points.append((x, y))
        return points


@dataclass
class DoorEvent:
    timestamp: float
    frame_index: int
    door_id: str
    room_id: str
    direction: str  # ENTER or EXIT
    track_id: int
    confidence: float
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "door_id": self.door_id,
            "room_id": self.room_id,
            "direction": self.direction,
            "track_id": self.track_id,
            "confidence": round(float(self.confidence), 4),
            "evidence": self.evidence,
        }
