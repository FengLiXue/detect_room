from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


def _quad(values: Iterable[Iterable[float]]) -> np.ndarray:
    points = np.asarray(list(values), dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("门洞必须是四个二维点")
    return points


def _area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


@dataclass
class DoorGeometry:
    door_id: str
    room_id: str
    room_name: str
    polygon: np.ndarray
    initial_count: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "DoorGeometry":
        initial_count = int(data.get("initial_count", 0))
        if initial_count < 0:
            raise ValueError(f"{data.get('door_id', '?')}: initial_count 不能为负数")
        door = cls(
            door_id=str(data["door_id"]),
            room_id=str(data["room_id"]),
            room_name=str(data.get("room_name", data["room_id"])),
            polygon=_quad(data["polygon"]),
            initial_count=initial_count,
        )
        door.validate()
        return door

    def validate(self) -> None:
        if _area(self.polygon) < 1e-8:
            raise ValueError(f"{self.door_id}: 门洞四边形面积不能为零")

    def scale(self, width: int, height: int) -> "DoorGeometry":
        return DoorGeometry(
            door_id=self.door_id,
            room_id=self.room_id,
            room_name=self.room_name,
            polygon=self.polygon * np.array([width, height], dtype=np.float32),
            initial_count=self.initial_count,
        )

    def door_height(self) -> float:
        return float(self.polygon[:, 1].max() - self.polygon[:, 1].min())

    def threshold_y(self, x: float) -> float:
        """门洞下沿在某个 x 上的画面高度。下沿越靠画面下方，门越靠近镜头。"""
        order = np.argsort(self.polygon[:, 1])
        first = self.polygon[int(order[-1])]
        second = self.polygon[int(order[-2])]
        x1, y1 = float(first[0]), float(first[1])
        x2, y2 = float(second[0]), float(second[1])
        if abs(x2 - x1) < 1e-6:
            return max(y1, y2)
        ratio = (float(x) - x1) / (x2 - x1)
        ratio = min(1.0, max(0.0, ratio))
        return y1 + ratio * (y2 - y1)

    def feet_near_threshold(self, bbox: np.ndarray, margin_ratio: float = 0.25) -> bool:
        """脚在门洞下沿附近。明显更低的人在镜头和门之间，只是挡住了门。"""
        foot_x = (float(bbox[0]) + float(bbox[2])) * 0.5
        foot_y = float(bbox[3])
        lip = self.threshold_y(foot_x)
        margin = max(self.door_height() * margin_ratio, 8.0)
        return lip - margin <= foot_y <= lip + margin

    def part_at_door(self, point: Iterable[float], margin_ratio: float = 0.25) -> bool:
        """点在门洞内，或贴着下沿。窄门洞装不下的脚踝仍算到了门口。"""
        if self.contains(point):
            return True
        x, y = float(point[0]), float(point[1])
        margin = max(self.door_height() * margin_ratio, 8.0)
        return abs(y - self.threshold_y(x)) <= margin

    def contains(self, point: Iterable[float]) -> bool:
        x, y = float(point[0]), float(point[1])
        contour = self.polygon.reshape((-1, 1, 2))
        return cv2.pointPolygonTest(contour, (x, y), False) >= 0

    def overlaps_bbox(self, bbox: np.ndarray) -> bool:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        probes = (
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
            ((x1 + x2) * 0.5, y2),
            ((x1 + x2) * 0.5, (y1 + y2) * 0.5),
        )
        if any(self.contains(point) for point in probes):
            return True
        return any(x1 <= float(px) <= x2 and y1 <= float(py) <= y2 for px, py in self.polygon)

    def draw(self, image: np.ndarray) -> None:
        pts = self.polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (0, 180, 255), 2, cv2.LINE_AA)
