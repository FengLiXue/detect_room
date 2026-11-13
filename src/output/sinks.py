from __future__ import annotations

import json
from pathlib import Path

import cv2

from src.common.types import DoorEvent, Track
from src.geometry.door import DoorGeometry


class EventJsonlSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")

    def write(self, event: DoorEvent) -> None:
        self.file.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def draw_overlay(image, doors: list[DoorGeometry], tracks: list[Track], counts: dict[str, int], events: list[DoorEvent]):
    output = image.copy()
    for door in doors:
        door.draw(output)
        anchor = door.polygon.mean(axis=0).astype(int)
        count = int(counts.get(door.room_id, 0))
        cv2.putText(output, f"{door.room_name}: {count}", tuple(anchor), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    for track in tracks:
        x1, y1, x2, y2 = track.bbox.astype(int)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(output, f"ID {track.track_id}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2, cv2.LINE_AA)
        hx1, hy1, hx2, hy2 = track.head_box().astype(int)
        cv2.rectangle(output, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)

    for index, event in enumerate(events):
        text = f"{event.direction} {event.room_id} ID={event.track_id}"
        cv2.putText(output, text, (10, 25 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    return output


class VideoWriterSink:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"无法创建输出视频：{self.path}")

    def write(self, image) -> None:
        self.writer.write(image)

    def close(self) -> None:
        self.writer.release()
