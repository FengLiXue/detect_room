from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.events.state_machine import DoorEventStateMachine
from src.geometry.door import DoorGeometry
from src.input.video_source import OfflineVideoSource
from src.models.yolo_detector import YOLODetector
from src.output.sinks import EventJsonlSink, VideoWriterSink, draw_overlay
from src.tracking.byte_track import ByteTrack


def load_doors(config: dict, width: int, height: int) -> list[DoorGeometry]:
    return [DoorGeometry.from_dict(item).scale(width, height) for item in config.get("doors", [])]


def _in_doorway(doors: list[DoorGeometry], bbox: np.ndarray) -> bool:
    return any(door.overlaps_bbox(bbox) for door in doors)


@dataclass
class PipelineResult:
    counts: dict[str, int]


def run(video_path: str, config: dict, model_path: str, output_path: str | None, events_path: str, fps: float = 15.0, confidence: float = 0.35, image_size: int = 640, device: str | None = None, realtime: bool = True, display: bool = False) -> PipelineResult:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    doors = load_doors(config, width, height)
    detector = YOLODetector(model_path, confidence=confidence, image_size=image_size, device=device)
    tracker = ByteTrack()
    state_machine = DoorEventStateMachine(doors, fps=fps)
    source = OfflineVideoSource(video_path, target_fps=fps, realtime=realtime)
    writer = VideoWriterSink(output_path, fps, (width, height)) if output_path else None
    sink = EventJsonlSink(events_path)

    last_timestamp = 0.0
    try:
        for frame in source:
            last_timestamp = frame.timestamp
            detections = detector(frame.image)
            tracks = tracker.update(detections, doorway_fn=lambda bbox: _in_doorway(doors, bbox))
            events = state_machine.update(tracks, frame.index, frame.timestamp)
            for event in events:
                sink.write(event)
            if writer or display:
                annotated = draw_overlay(frame.image, doors, tracks, state_machine.counts(), events)
                if writer:
                    writer.write(annotated)
                if display:
                    cv2.imshow("doorway-occupancy", annotated)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
    finally:
        for event in state_machine.flush(last_timestamp):
            sink.write(event)
        sink.close()
        if writer:
            writer.close()
        if display:
            cv2.destroyAllWindows()
    return PipelineResult(state_machine.counts())
