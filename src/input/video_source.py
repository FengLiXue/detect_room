from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np


@dataclass
class VideoFrame:
    index: int
    timestamp: float
    image: np.ndarray


def advance_sample_phase(phase: float, source_fps: float, target_fps: float) -> tuple[bool, float]:
    """按源帧率与目标帧率的比例决定这一帧要不要送给检测器。"""
    emit = phase + 1e-6 >= source_fps
    if emit:
        phase -= source_fps
    return emit, phase + target_fps


class OfflineVideoSource:
    """读取本地视频。按目标 FPS 抽帧，时间戳用视频 PTS，处理落后时丢帧。"""

    def __init__(self, path: str | Path, target_fps: float = 15.0, realtime: bool = True, clock: Callable[[], float] | None = None, sleeper: Callable[[float], None] | None = None):
        self.path = str(path)
        self.target_fps = float(target_fps)
        self.realtime = realtime
        self._clock = clock or time.monotonic
        self._sleep = sleeper or time.sleep

    def __iter__(self) -> Iterator[VideoFrame]:
        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            raise FileNotFoundError(f"无法打开视频：{self.path}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps < 1.0 or source_fps > 240.0:
            source_fps = self.target_fps if self.target_fps > 1e-6 else 15.0
        target_fps = self.target_fps if self.target_fps > 1e-6 else source_fps
        period = 1.0 / target_fps
        phase = source_fps
        started = self._clock()
        source_index = 0
        last_pts_msec = -1.0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                pts_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if pts_msec > last_pts_msec + 0.1:
                    timestamp = pts_msec / 1000.0
                    last_pts_msec = pts_msec
                else:
                    timestamp = source_index / source_fps
                emit, phase = advance_sample_phase(phase, source_fps, target_fps)
                frame_index = source_index
                source_index += 1
                if not emit:
                    continue
                if self.realtime:
                    lag = self._clock() - started - timestamp
                    if lag > period:
                        continue
                    if lag < 0:
                        self._sleep(-lag)
                yield VideoFrame(frame_index, timestamp, image)
        finally:
            capture.release()
