from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.common.types import Detection, Track


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def _center(bbox: np.ndarray) -> tuple[float, float]:
    return float((bbox[0] + bbox[2]) * 0.5), float((bbox[1] + bbox[3]) * 0.5)


def _size(bbox: np.ndarray) -> tuple[float, float]:
    return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])


def _shift(bbox: np.ndarray, vx: float, vy: float, vw: float, vh: float) -> np.ndarray:
    cx, cy = _center(bbox)
    w, h = _size(bbox)
    cx += vx
    cy += vy
    w = max(1.0, w + vw)
    h = max(1.0, h + vh)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


@dataclass
class _InternalTrack:
    track_id: int
    bbox: np.ndarray
    confidence: float
    matched_cx: float
    matched_cy: float
    matched_w: float
    matched_h: float
    born_in_doorway: bool = False
    keypoints: np.ndarray | None = None
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vw: float = 0.0
    vh: float = 0.0


class ByteTrack:
    """两阶段 IoU 关联，并用上一帧速度预测下一帧位置。

    门口区域内置信度达到 doorway_threshold 的框可以立即建轨迹；
    其余位置仍要高置信度框，且至少命中 min_hits 次才输出。
    """

    def __init__(self, high_threshold: float = 0.5, low_threshold: float = 0.1, doorway_threshold: float = 0.35, match_iou: float = 0.3, max_lost: int = 15, min_hits: int = 2):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.doorway_threshold = doorway_threshold
        self.match_iou = match_iou
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks: list[_InternalTrack] = []

    def _associate(self, tracks: list[_InternalTrack], detections: list[Detection], min_iou: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        track_boxes = np.asarray([t.bbox for t in tracks], dtype=np.float32)
        det_boxes = np.asarray([d.bbox for d in detections], dtype=np.float32)
        cost = 1.0 - iou_matrix(track_boxes, det_boxes)
        rows, cols = linear_sum_assignment(cost)
        matches = []
        used_t, used_d = set(), set()
        for r, c in zip(rows, cols):
            if 1.0 - cost[r, c] >= min_iou:
                matches.append((r, c))
                used_t.add(r)
                used_d.add(c)
        unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
        unmatched_d = [i for i in range(len(detections)) if i not in used_d]
        return matches, unmatched_t, unmatched_d

    def _in_doorway(self, bbox: np.ndarray, doorway_fn: Callable[[np.ndarray], bool] | None) -> bool:
        return bool(doorway_fn is not None and doorway_fn(bbox))

    def _spawn(self, det: Detection, in_doorway: bool) -> None:
        cx, cy = _center(det.bbox)
        w, h = _size(det.bbox)
        self.tracks.append(
            _InternalTrack(
                self.next_id,
                det.bbox.astype(np.float32, copy=True),
                det.confidence,
                cx,
                cy,
                w,
                h,
                born_in_doorway=in_doorway,
                keypoints=None if det.keypoints is None else det.keypoints.copy(),
            )
        )
        self.next_id += 1

    def _update_matched(self, track: _InternalTrack, det: Detection) -> None:
        cx, cy = _center(det.bbox)
        w, h = _size(det.bbox)
        dt = max(track.time_since_update, 1)
        blend = 0.75
        track.vx = blend * ((cx - track.matched_cx) / dt) + (1.0 - blend) * track.vx
        track.vy = blend * ((cy - track.matched_cy) / dt) + (1.0 - blend) * track.vy
        track.vw = blend * ((w - track.matched_w) / dt) + (1.0 - blend) * track.vw
        track.vh = blend * ((h - track.matched_h) / dt) + (1.0 - blend) * track.vh
        track.matched_cx = cx
        track.matched_cy = cy
        track.matched_w = w
        track.matched_h = h
        track.bbox = det.bbox.astype(np.float32, copy=True)
        track.keypoints = None if det.keypoints is None else det.keypoints.copy()
        track.confidence = det.confidence
        track.hits += 1
        track.time_since_update = 0

    def update(self, detections: Iterable[Detection], doorway_fn: Callable[[np.ndarray], bool] | None = None) -> list[Track]:
        detections = list(detections)
        high = [d for d in detections if d.confidence >= self.high_threshold]
        low = [d for d in detections if self.low_threshold <= d.confidence < self.high_threshold]

        for track in self.tracks:
            track.bbox = _shift(track.bbox, track.vx, track.vy, track.vw, track.vh)
            track.age += 1
            track.time_since_update += 1

        matches, unmatched_tracks, _ = self._associate(self.tracks, high, self.match_iou)
        for ti, di in matches:
            self._update_matched(self.tracks[ti], high[di])

        # 未配上的轨迹再用低置信度框续上，避免门口遮挡直接断轨。
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        low_matches, _, _ = self._associate(remaining_tracks, low, self.match_iou * 0.8)
        for local_ti, di in low_matches:
            self._update_matched(remaining_tracks[local_ti], low[di])

        used_high = {di for _, di in matches}
        used_low = {di for _, di in low_matches}
        for di, det in enumerate(high):
            if di not in used_high:
                self._spawn(det, self._in_doorway(det.bbox, doorway_fn))
        for di, det in enumerate(low):
            if di in used_low or det.confidence < self.doorway_threshold:
                continue
            if self._in_doorway(det.bbox, doorway_fn):
                self._spawn(det, True)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost]
        visible: list[Track] = []
        for track in self.tracks:
            if track.time_since_update != 0:
                continue
            if track.hits < self.min_hits and not track.born_in_doorway:
                continue
            visible.append(
                Track(
                    track_id=track.track_id,
                    bbox=track.bbox.copy(),
                    confidence=track.confidence,
                    age=track.age,
                    hits=track.hits,
                    time_since_update=track.time_since_update,
                    state="confirmed" if track.hits >= self.min_hits else "tentative",
                    keypoints=None if track.keypoints is None else track.keypoints.copy(),
                )
            )
        return visible
