import numpy as np

from src.common.types import Detection
from src.tracking.byte_track import ByteTrack


def box(x, width=100):
    return np.array([x, 0, x + width, 200], dtype=np.float32)


def test_constant_velocity_keeps_the_id_when_boxes_stop_overlapping():
    tracker = ByteTrack()
    first = tracker.update([Detection(box(0), 0.9)])
    second = tracker.update([Detection(box(50), 0.9)])
    third = tracker.update([Detection(box(130), 0.9)])
    assert first == []
    assert len(second) == 1
    assert len(third) == 1
    assert third[0].track_id == second[0].track_id


def test_doorway_track_starts_on_the_first_medium_confidence_box():
    bbox = np.array([450, 480, 550, 680], dtype=np.float32)
    started = ByteTrack().update([Detection(bbox, 0.4)], doorway_fn=lambda _bbox: True)
    rejected = ByteTrack().update([Detection(bbox, 0.4)])
    too_low = ByteTrack().update([Detection(bbox, 0.2)], doorway_fn=lambda _bbox: True)
    assert len(started) == 1
    assert started[0].state == "tentative"
    assert rejected == []
    assert too_low == []


def test_keypoints_follow_the_matched_detection():
    tracker = ByteTrack(min_hits=1)
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[0] = [40, 30, 0.9]
    visible = tracker.update([Detection(box(0), 0.9, keypoints=keypoints)])
    assert len(visible) == 1
    assert visible[0].head_box().shape == (4,)


def test_track_outside_the_door_waits_for_a_second_hit():
    tracker = ByteTrack()
    bbox = box(0, width=80)
    assert tracker.update([Detection(bbox, 0.9)]) == []
    visible = tracker.update([Detection(bbox, 0.9)])
    assert len(visible) == 1
    assert visible[0].hits >= 2
