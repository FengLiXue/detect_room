import numpy as np
import pytest

from src.geometry.door import DoorGeometry


def _door():
    return DoorGeometry.from_dict({
        "door_id": "d1",
        "room_id": "r1",
        "room_name": "room",
        "polygon": [[0.2, 0.3], [0.8, 0.3], [0.8, 0.7], [0.2, 0.7]],
    }).scale(1000, 1000)


def test_polygon_contains_the_door_opening():
    door = _door()
    assert door.contains((500, 500))
    assert door.contains((200, 300))
    assert not door.contains((500, 900))
    assert not door.contains((40, 500))


def test_bbox_overlap_uses_the_polygon():
    door = _door()
    assert door.overlaps_bbox(np.array([450, 400, 550, 650], dtype=np.float32))
    assert not door.overlaps_bbox(np.array([0, 800, 40, 900], dtype=np.float32))


def test_feet_far_below_the_threshold_are_in_front_of_the_door():
    door = _door()
    assert door.feet_near_threshold(np.array([450, 400, 550, 700], dtype=np.float32))
    assert door.feet_near_threshold(np.array([450, 400, 550, 760], dtype=np.float32))
    assert not door.feet_near_threshold(np.array([450, 200, 550, 900], dtype=np.float32))
    assert not door.feet_near_threshold(np.array([450, 100, 550, 500], dtype=np.float32))


def test_zero_area_polygon_is_rejected():
    with pytest.raises(ValueError):
        DoorGeometry.from_dict({
            "door_id": "d1",
            "room_id": "r1",
            "polygon": [[0.2, 0.2], [0.2, 0.2], [0.2, 0.2], [0.2, 0.2]],
        })


def test_initial_count_cannot_be_negative():
    with pytest.raises(ValueError):
        DoorGeometry.from_dict({
            "door_id": "d1",
            "room_id": "r1",
            "polygon": [[0.2, 0.3], [0.8, 0.3], [0.8, 0.7], [0.2, 0.7]],
            "initial_count": -1,
        })
