import numpy as np

import pytest

from src.common.types import Track
from src.events.state_machine import DoorEventStateMachine
from src.geometry.door import DoorGeometry


def door_quad(room, door_id="d1", x0=0.2, x1=0.8, y0=0.45, y1=0.75, initial=0):
    return DoorGeometry.from_dict({
        "door_id": door_id,
        "room_id": room,
        "room_name": room,
        "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "initial_count": initial,
    }).scale(1000, 1000)


def track(track_id, head, confidence=0.9, bbox=None):
    x, y = head
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[0] = [x, y, 0.9]
    keypoints[1] = [x - 6, y - 4, 0.9]
    keypoints[2] = [x + 6, y - 4, 0.9]
    if bbox is None:
        bbox = np.array([x - 40, 550.0, x + 40, 750.0], dtype=np.float32)
    return Track(track_id, np.asarray(bbox, dtype=np.float32), confidence, hits=5, keypoints=keypoints)


INSIDE = (500, 600)
OUTSIDE = (500, 900)


def test_enter_and_exit_update_count():
    machine = DoorEventStateMachine([door_quad("r1")], stable_frames=1, lost_seconds=0.7)
    assert machine.update([track(1, OUTSIDE)], 0, 0.0) == []
    assert machine.update([track(1, INSIDE)], 1, 0.1) == []
    events = machine.update([], 2, 0.9)
    assert events[0].direction == "ENTER"
    assert events[0].track_id == 1
    assert "track_lost_inside" in events[0].evidence
    assert "head_inside_polygon" in events[0].evidence
    assert "feet_at_threshold" in events[0].evidence
    assert machine.counts()["r1"] == 1

    events = machine.update([track(2, INSIDE)], 3, 1.0)
    assert events[0].direction == "EXIT"
    assert events[0].track_id == 2
    assert "returned_to_door" in events[0].evidence
    assert machine.update([track(2, OUTSIDE)], 4, 1.1) == []
    assert machine.counts()["r1"] == 0


def test_same_track_that_leaves_the_door_while_visible_is_passing_by():
    machine = DoorEventStateMachine([door_quad("r1")], stable_seconds=0)
    machine.update([track(1, OUTSIDE)], 0, 0.0)
    assert machine.update([track(1, INSIDE)], 1, 0.1) == []
    assert machine.update([track(1, OUTSIDE)], 2, 0.2) == []
    assert machine.counts()["r1"] == 0


def test_jitter_across_the_edge_does_not_count():
    machine = DoorEventStateMachine([door_quad("r1")], stable_seconds=0.2)
    machine.update([track(1, OUTSIDE)], 0, 0.0)
    events = []
    for index in range(10):
        points = INSIDE if index % 2 == 0 else OUTSIDE
        events.extend(machine.update([track(1, points)], index + 1, 0.1 * (index + 1)))
    assert events == []
    assert machine.counts()["r1"] == 0


def test_nose_beside_a_narrow_door_does_not_enter():
    narrow = door_quad("r1", x0=0.40, x1=0.43)
    machine = DoorEventStateMachine([narrow], stable_seconds=0, lost_seconds=0.7)
    machine.update([track(1, (200, 600))], 0, 0.0)
    assert machine.update([track(1, (448, 600))], 1, 0.1) == []
    assert machine.update([], 2, 0.9) == []
    assert machine.counts()["r1"] == 0


def test_nose_inside_a_narrow_door_counts_when_the_track_disappears():
    narrow = door_quad("r1", x0=0.40, x1=0.43)
    machine = DoorEventStateMachine([narrow], stable_seconds=0, lost_seconds=0.7)
    machine.update([track(1, (200, 600))], 0, 0.0)
    assert machine.update([track(1, (415, 600))], 1, 0.1) == []
    events = machine.update([], 2, 0.9)
    assert [event.direction for event in events] == ["ENTER"]


def test_close_person_covering_the_door_does_not_enter():
    machine = DoorEventStateMachine([door_quad("r1")], stable_seconds=0, lost_seconds=0.7)
    machine.update([track(1, OUTSIDE)], 0, 0.0)
    covering = track(1, INSIDE, bbox=np.array([460, 400, 540, 980], dtype=np.float32))
    assert machine.update([covering], 1, 0.1) == []
    assert machine.update([], 2, 0.9) == []
    assert machine.counts()["r1"] == 0


def test_return_counts_exit_when_the_head_is_already_outside():
    machine = DoorEventStateMachine([door_quad("r1")], stable_seconds=0, lost_seconds=0.7)
    machine.update([track(1, OUTSIDE)], 0, 0.0)
    machine.update([track(1, INSIDE)], 1, 0.1)
    assert [event.direction for event in machine.update([], 2, 0.9)] == ["ENTER"]
    coming_out = track(2, (500, 760))
    events = machine.update([coming_out], 3, 1.0)
    assert [event.direction for event in events] == ["EXIT"]
    assert machine.counts()["r1"] == 0


def test_missing_head_keypoints_use_the_top_of_the_body_box():
    machine = DoorEventStateMachine([door_quad("r1")], stable_seconds=0, lost_seconds=0.7)
    machine.update([track(1, OUTSIDE)], 0, 0.0)
    body = Track(1, np.array([460, 520, 540, 750], dtype=np.float32), 0.9, hits=5, keypoints=None)
    assert machine.update([body], 1, 0.1) == []
    events = machine.update([], 2, 0.9)
    assert [event.direction for event in events] == ["ENTER"]


def test_crossing_then_disappearance_counts_enter_once():
    machine = DoorEventStateMachine([door_quad("r1")], stable_seconds=0.2, lost_seconds=0.7)
    machine.update([track(1, OUTSIDE, confidence=0.9)], 0, 0.0)
    assert machine.update([track(1, INSIDE, confidence=0.9)], 1, 0.1) == []
    events = machine.update([], 2, 0.8)
    assert len(events) == 1
    assert events[0].direction == "ENTER"
    assert "track_lost_inside" in events[0].evidence
    assert events[0].confidence < 0.9
    assert machine.counts()["r1"] == 1
    assert machine.update([], 3, 1.6) == []


def test_disappearance_after_enter_does_not_exit():
    machine = DoorEventStateMachine([door_quad("r1")], stable_seconds=0, lost_seconds=0.7)
    machine.update([track(1, OUTSIDE)], 0, 0.0)
    assert machine.update([track(1, INSIDE)], 1, 0.1) == []
    events = machine.update([], 2, 0.9)
    assert [event.direction for event in events] == ["ENTER"]
    assert machine.update([], 3, 2.0) == []
    assert machine.counts()["r1"] == 1


def test_pending_exit_is_cancelled_when_the_track_disappears():
    machine = DoorEventStateMachine([door_quad("r1")], stable_seconds=0.2, lost_seconds=0.7)
    machine.update([track(1, INSIDE)], 0, 0.0)
    assert machine.update([track(1, OUTSIDE)], 1, 0.1) == []
    assert machine.update([], 2, 1.0) == []
    assert machine.counts()["r1"] == 0


def test_disappearance_without_entering_does_not_count():
    machine = DoorEventStateMachine([door_quad("r1")])
    machine.update([track(1, OUTSIDE)], 0, 0.0)
    assert machine.update([], 1, 1.0) == []
    assert machine.counts()["r1"] == 0


def test_stream_end_confirms_pending_enter_only():
    machine = DoorEventStateMachine([door_quad("r1")], stable_seconds=0.2)
    machine.update([track(1, OUTSIDE)], 0, 0.0)
    assert machine.update([track(1, INSIDE)], 1, 0.05) == []
    events = machine.flush(0.05)
    assert [event.direction for event in events] == ["ENTER"]
    assert machine.counts()["r1"] == 1
    assert machine.flush(0.05) == []


def test_count_is_a_plain_int():
    machine = DoorEventStateMachine([door_quad("r1")], stable_seconds=0, lost_seconds=0.7)
    machine.update([track(1, OUTSIDE), track(2, OUTSIDE)], 0, 0.0)
    machine.update([track(1, INSIDE), track(2, INSIDE)], 1, 0.1)
    machine.update([], 2, 0.9)
    assert machine.counts()["r1"] == 2
    assert isinstance(machine.counts()["r1"], int)
    machine.update([track(2, INSIDE)], 3, 1.0)
    machine.update([track(2, OUTSIDE)], 4, 1.1)
    assert machine.counts()["r1"] == 1


def test_early_exit_is_kept_in_the_raw_count():
    machine = DoorEventStateMachine([door_quad("r1", initial=0)], stable_seconds=0, lost_seconds=0.7)
    machine.update([track(1, INSIDE)], 0, 0.0)
    machine.update([track(1, OUTSIDE)], 1, 0.1)
    assert machine.counts()["r1"] == -1
    machine.update([track(1, INSIDE)], 2, 0.2)
    machine.update([], 3, 1.0)
    assert machine.counts()["r1"] == 0


def test_initial_count_is_the_starting_occupancy():
    machine = DoorEventStateMachine([door_quad("r1", initial=2)], stable_seconds=0)
    machine.update([track(1, INSIDE)], 0, 0.0)
    machine.update([track(1, OUTSIDE)], 1, 0.1)
    assert machine.counts()["r1"] == 1


def test_conflicting_initial_count_is_rejected():
    with pytest.raises(ValueError):
        DoorEventStateMachine([
            door_quad("r1", door_id="a", initial=1),
            door_quad("r1", door_id="b", y0=0.1, y1=0.3, initial=2),
        ])


def test_head_inside_one_polygon_does_not_enter_the_other():
    machine = DoorEventStateMachine([
        door_quad("near", door_id="near", x0=0.2, x1=0.45),
        door_quad("far", door_id="far", x0=0.55, x1=0.8),
    ], stable_seconds=0)
    machine.update([track(1, OUTSIDE)], 0, 0.0)
    assert machine.update([track(1, (350, 600))], 1, 0.1) == []
    events = machine.update([], 2, 0.9)
    assert [event.room_id for event in events] == ["near"]
    assert machine.counts() == {"near": 1, "far": 0}


def test_passing_one_door_and_disappearing_in_another_counts_only_the_second():
    machine = DoorEventStateMachine([
        door_quad("hall", door_id="hall", x0=0.2, x1=0.45),
        door_quad("inner", door_id="inner", x0=0.55, x1=0.8),
    ], stable_seconds=0, lost_seconds=0.7)
    machine.update([track(1, OUTSIDE)], 0, 0.0)
    assert machine.update([track(1, (350, 600))], 1, 0.1) == []
    assert machine.update([track(1, (650, 600))], 2, 0.2) == []
    events = machine.update([], 3, 1.0)
    assert [(event.room_id, event.direction) for event in events] == [("inner", "ENTER")]
    assert machine.counts() == {"hall": 0, "inner": 1}


def _with_ankles(person, points):
    for index, point in zip((15, 16), points):
        person.keypoints[index] = [point[0], point[1], 0.9]
    return person


def test_missing_ankles_still_count_from_the_head():
    machine = DoorEventStateMachine([door_quad("r1")], stable_seconds=0, lost_seconds=0.7)
    machine.update([track(1, OUTSIDE)], 0, 0.0)
    machine.update([track(1, INSIDE)], 1, 0.1)
    events = machine.update([], 2, 0.9)
    assert [event.direction for event in events] == ["ENTER"]
    assert "ankles_missing" in events[0].evidence
    assert all(track_id != 1 for track_id, _door_id in machine.track_states)


def test_detected_ankles_must_reach_the_door():
    machine = DoorEventStateMachine([door_quad("r1")], stable_seconds=0, lost_seconds=0.7)
    blocking = _with_ankles(track(1, INSIDE), ((500, 980), (520, 980)))
    machine.update([track(1, OUTSIDE)], 0, 0.0)
    assert machine.update([blocking], 1, 0.1) == []
    assert machine.update([], 2, 0.9) == []
    assert machine.counts()["r1"] == 0

    machine.update([_with_ankles(track(2, OUTSIDE), ((500, 760), (520, 760)))], 3, 1.0)
    machine.update([_with_ankles(track(2, INSIDE), ((500, 760), (520, 760)))], 4, 1.1)
    events = machine.update([], 5, 1.9)
    assert [event.direction for event in events] == ["ENTER"]
    assert "ankles_at_door" in events[0].evidence
    assert all(track_id != 2 for track_id, _door_id in machine.track_states)


def test_head_outside_the_polygon_is_ignored():
    machine = DoorEventStateMachine([door_quad("r1")], stable_seconds=0)
    machine.update([track(1, (50, 900))], 0, 0.0)
    assert machine.update([track(1, (50, 600))], 1, 0.1) == []
    assert machine.counts()["r1"] == 0
