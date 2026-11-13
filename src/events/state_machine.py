from __future__ import annotations

from dataclasses import dataclass

from src.common.types import DoorEvent, Track
from src.geometry.door import DoorGeometry


@dataclass
class TrackDoorState:
    phase: str | None = None  # outside, near, in_door
    saw_outside: bool = False
    started_inside: bool = False
    ankles_seen: bool = False
    pending_direction: str | None = None
    pending_since: float | None = None
    hold_outside: bool = False
    last_seen_time: float = -1.0
    last_frame: int = -1
    confidence: float = 0.0


@dataclass
class OpenEnter:
    track_id: int
    released: bool = False


@dataclass
class RoomState:
    count: int = 0


def _ankles_ready(door: DoorGeometry, track: Track) -> tuple[bool, bool]:
    """没检到脚踝时只看头。检到了，脚踝必须进入门洞或贴着下沿。"""
    points = track.ankle_points()
    if not points:
        return True, False
    return all(door.part_at_door(point) for point in points), True


def _phase(door: DoorGeometry, track: Track) -> tuple[str, bool]:
    """门外、靠近、头和脚踝都进入门洞。返回阶段，以及这一帧有没有检到脚踝。"""
    head_inside = door.contains(track.head_point())
    near = door.feet_near_threshold(track.bbox)
    ankles_ready, ankles_seen = _ankles_ready(door, track)
    if head_inside and near and ankles_ready:
        return "in_door", ankles_seen
    if near or head_inside:
        return "near", ankles_seen
    return "outside", ankles_seen


def _at_door(door: DoorGeometry, track: Track) -> bool:
    """人回到门口：脚仍在下沿附近，头或脚踝有一个碰到这扇门。"""
    if not door.feet_near_threshold(track.bbox):
        return False
    if door.contains(track.head_point()) or door.overlaps_bbox(track.head_box()):
        return True
    return any(door.part_at_door(point) for point in track.ankle_points())


class DoorEventStateMachine:
    """每个 track 对每扇门走一遍：门外，靠近，头和脚踝进入，轨迹消失后记进入。

    进入一旦发出，这条 track 的状态清掉。脚踝没检到时只看头；检到了就必须也到门口。
    脚明显低于门洞下沿时，人在镜头和门之间，挡住门洞也不计数。
    """

    def __init__(self, doors: list[DoorGeometry], fps: float = 15.0, stable_seconds: float = 0.2, lost_seconds: float = 0.7, state_ttl_seconds: float = 2.0, stable_frames: int | None = None):
        if stable_frames is not None:
            stable_seconds = max(0.0, (stable_frames - 1) / max(float(fps), 1e-6))
        if stable_seconds < 0 or lost_seconds < 0 or state_ttl_seconds < 0:
            raise ValueError("稳定时间、丢失时间和状态保留时间不能为负数")
        self.doors = doors
        self.fps = fps
        self.stable_seconds = stable_seconds
        self.lost_seconds = lost_seconds
        self.state_ttl_seconds = state_ttl_seconds
        self.track_states: dict[tuple[int, str], TrackDoorState] = {}
        self.open_enters: dict[str, list[OpenEnter]] = {}
        self._doors_by_id: dict[str, DoorGeometry] = {}
        self.rooms: dict[str, RoomState] = {}
        for door in doors:
            if door.door_id in self._doors_by_id:
                raise ValueError(f"重复的 door_id：{door.door_id}")
            self._doors_by_id[door.door_id] = door
            room = self.rooms.get(door.room_id)
            if room is None:
                self.rooms[door.room_id] = RoomState(count=int(door.initial_count))
                continue
            if room.count != door.initial_count:
                raise ValueError(f"房间 {door.room_id} 的 initial_count 在多个门上不一致")

    def update(self, tracks: list[Track], frame_index: int, timestamp: float) -> list[DoorEvent]:
        events: list[DoorEvent] = []
        active: set[tuple[int, str]] = set()
        for track in tracks:
            for door in self.doors:
                key = (track.track_id, door.door_id)
                active.add(key)
                state = self.track_states.setdefault(key, TrackDoorState())
                sample, ankles_seen = _phase(door, track)
                confirmed = self._observe(state, door, track, sample, ankles_seen, frame_index, timestamp)
                if confirmed is not None:
                    events.append(confirmed)
                state.last_seen_time = timestamp
                state.last_frame = frame_index
                state.confidence = float(track.confidence)
        events.extend(self._retire(timestamp, active))
        events.extend(self._match_returns(tracks, frame_index, timestamp))
        return events

    def flush(self, timestamp: float) -> list[DoorEvent]:
        """视频结束时，已经进入门洞且还没离开的轨迹补记进入，并清掉该 track。"""
        events: list[DoorEvent] = []
        entered: list[int] = []
        for (track_id, door_id), state in list(self.track_states.items()):
            if state.pending_direction != "ENTER" or state.phase != "in_door" or not state.saw_outside:
                continue
            events.append(self._emit(state, self._doors_by_id[door_id], track_id, state.last_frame, timestamp, "lost"))
            entered.append(track_id)
        for track_id in entered:
            self._clear_track(track_id)
        return events

    def counts(self) -> dict[str, int]:
        return {room_id: int(room.count) for room_id, room in self.rooms.items()}

    def _observe(self, state: TrackDoorState, door: DoorGeometry, track: Track, sample: str, ankles_seen: bool, frame_index: int, timestamp: float) -> DoorEvent | None:
        if state.hold_outside:
            if sample == "outside":
                state.hold_outside = False
                state.phase = "outside"
                state.saw_outside = True
                state.pending_direction = None
                state.pending_since = None
            return None
        if state.phase is None:
            state.phase = sample
            state.saw_outside = sample != "in_door"
            state.started_inside = sample == "in_door"
            state.ankles_seen = ankles_seen
            return None
        if sample == "in_door" and state.saw_outside:
            state.phase = "in_door"
            state.ankles_seen = ankles_seen
            state.pending_direction = "ENTER"
            state.pending_since = timestamp
            return None
        if sample == "near" and state.saw_outside:
            state.phase = "near"
            state.pending_direction = None
            state.pending_since = None
            return None
        if state.started_inside and sample != "in_door" and state.pending_direction != "ENTER":
            if state.pending_direction != "EXIT":
                state.pending_direction = "EXIT"
                state.pending_since = timestamp
            if state.pending_since is None or timestamp - state.pending_since + 1e-6 < self.stable_seconds:
                return None
            state.phase = "outside"
            state.saw_outside = True
            return self._emit(state, door, track.track_id, frame_index, timestamp, "stable")
        state.phase = sample
        if sample == "outside":
            state.saw_outside = True
        state.pending_direction = None
        state.pending_since = None
        return None

    def _retire(self, timestamp: float, active: set[tuple[int, str]]) -> list[DoorEvent]:
        events: list[DoorEvent] = []
        kept: dict[tuple[int, str], TrackDoorState] = {}
        entered: list[int] = []
        for key, state in self.track_states.items():
            if key in active:
                kept[key] = state
                continue
            missing = timestamp - state.last_seen_time
            if state.pending_direction == "ENTER" and state.phase == "in_door" and state.saw_outside and missing >= self.lost_seconds:
                track_id, door_id = key
                events.append(self._emit(state, self._doors_by_id[door_id], track_id, state.last_frame, timestamp, "lost"))
                entered.append(track_id)
                continue
            if state.pending_direction == "EXIT":
                state.pending_direction = None
                state.pending_since = None
            if timestamp - state.last_seen_time <= self.state_ttl_seconds:
                kept[key] = state
        self.track_states = kept
        for track_id in entered:
            self._clear_track(track_id)
        return events

    def _clear_track(self, track_id: int) -> None:
        """进入已经记下，这条 track 的状态不再参与后续进出。"""
        self.track_states = {key: state for key, state in self.track_states.items() if key[0] != track_id}

    def _match_returns(self, tracks: list[Track], frame_index: int, timestamp: float) -> list[DoorEvent]:
        events: list[DoorEvent] = []
        active_ids = {track.track_id for track in tracks}
        for door in self.doors:
            opens = self.open_enters.get(door.door_id, [])
            for item in opens:
                if item.track_id not in active_ids:
                    item.released = True
            released = [item for item in opens if item.released]
            if not released:
                continue
            for track in tracks:
                if any(item.track_id == track.track_id and not item.released for item in opens):
                    continue
                if not _at_door(door, track):
                    continue
                opens.remove(released[0])
                events.append(self._emit_return(door, track, frame_index, timestamp))
                break
        return events

    def _emit_return(self, door: DoorGeometry, track: Track, frame_index: int, timestamp: float) -> DoorEvent:
        state = self.track_states.setdefault((track.track_id, door.door_id), TrackDoorState())
        state.pending_direction = "EXIT"
        state.confidence = float(track.confidence)
        state.last_seen_time = timestamp
        state.last_frame = frame_index
        event = self._emit(state, door, track.track_id, frame_index, timestamp, "return")
        phase, _ = _phase(door, track)
        if phase == "in_door":
            state.hold_outside = True
        return event

    def _emit(self, state: TrackDoorState, door: DoorGeometry, track_id: int, frame_index: int, timestamp: float, confirmed_by: str) -> DoorEvent:
        direction = state.pending_direction or ""
        state.phase = "in_door" if direction == "ENTER" else "outside"
        state.pending_direction = None
        state.pending_since = None
        if direction == "ENTER":
            self.open_enters.setdefault(door.door_id, []).append(OpenEnter(track_id))
        else:
            state.saw_outside = True
            state.started_inside = False
        room = self.rooms[door.room_id]
        room.count = int(room.count + (1 if direction == "ENTER" else -1))
        if direction == "ENTER":
            evidence = [
                "head_inside_polygon",
                "feet_at_threshold",
                "ankles_at_door" if state.ankles_seen else "ankles_missing",
                "track_lost_inside" if confirmed_by == "lost" else "stable_inside",
            ]
        elif confirmed_by == "return":
            evidence = ["feet_at_threshold", "returned_to_door"]
        else:
            evidence = ["head_outside_polygon", "feet_at_threshold", "stable_outside"]
        confidence = state.confidence * (0.55 if confirmed_by == "lost" else 1.0)
        return DoorEvent(
            timestamp=timestamp,
            frame_index=frame_index,
            door_id=door.door_id,
            room_id=door.room_id,
            direction=direction,
            track_id=track_id,
            confidence=confidence,
            evidence=evidence,
        )
