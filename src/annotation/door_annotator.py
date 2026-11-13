from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


class DoorAnnotator:
    """在视频首帧上拖动四边形的四个角，使它贴住门洞。"""

    def __init__(self, image: np.ndarray, room_prefix: str = "room", single_door: bool = False, room_name_provider: Callable[[int], str] | None = None, meta_provider: Callable[[int], dict] | None = None):
        self.image = image
        self.height, self.width = image.shape[:2]
        self.room_prefix = room_prefix
        self.single_door = single_door
        self.room_name_provider = room_name_provider
        self.meta_provider = meta_provider
        self.window = "door annotation"
        self.doors: list[dict] = []
        self.corners = self._default_corners()
        self.drag_index: int | None = None
        self.drag_origin: tuple[int, int] | None = None
        self.drag_snapshot: list[tuple[int, int]] | None = None
        self.drag_space = "raw"
        self.message = "拖动四个角或框内空白贴住门洞，Enter 确认这一扇门"
        self.finished = False
        self.cancelled = False

    def _default_corners(self, shift: int = 0) -> list[tuple[int, int]]:
        box_width = max(40, int(self.width * 0.18))
        box_height = max(80, int(self.height * 0.46))
        center_x = self.width // 2 + shift
        center_y = int(self.height * 0.48)
        left = int(np.clip(center_x - box_width / 2, 0, self.width - 1))
        right = int(np.clip(center_x + box_width / 2, 0, self.width - 1))
        top = int(np.clip(center_y - box_height / 2, 0, self.height - 1))
        bottom = int(np.clip(center_y + box_height / 2, 0, self.height - 1))
        return [(left, top), (right, top), (right, bottom), (left, bottom)]

    def _to_image(self, x: int, y: int) -> tuple[int, int]:
        try:
            origin_x, origin_y, window_width, window_height = cv2.getWindowImageRect(self.window)
        except cv2.error:
            origin_x, origin_y, window_width, window_height = 0, 0, self.width, self.height
        if window_width <= 0 or window_height <= 0:
            return int(x), int(y)
        image_x = int(np.clip((x - origin_x) * self.width / window_width, 0, self.width - 1))
        image_y = int(np.clip((y - origin_y) * self.height / window_height, 0, self.height - 1))
        return image_x, image_y

    def _hit_radius(self) -> float:
        return max(28.0, min(self.width, self.height) * 0.05)

    def _spaces(self, x: int, y: int) -> list[tuple[str, tuple[int, int]]]:
        raw = (int(np.clip(x, 0, self.width - 1)), int(np.clip(y, 0, self.height - 1)))
        mapped = self._to_image(x, y)
        spaces = [("raw", raw)]
        if mapped != raw:
            spaces.append(("mapped", mapped))
        return spaces

    def _point(self, x: int, y: int) -> tuple[int, int]:
        for name, point in self._spaces(x, y):
            if name == self.drag_space:
                return point
        return self._spaces(x, y)[0][1]

    def _inside_polygon(self, point: tuple[int, int]) -> bool:
        contour = np.asarray(self.corners, dtype=np.float32).reshape((-1, 1, 2))
        return cv2.pointPolygonTest(contour, point, False) >= 0

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._begin_drag(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and (self.drag_index is not None or self.drag_snapshot is not None):
            self._drag_to(self._point(x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_index = None
            self.drag_origin = None
            self.drag_snapshot = None

    def _begin_drag(self, x: int, y: int) -> None:
        radius = self._hit_radius()
        best: tuple[float, str, int, tuple[int, int]] | None = None
        inside: tuple[str, tuple[int, int]] | None = None
        for name, point in self._spaces(x, y):
            if inside is None and self._inside_polygon(point):
                inside = (name, point)
            for index, corner in enumerate(self.corners):
                distance = float(np.hypot(point[0] - corner[0], point[1] - corner[1]))
                if best is None or distance < best[0]:
                    best = (distance, name, index, point)
        if best is not None and best[0] <= radius:
            self.drag_space = best[1]
            self.drag_index = best[2]
            self.drag_snapshot = None
            self.corners[self.drag_index] = best[3]
            return
        if inside is None:
            return
        self.drag_space = inside[0]
        self.drag_index = None
        self.drag_origin = inside[1]
        self.drag_snapshot = list(self.corners)

    def _drag_to(self, point: tuple[int, int]) -> None:
        if self.drag_index is not None:
            self.corners[self.drag_index] = point
            return
        if self.drag_origin is None or self.drag_snapshot is None:
            return
        dx = point[0] - self.drag_origin[0]
        dy = point[1] - self.drag_origin[1]
        self.corners = [
            (int(np.clip(x + dx, 0, self.width - 1)), int(np.clip(y + dy, 0, self.height - 1)))
            for x, y in self.drag_snapshot
        ]

    def _polygon_area(self) -> float:
        points = np.asarray(self.corners, dtype=np.float32)
        xs, ys = points[:, 0], points[:, 1]
        return float(0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))

    def _normalize(self) -> list[list[float]]:
        return [[round(x / self.width, 6), round(y / self.height, 6)] for x, y in self.corners]

    def _complete_door(self) -> None:
        if self._polygon_area() < 16:
            self.message = "四边形太小，请拉开四个角后再按 Enter"
            return
        index = len(self.doors) + 1
        default_name = f"{self.room_prefix}_{index:03d}"
        if self.meta_provider is not None:
            meta = self.meta_provider(index) or {}
            name = str(meta.get("room_name") or default_name).strip()
            initial_count = int(meta.get("initial_count", 0))
        else:
            name = self.room_name_provider(index) if self.room_name_provider else default_name
            name = (name or default_name).strip()
            initial_count = 0
        self.doors.append({
            "door_id": f"door_{index:03d}",
            "room_id": name,
            "room_name": name,
            "polygon": self._normalize(),
            "initial_count": initial_count,
        })
        if self.single_door:
            self.finished = True
            self.message = f"已保存 {name}，按任意键退出"
            return
        self.corners = self._default_corners(shift=40 * index)
        self.message = f"已保存 {name}。拖动下一扇门，Enter 确认，S 保存并退出"

    def _draw(self) -> np.ndarray:
        canvas = self.image.copy()
        for door in self.doors:
            pixels = np.asarray([[int(x * self.width), int(y * self.height)] for x, y in door["polygon"]], dtype=np.int32)
            cv2.polylines(canvas, [pixels.reshape((-1, 1, 2))], True, (0, 180, 255), 2, cv2.LINE_AA)
            center = pixels.mean(axis=0).astype(int)
            cv2.putText(canvas, door["room_name"], tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        active = np.asarray(self.corners, dtype=np.int32)
        cv2.polylines(canvas, [active.reshape((-1, 1, 2))], True, (0, 0, 255), 2, cv2.LINE_AA)
        for index, point in enumerate(self.corners):
            cv2.circle(canvas, point, 12, (0, 255, 255), -1)
            cv2.circle(canvas, point, 12, (0, 0, 0), 1)
            cv2.putText(canvas, str(index + 1), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, 64), (0, 0, 0), -1)
        canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)
        cv2.putText(canvas, self.message, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(canvas, "拖四角或框内  Enter确认  R重置  S保存  Q退出", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 180), 2, cv2.LINE_AA)
        return canvas

    def run(self) -> list[dict]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, min(self.width, 1280), min(self.height, 720))
        cv2.setMouseCallback(self.window, self._on_mouse)
        while not self.finished and not self.cancelled:
            cv2.imshow(self.window, self._draw())
            key = cv2.waitKey(30) & 0xFF
            if self.finished:
                break
            if key in (ord("q"), ord("Q"), 27):
                self.cancelled = True
            elif key in (13, 10):
                self._complete_door()
            elif key in (ord("r"), ord("R")):
                self.corners = self._default_corners(shift=40 * len(self.doors))
                self.message = "已重置当前四边形"
            elif key in (ord("s"), ord("S")) and self.doors:
                self.finished = True
        cv2.destroyWindow(self.window)
        return [] if self.cancelled else self.doors


def _prompt_int(label: str, default: int = 0) -> int:
    try:
        raw = input(label)
    except EOFError:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"无法解析为整数，已使用 {default}")
        return default
    if value < 0:
        print(f"不能为负数，已使用 {default}")
        return default
    return value


def annotate_video(video_path: str | Path, config_path: str | Path, room_prefix: str = "room", single_door: bool = False) -> list[dict]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError("视频没有可读取的首帧")

    def meta_provider(index: int) -> dict:
        default_name = f"{room_prefix}_{index:03d}"
        try:
            value = input(f"第 {index} 个门的房间名称（与已有门相同则计入同一房间，直接回车使用 {default_name}）：")
        except EOFError:
            value = ""
        return {
            "room_name": value.strip() or default_name,
            "initial_count": _prompt_int("初始人数（整数，直接回车为 0）："),
        }

    annotator = DoorAnnotator(frame, room_prefix=room_prefix, single_door=single_door, meta_provider=meta_provider)
    doors = annotator.run()
    if not doors:
        return []
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"doors": doors}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已保存 {len(doors)} 个门口配置：{path}")
    return doors
