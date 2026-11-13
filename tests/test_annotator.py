import cv2
import numpy as np

from src.annotation.door_annotator import DoorAnnotator


def _annotator():
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    annotator = DoorAnnotator(image, meta_provider=lambda _index: {"room_name": "r", "initial_count": 0})
    annotator.corners = [(10, 10), (80, 10), (80, 70), (10, 70)]
    return annotator


def test_corner_follows_the_mouse_without_a_button_flag():
    annotator = _annotator()
    annotator._on_mouse(cv2.EVENT_LBUTTONDOWN, 10, 10, 0, None)
    annotator._on_mouse(cv2.EVENT_MOUSEMOVE, 25, 30, 0, None)
    assert annotator.corners[0] == (25, 30)


def test_dragging_inside_the_quad_moves_every_corner():
    annotator = _annotator()
    annotator._on_mouse(cv2.EVENT_LBUTTONDOWN, 40, 40, 0, None)
    annotator._on_mouse(cv2.EVENT_MOUSEMOVE, 50, 55, 0, None)
    assert annotator.corners == [(20, 25), (90, 25), (90, 85), (20, 85)]
