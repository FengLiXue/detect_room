import numpy as np

from src.input.video_source import OfflineVideoSource, advance_sample_phase


def test_sample_phase_keeps_every_other_frame_at_half_rate():
    phase = 30.0
    chosen = []
    for index in range(30):
        emit, phase = advance_sample_phase(phase, 30.0, 15.0)
        if emit:
            chosen.append(index)
    assert chosen == list(range(0, 30, 2))


def test_source_uses_media_timestamps_and_drops_frames_when_behind(tmp_path):
    import cv2

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
    assert writer.isOpened()
    for index in range(5):
        writer.write(np.full((32, 32, 3), index * 30, np.uint8))
    writer.release()

    times = iter([0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5])

    def clock() -> float:
        return next(times)

    frames = list(OfflineVideoSource(path, target_fps=10, realtime=True, clock=clock, sleeper=lambda _seconds: None))
    assert [frame.index for frame in frames] == [0, 4]
    assert frames[1].timestamp > frames[0].timestamp
