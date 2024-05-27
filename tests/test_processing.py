# File: tests/test_preprocessing.py
import numpy as np
from breakout_rl_dqn.pre_processing import process_input_frames, rgb_to_luminance

def test_rgb_to_luminance():
    frame = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [0, 0, 0]]], dtype=np.uint8)
    luminance = rgb_to_luminance(frame)
    print(luminance)
    assert np.allclose(luminance, np.array([[76, 149], [29, 0]]), atol=1)


def test_process_input_frames():
    frames = [np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8) for _ in range(4)]
    processed = process_input_frames(frames)
    assert processed.shape == (84, 84)
