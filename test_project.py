import project
import numpy as np
import pyautogui

from project import detect_click, detect_scroll, move_cursor

def test_move_cursor(monkeypatch):
    landmarks = []

    for i in range(21):
        landmark = type("Landmark", (), {})()
        landmark.x = 0.5
        landmark.y = 0.5
        landmarks.append(landmark)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    calls = []

    monkeypatch.setattr(
        pyautogui,
        "moveTo",
        lambda x, y: calls.append((x, y))
    )

    project.move_cursor(
        landmarks,
        1920,
        1080,
        640,
        480,
        frame
    )

    assert len(calls) == 1

    # Allow tiny floating-point differences.
    assert np.isclose(calls[0][0], 960)
    assert np.isclose(calls[0][1], 540)


def test_detect_click(monkeypatch):
    landmarks = []

    for i in range(21):
        landmark = type("Landmark", (), {})()
        landmark.x = 0.5
        landmark.y = 0.5
        landmarks.append(landmark)
    landmarks[8].x = 0.50
    landmarks[8].y = 0.50
    landmarks[4].x = 0.51
    landmarks[4].y = 0.51
    calls = []
    monkeypatch.setattr(
        pyautogui, "click", lambda: calls.append(True))
    assert project.detect_click(landmarks, False) is True
    assert calls == [True]
    assert project.detect_click(landmarks, True) is True

def test_detect_scroll():
    landmarks = []


    for i in range(21):
        landmark = type("Landmark", (), {})()
        landmark.x = 0.5
        landmark.y = 0.5
        landmarks.append(landmark)


    landmarks[8].y = 0.3
    landmarks[6].y = 0.5

    landmarks[12].y = 0.3
    landmarks[10].y = 0.5





    result = project.detect_scroll(landmarks, None)


    assert result == 0.3


