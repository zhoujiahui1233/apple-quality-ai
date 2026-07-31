import numpy as np
from detector.pipeline import run_detection


def test_run_detection_apple():
    img = np.full((200, 200, 3), (40, 40, 200), dtype=np.uint8)  # 红色苹果
    result = run_detection(img, "apple")
    assert result["category"] == "apple"
    assert "grade" in result
    assert "grade_label" in result
    assert "defect_ratio" in result
    assert "visualized" in result


def test_run_detection_bread():
    img = np.full((200, 200, 3), (210, 180, 140), dtype=np.uint8)
    result = run_detection(img, "bread")
    assert result["category"] == "bread"
    assert result["grade"] in ("premium", "standard", "reject")


def test_run_detection_unknown():
    img = np.full((100, 100, 3), 0, dtype=np.uint8)
    try:
        run_detection(img, "unknown")
        assert False
    except ValueError:
        pass
