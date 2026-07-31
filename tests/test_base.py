from detector.base import CategoryAdapter, DetectionResult, GradeResult


def test_detection_result_defaults():
    r = DetectionResult(defects=[], defect_ratio=0.0)
    assert r.defect_count == 0
    assert r.severity == "none"


def test_grade_result():
    g = GradeResult(grade="good", label="优等", color=(0, 200, 0))
    assert g.grade == "good"
    assert g.label == "优等"
