import cv2
import numpy as np
from detector.registry import CategoryRegistry
from apple_defect_detector import segment_apple


_registry = CategoryRegistry()


def run_detection(img_bgr: np.ndarray, category: str) -> dict:
    """统一检测管线: 预处理→分割→品类检测→分级→可视化"""
    adapter = _registry.get(category)
    if adapter is None:
        raise ValueError(f"Unknown category: {category}")

    # 预处理 + 定位
    if category == "apple":
        mask, contour = segment_apple(img_bgr)
        if mask is None:
            return {"category": category, "error": "未检测到食品区域",
                    "grade": None, "grade_label": "未检出", "defect_ratio": 0.0}
    else:
        mask = np.ones(img_bgr.shape[:2], np.uint8) * 255
        contour = None

    result = adapter.detect_surface(img_bgr, mask, contour)
    grade = adapter.classify_grade(result)
    visualized = adapter.visualize(img_bgr, result, grade)

    return {
        "category": category,
        "grade": grade.grade,
        "grade_label": grade.label,
        "grade_color": grade.color,
        "defect_ratio": round(result.defect_ratio, 4),
        "defect_count": result.defect_count,
        "defects": [d.kind for d in result.defects],
        "visualized": visualized,
    }
