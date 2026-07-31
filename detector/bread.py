from detector.base import CategoryAdapter, DetectionResult, GradeResult
import numpy as np


class BreadAdapter(CategoryAdapter):
    category_id = "bread"
    display_name = "苹果"

    def detect_surface(self, img_bgr, mask, contour):
        raise NotImplementedError("Phase 2")

    def classify_grade(self, result):
        raise NotImplementedError("Phase 2")

    def visualize(self, img_bgr, result, grade):
        raise NotImplementedError("Phase 2")
