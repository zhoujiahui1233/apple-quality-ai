from detector.base import CategoryAdapter, DetectionResult, DefectRegion, GradeResult
import cv2
import numpy as np
import config
from apple_defect_detector import (
    detect_defects, classify_quality, visualize_result,
    DEFECT_HSV_RANGES
)


class AppleAdapter(CategoryAdapter):
    category_id = "apple"
    display_name = "苹果"

    def detect_surface(self, img_bgr, mask, contour):
        defect_mask, defect_contours = detect_defects(img_bgr, mask)
        defects = []
        for c in defect_contours:
            x, y, w, h = cv2.boundingRect(c)
            kind = self._classify_defect_kind(img_bgr, c)
            defects.append(DefectRegion(
                contour=c, bbox=(x, y, w, h), kind=kind,
                area=int(cv2.contourArea(c)), severity="medium"
            ))
        apple_pixels = int(np.sum(mask > 0))
        defect_pixels = int(np.sum(defect_mask > 0))
        ratio = defect_pixels / apple_pixels if apple_pixels else 0.0
        return DetectionResult(defects=defects, defect_ratio=ratio,
                               apple_mask=mask, defect_mask=defect_mask)

    def classify_grade(self, result):
        apple_pixels = int(np.sum(result.apple_mask > 0)) if result.apple_mask is not None else 0
        defect_pixels = int(np.sum(result.defect_mask > 0)) if result.defect_mask is not None else 0
        grade, ratio, color = classify_quality(
            defect_pixels, apple_pixels,
            premium_th=config.PREMIUM_THRESHOLD,
            standard_th=config.STANDARD_THRESHOLD
        )
        label_map = {"Premium": "优等", "Standard": "合格",
                     "Reject": "不合格", "Unknown": "未知"}
        return GradeResult(
            grade=grade.lower(), label=label_map.get(grade, grade),
            color=tuple(color), confidence=1.0
        )

    def visualize(self, img_bgr, result, grade):
        contour = None
        if result.apple_mask is not None:
            co, _ = cv2.findContours(result.apple_mask.astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if co:
                contour = max(co, key=cv2.contourArea)
        return visualize_result(
            img_bgr, contour, [d.contour for d in result.defects],
            result.defect_mask if result.defect_mask is not None else
            np.zeros(img_bgr.shape[:2], np.uint8),
            grade.label, result.defect_ratio, grade.color
        )

    def _classify_defect_kind(self, img_bgr, contour) -> str:
        """根据颜色判断缺陷类型: scar/rot/stain"""
        x, y, w, h = cv2.boundingRect(contour)
        region = img_bgr[y:y+h, x:x+w]
        if region.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mean_h = float(np.mean(hsv[:, :, 0]))
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_v = float(np.mean(hsv[:, :, 2]))
        if mean_v < 50:
            return "rot"
        if 5 <= mean_h <= 40 and mean_s > 50:
            return "scar"
        if mean_s < 50 and mean_v > 60:
            return "stain"
        return "unknown"
