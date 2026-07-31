from detector.base import CategoryAdapter, DetectionResult, DefectRegion, GradeResult
import cv2
import numpy as np
import config


class BreadAdapter(CategoryAdapter):
    category_id = "bread"
    display_name = "面包"

    MOLD_RANGES = [
        (np.array([35, 40, 60]), np.array([90, 255, 200])),   # 绿霉
        (np.array([0, 0, 180]), np.array([180, 30, 255])),    # 白霉（高亮偏白）
        (np.array([0, 0, 100]), np.array([180, 40, 180])),    # 灰霉
    ]
    BURN_RANGES = [
        (np.array([0, 60, 0]), np.array([30, 255, 130])),      # 焦烘深褐
    ]

    def detect_surface(self, img_bgr, mask, contour):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        defects = []

        # 霉斑
        mold_mask = np.zeros(mask.shape, np.uint8)
        for lower, upper in self.MOLD_RANGES:
            mold_mask = cv2.bitwise_or(mold_mask,
                                       cv2.inRange(hsv, lower, upper))
        mold_mask = cv2.bitwise_and(mold_mask, mask)
        mold_mask = self._filter_small(mold_mask)
        for c in self._contours_of(mold_mask):
            x, y, w, h = cv2.boundingRect(c)
            defects.append(DefectRegion(c, (x, y, w, h), "mold",
                                        int(cv2.contourArea(c)), "high"))

        # 焦烘
        burn_mask = np.zeros(mask.shape, np.uint8)
        for lower, upper in self.BURN_RANGES:
            burn_mask = cv2.bitwise_or(burn_mask,
                                       cv2.inRange(hsv, lower, upper))
        burn_mask = cv2.bitwise_and(burn_mask, mask)
        burn_mask = self._filter_small(burn_mask)
        for c in self._contours_of(burn_mask):
            x, y, w, h = cv2.boundingRect(c)
            defects.append(DefectRegion(c, (x, y, w, h), "burn",
                                        int(cv2.contourArea(c)), "medium"))

        total_pixels = int(np.sum(mask > 0)) or 1
        defect_pixels = sum(d.area for d in defects)
        return DetectionResult(defects=defects,
                               defect_ratio=defect_pixels / total_pixels,
                               apple_mask=mask, defect_mask=mold_mask + burn_mask)

    def classify_grade(self, result):
        has_mold = any(d.kind == "mold" for d in result.defects)
        if has_mold or result.defect_ratio >= config.STANDARD_THRESHOLD:
            return GradeResult("reject", "不合格", (0, 0, 255))
        if result.defect_ratio >= config.PREMIUM_THRESHOLD:
            return GradeResult("standard", "合格", (0, 255, 255))
        return GradeResult("premium", "优等", (0, 200, 0))

    def visualize(self, img_bgr, result, grade):
        out = img_bgr.copy()
        for d in result.defects:
            x, y, w, h = d.bbox
            color = (0, 0, 255) if d.kind == "mold" else (0, 165, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, d.kind, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return out

    def _filter_small(self, mask, min_area=80):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out

    def _contours_of(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in cnts if cv2.contourArea(c) >= 80]
