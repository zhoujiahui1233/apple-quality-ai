from detector.base import CategoryAdapter, DetectionResult, DefectRegion, GradeResult
import cv2
import numpy as np
import config


class BeverageAdapter(CategoryAdapter):
    category_id = "beverage"
    display_name = "饮料"

    def detect_surface(self, img_bgr, mask, contour):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        defects = []

        # 液位检测: 每列找颜色梯度突变最大的行
        h, w = img_bgr.shape[:2]
        column_levels = []
        for x in range(int(w * 0.3), int(w * 0.7)):
            col = gray[:, x]
            diff = np.abs(np.diff(col.astype(np.int16)))
            if diff.max() > 20:
                column_levels.append(int(np.argmax(diff)))
        if column_levels:
            level_ratio = float(np.mean(column_levels)) / h
            if level_ratio < 0.35 or level_ratio > 0.75:
                defects.append(DefectRegion(
                    None, (0, int(level_ratio * h), w, 4), "level",
                    int(w * 4), "medium"))

        # 污渍/划痕: 瓶身区域高梯度异常块
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.bitwise_and(edges, mask)
        for c in self._contours_of(edges):
            if cv2.contourArea(c) < 30:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            defects.append(DefectRegion(c, (x, y, ww, hh), "stain",
                                        int(cv2.contourArea(c)), "low"))

        total_pixels = int(np.sum(mask > 0)) or 1
        defect_pixels = sum(d.area for d in defects)
        return DetectionResult(defects=defects,
                               defect_ratio=defect_pixels / total_pixels,
                               apple_mask=mask)

    def classify_grade(self, result):
        has_level = any(d.kind == "level" for d in result.defects)
        if has_level or result.defect_ratio >= config.STANDARD_THRESHOLD:
            return GradeResult("reject", "不合格", (0, 0, 255))
        if result.defect_ratio >= config.PREMIUM_THRESHOLD:
            return GradeResult("standard", "合格", (0, 255, 255))
        return GradeResult("premium", "优等", (0, 200, 0))

    def visualize(self, img_bgr, result, grade):
        out = img_bgr.copy()
        for d in result.defects:
            if d.bbox:
                x, y, w, h = d.bbox
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(out, d.kind, (x, max(0, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return out

    def _contours_of(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        return list(cnts)
