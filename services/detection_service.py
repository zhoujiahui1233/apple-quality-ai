import uuid
from pathlib import Path
import cv2
import os
import config
from detector.pipeline import run_detection
from models import db, DetectionRecord
from flask import url_for
import numpy as np


class DetectionService:
    def __init__(self):
        pass

    def _save_upload(self, file_storage):
        ext = Path(file_storage.filename).suffix.lower() or ".jpg"
        unique_name = str(uuid.uuid4()) + ext
        upload_path = config.UPLOAD_DIR / unique_name
        file_storage.save(str(upload_path))
        return unique_name, str(upload_path)

    def _auto_resize(self, img, upload_path):
        h, w = img.shape[:2]
        if max(w, h) > 1600:
            scale = 1600.0 / max(w, h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
            cv2.imwrite(upload_path, img)
        return img

    def process_upload(self, file_storage, category: str, tuning=None) -> dict:
        tuning = tuning or {}
        if category == "apple":
            return self._process_apple(file_storage, tuning)
        return self._process_generic(file_storage, category)

    def _process_generic(self, file_storage, category: str) -> dict:
        """面包/饮料: 走统一管线"""
        unique_name, upload_path = self._save_upload(file_storage)
        img = cv2.imread(upload_path)
        if img is None:
            return {"error": "无法解析图片文件"}
        img = self._auto_resize(img, upload_path)

        pipe = run_detection(img, category)
        if "error" in pipe:
            return pipe

        result_filename = f"result_{unique_name}"
        cv2.imwrite(str(config.UPLOAD_DIR / result_filename), pipe["visualized"])

        record = DetectionRecord(
            filename=unique_name,
            result_filename=result_filename,
            grade=pipe["grade_label"],
            grade_color=str(pipe["grade_color"]),
            defect_ratio=pipe["defect_ratio"],
            defect_count=pipe["defect_count"],
        )
        db.session.add(record)
        db.session.commit()

        return {
            "id": record.id,
            "category": category,
            "grade": pipe["grade_label"],
            "grade_en": pipe["grade"],
            "defect_ratio": f"{pipe['defect_ratio'] * 100:.2f}%",
            "defect_count": pipe["defect_count"],
            "defects": pipe["defects"],
            "original_image": url_for("static", filename=f"uploads/{unique_name}"),
            "result_image": url_for("static", filename=f"uploads/{result_filename}"),
        }

    def _process_apple(self, file_storage, tuning=None) -> dict:
        """苹果: 四模型对比全功能（原有逻辑）"""
        from apple_defect_detector import (
            preprocess, segment_apple, detect_defects, classify_quality,
            predict_with_svm, predict_with_rf, predict_with_dl, predict_with_yolo,
            visualize_result, visualize_svm, visualize_rf, visualize_dl,
            visualize_yolo, predict_with_yolo_detect, visualize_yolo_detect,
        )

        unique_name, upload_path = self._save_upload(file_storage)
        img = cv2.imread(upload_path)
        if img is None:
            return {"error": "无法解析图片文件"}
        img = self._auto_resize(img, upload_path)

        img_p, _ = preprocess(img)
        apple_mask, apple_contour = segment_apple(img_p)
        if apple_mask is None:
            return {"error": "未检测到苹果区域，请换一张图片"}

        apple_pixels = int(np.sum(apple_mask > 0))
        sensitivity = int(tuning.get("sensitivity", 5)) if tuning else 5
        min_area = int(tuning.get("min_area", 200)) if tuning else 200
        premium_th = float(tuning.get("premium_th", config.PREMIUM_THRESHOLD)) if tuning else config.PREMIUM_THRESHOLD
        standard_th = float(tuning.get("standard_th", config.STANDARD_THRESHOLD)) if tuning else config.STANDARD_THRESHOLD

        defect_mask, defect_contours = detect_defects(img_p, apple_mask, sensitivity=sensitivity, min_area=min_area)
        defect_pixels = int(np.sum(defect_mask > 0))
        defect_count = len(defect_contours)
        grade, defect_ratio, grade_color = classify_quality(
            defect_pixels, apple_pixels,
            premium_th=premium_th,
            standard_th=standard_th
        )

        yolo_grade, yolo_color, yolo_conf, yolo_detail = predict_with_yolo(img_p)
        if yolo_grade is not None and yolo_conf > 0.5:
            grade = yolo_grade
            grade_color = yolo_color

        svm_grade, svm_color, svm_detail = predict_with_svm(img_p, apple_mask, apple_contour)
        rf_grade, rf_color, rf_detail = predict_with_rf(img_p, apple_mask, apple_contour)
        dl_grade, dl_color, dl_conf, dl_detail = predict_with_dl(img_p)

        result_img = visualize_result(img_p, apple_contour, defect_contours,
                                      defect_mask, grade, defect_ratio, grade_color,
                                      yolo_conf=yolo_conf)
        result_filename = f"result_{unique_name}"
        cv2.imwrite(str(config.UPLOAD_DIR / result_filename), result_img)

        svm_viz = visualize_svm(img_p, str(svm_grade) if svm_grade else "N/A",
                                svm_color if svm_color else (0, 200, 0), defect_ratio=defect_ratio)
        rf_viz = visualize_rf(img_p, str(rf_grade) if rf_grade else "N/A",
                              rf_color if rf_color else (0, 200, 0), defect_ratio=defect_ratio)
        dl_viz = visualize_dl(img_p, str(dl_grade) if dl_grade else "N/A",
                              dl_conf if dl_conf else 0.0,
                              dl_color if dl_color else (0, 200, 0), defect_ratio=defect_ratio)
        yolo_viz = visualize_yolo(img_p, str(yolo_grade) if yolo_grade else "N/A",
                                  yolo_conf if yolo_conf else 0.0,
                                  yolo_color if yolo_color else (0, 200, 0), defect_ratio=defect_ratio)

        viz_names = {}
        for key, viz in [("svm", svm_viz), ("rf", rf_viz), ("dl", dl_viz), ("yolo", yolo_viz)]:
            viz_names[key] = f"{key}_{unique_name}"
            cv2.imwrite(str(config.UPLOAD_DIR / viz_names[key]), viz)

        detect_results = predict_with_yolo_detect(img_p)
        detect_viz = visualize_yolo_detect(img_p, detect_results)
        detect_viz_name = "detect_" + unique_name
        cv2.imwrite(str(config.UPLOAD_DIR / detect_viz_name), detect_viz)

        record = DetectionRecord(
            filename=unique_name, result_filename=result_filename,
            grade=grade, grade_color=str(grade_color),
            defect_ratio=float(defect_ratio), defect_count=defect_count,
            apple_area=apple_pixels, defect_area=defect_pixels,
        )
        db.session.add(record)
        db.session.commit()

        return {
            "id": record.id,
            "category": "apple",
            "grade": grade,
            "svm_detail": svm_detail if isinstance(svm_detail, str) else "",
            "rf_detail": rf_detail if isinstance(rf_detail, str) else "",
            "dl_detail": dl_detail if isinstance(dl_detail, str) else "",
            "yolo_detail": yolo_detail if isinstance(yolo_detail, str) else "",
            "svm_image": url_for("static", filename="uploads/" + viz_names["svm"]),
            "rf_image": url_for("static", filename="uploads/" + viz_names["rf"]),
            "dl_image": url_for("static", filename="uploads/" + viz_names["dl"]),
            "yolo_image": url_for("static", filename="uploads/" + viz_names["yolo"]),
            "detect_image": url_for("static", filename="uploads/" + detect_viz_name),
            "detect_count": len(detect_results) if detect_results else 0,
            "svm_grade": svm_grade, "rf_grade": rf_grade,
            "dl_grade": dl_grade,
            "dl_conf": f"{dl_conf * 100:.1f}%",
            "yolo_grade": yolo_grade,
            "yolo_conf": f"{yolo_conf * 100:.1f}%",
            "defect_ratio": f"{defect_ratio * 100:.2f}%",
            "defect_count": defect_count,
            "apple_area": apple_pixels,
            "defect_area": defect_pixels,
            "original_image": url_for("static", filename=f"uploads/{unique_name}"),
            "result_image": url_for("static", filename=f"uploads/{result_filename}"),
        }
