import uuid
from pathlib import Path
import cv2
import os
import config
from detector.pipeline import run_detection
from models import db, DetectionRecord
from flask import url_for
import numpy as np


def severity_of(defect_ratio) -> str:
    """按缺陷占比划分严重度：<10% 轻度，10%~30% 中度，>=30% 重度"""
    try:
        ratio = float(defect_ratio)
    except (TypeError, ValueError):
        return "轻度"
    if ratio >= config.SEVERITY_MODERATE_TH:
        return "重度"
    if ratio >= config.SEVERITY_MILD_TH:
        return "中度"
    return "轻度"


def review_reasons(yolo_conf=None, dl_conf=None, grades=(), defect_ratio=None,
                   premium_th=None, standard_th=None) -> list:
    """置信度灰区 / 模型结论不一致 / 缺陷占比临界 → 返回待复核原因"""
    reasons = []
    for name, conf in (("YOLO", yolo_conf), ("深度学习", dl_conf)):
        if conf and config.REVIEW_CONF_LOW <= conf < config.REVIEW_CONF_HIGH:
            reasons.append(f"{name}置信度偏低")
    grade_set = {g for g in grades if g}
    if len(grade_set) > 1:
        reasons.append("模型结论不一致")
    if defect_ratio is not None:
        try:
            ratio = float(defect_ratio)
        except (TypeError, ValueError):
            ratio = None
        if ratio is not None:
            if premium_th is not None and abs(ratio - float(premium_th)) < config.REVIEW_RATIO_MARGIN:
                reasons.append("缺陷占比接近优质/合格阈值")
            if standard_th is not None and abs(ratio - float(standard_th)) < config.REVIEW_RATIO_MARGIN:
                reasons.append("缺陷占比接近合格/不合格阈值")
    return reasons


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

    @staticmethod
    def _remove_upload(upload_path):
        """处理失败时清理已上传的图片，避免孤儿文件"""
        if upload_path:
            try:
                if os.path.exists(upload_path):
                    os.remove(upload_path)
            except Exception:
                pass

    def _queue_for_review(self, unique_name):
        """将待复核检测的原始图片复制进复核队列（dataset_collected/unlabeled）"""
        import shutil
        src = config.UPLOAD_DIR / unique_name
        dst_dir = config.ROOT_DIR / "dataset_collected" / "unlabeled"
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"review_{uuid.uuid4().hex[:8]}_{unique_name}"
            shutil.copy2(str(src), str(dst))
        except Exception:
            pass

    @staticmethod
    def _parse_tuning(tuning):
        """容错解析参数：无效值回落默认"""
        tuning = tuning or {}

        def _int(v, d):
            try:
                return int(float(v))
            except Exception:
                return d

        def _float(v, d):
            try:
                return float(v)
            except Exception:
                return d

        return {
            "sensitivity": _int(tuning.get("sensitivity"), config.DEFECT_SENSITIVITY),
            "min_area": _int(tuning.get("min_area"), config.DEFECT_MIN_AREA),
            "premium_th": _float(tuning.get("premium_th"), config.PREMIUM_THRESHOLD),
            "standard_th": _float(tuning.get("standard_th"), config.STANDARD_THRESHOLD),
            "white_balance": 1 if str(tuning.get("white_balance", "")).strip() in ("1", "true", "on") else 0,
        }

    def process_upload(self, file_storage, category: str, tuning=None, auto_review=True) -> dict:
        tuning = tuning or {}
        if category == "auto":
            return self._process_auto(file_storage, tuning, auto_review)
        unique_name, upload_path = self._save_upload(file_storage)
        if category == "apple":
            return self._process_apple(upload_path, unique_name, tuning, auto_review)
        return self._process_generic(upload_path, unique_name, category, auto_review)

    def _process_auto(self, file_storage, tuning=None, auto_review=True) -> dict:
        """自动识别品类后再检测"""
        from detector.autodetect import auto_detect_category
        unique_name, upload_path = self._save_upload(file_storage)
        img = cv2.imread(upload_path)
        if img is None:
            self._remove_upload(upload_path)
            return {"error": "无法解析图片文件"}
        category, confidence = auto_detect_category(img)
        if category == "apple":
            result = self._process_apple(upload_path, unique_name, tuning, auto_review)
        else:
            result = self._process_generic(upload_path, unique_name, category, auto_review)
        if "error" not in result:
            result["category_detected"] = category
            result["auto_confidence"] = confidence
        return result

    def _process_generic(self, upload_path: str, unique_name: str, category: str, auto_review=True) -> dict:
        """面包/饮料: 走统一管线"""
        img = cv2.imread(upload_path)
        if img is None:
            self._remove_upload(upload_path)
            return {"error": "无法解析图片文件"}
        img = self._auto_resize(img, upload_path)

        try:
            pipe = run_detection(img, category)
        except ValueError:
            self._remove_upload(upload_path)
            return {"error": "暂不支持该品类"}
        if "error" in pipe:
            self._remove_upload(upload_path)
            return pipe

        severity = severity_of(pipe["defect_ratio"])
        reasons = review_reasons(defect_ratio=pipe["defect_ratio"],
                                 premium_th=config.PREMIUM_THRESHOLD,
                                 standard_th=config.STANDARD_THRESHOLD)
        needs_review = bool(reasons)
        review_reason = "；".join(reasons)
        result_filename = f"result_{unique_name}"
        cv2.imwrite(str(config.UPLOAD_DIR / result_filename), pipe["visualized"])

        record = DetectionRecord(
            filename=unique_name,
            result_filename=result_filename,
            category=category,
            grade=pipe["grade_label"],
            grade_color=str(pipe["grade_color"]),
            defect_ratio=pipe["defect_ratio"],
            defect_count=pipe["defect_count"],
            severity=severity, needs_review=needs_review, review_reason=review_reason,
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
            "severity": severity,
            "needs_review": needs_review,
            "review_reason": review_reason,
            "original_image": url_for("static", filename=f"uploads/{unique_name}"),
            "result_image": url_for("static", filename=f"uploads/{result_filename}"),
        }

    def _apple_detect_core(self, upload_path: str, unique_name: str, tuning=None) -> dict:
        """苹果单图检测核心：只检测+可视化，不写数据库（多角度检测复用）"""
        from apple_defect_detector import (
            preprocess, segment_apple, classify_quality,
            predict_with_svm, predict_with_rf, predict_with_dl, predict_with_yolo,
            visualize_result, visualize_svm, visualize_rf, visualize_dl,
            visualize_yolo, predict_with_yolo_detect, visualize_yolo_detect,
            analyze_defect_blocks, predict_with_yolo_defect,
            mask_from_yolo_boxes, extract_features_v2,
        )

        img = cv2.imread(upload_path)
        if img is None:
            return {"error": "无法解析图片文件"}
        img = self._auto_resize(img, upload_path)

        tuned = self._parse_tuning(tuning)
        if tuned["white_balance"]:
            from apple_defect_detector import auto_white_balance
            img = auto_white_balance(img)

        img_p, _ = preprocess(img)
        # 苹果检测模型单次推理约 180ms，这条流水线原先要跑 3~4 次
        # (分割兜底 / 缺陷 ROI / 末尾可视化)，这里只推理一次再复用，结果一致。
        detect_boxes = predict_with_yolo_detect(img_p, conf_th=0.15)
        apple_mask, apple_contour = segment_apple(img_p)
        if apple_mask is None:
            # fall back to YOLO apple box when color segmentation fails
            apple_mask, apple_contour = mask_from_yolo_boxes(img_p, detect_boxes, conf_th=0.5)
            if apple_mask is None:
                return {"error": "未检测到苹果区域，请换一张图片"}
        else:
            # If color segmentation looks incomplete vs the YOLO box, trust
            # the detector box (handles red apples with highlights where
            # HSV segmentation cuts part of the fruit away).
            yolo_m, yolo_c = mask_from_yolo_boxes(img_p, detect_boxes, conf_th=0.5)
            if yolo_c is not None:
                seg_area = cv2.contourArea(apple_contour)
                yolo_area = cv2.contourArea(yolo_c)
                if yolo_area > 0 and seg_area < yolo_area * 0.80:
                    apple_mask, apple_contour = yolo_m, yolo_c

        apple_pixels = int(np.sum(apple_mask > 0))
        sensitivity = tuned["sensitivity"]
        min_area = tuned["min_area"]
        premium_th = tuned["premium_th"]
        standard_th = tuned["standard_th"]

        # ---- AI defect detection is the PRIMARY defect signal. The YOLO
        # defect model was trained on real defect annotations, so it does not
        # report highlights/shadows/healthy skin like the old HSV rules. ----
        ai_defect_boxes = []
        apple_boxes = [d for d in (detect_boxes or [])
                       if float(d.get("confidence", 0.0)) >= 0.5]
        if apple_boxes:
            # largest apple box as the region of interest
            x1, y1, x2, y2 = max(
                (d["bbox"] for d in apple_boxes),
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            ai_defect_boxes = predict_with_yolo_defect(
                img_p, conf_th=0.25, region=(x1, y1, x2, y2))
        else:
            ai_defect_boxes = predict_with_yolo_defect(img_p, conf_th=0.25)
        # 保留有把握的 AI 缺陷。实测(30合格/30不合格)显示：门槛降到 0.40 只多救
        # 回 1 张不合格，却把 2 张干净果误判成 Reject；0.50 时干净果 100% 不误报，
        # 故维持 0.50。真正要提升召回靠补数据重训，而不是放宽门槛。
        ai_confirmed = [d for d in ai_defect_boxes
                        if d["confidence"] >= config.AI_DEFECT_CONF_TH]
        ai_found_defect = len(ai_confirmed) >= 1

        if ai_found_defect:
            # Grade based on AI-detected defect area (inside the apple).
            ai_area_px = 0
            for d in ai_confirmed:
                x1, y1, x2, y2 = d["bbox"]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_p.shape[1], x2); y2 = min(img_p.shape[0], y2)
                ai_area_px += max(0, (x2 - x1) * (y2 - y1))
            defect_pixels = ai_area_px
            defect_count = len(ai_confirmed)
            # Build a mask from AI boxes for visualization.
            ai_mask = np.zeros(img_p.shape[:2], dtype=np.uint8)
            for d in ai_confirmed:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(ai_mask, (x1, y1), (x2, y2), 255, -1)
            defect_mask = cv2.bitwise_and(ai_mask, apple_mask)
            defect_contours, _ = cv2.findContours(
                defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # No confident AI defect: trust the AI model and grade the apple
            # as clean. The old HSV rules produce too many false positives on
            # healthy apples (highlights, shading), so they are NOT used to
            # reject fruit anymore.
            defect_pixels = 0
            defect_count = 0
            defect_contours = []
            defect_mask = np.zeros_like(apple_mask)

        grade, defect_ratio, grade_color = classify_quality(
            defect_pixels, apple_pixels,
            premium_th=premium_th,
            standard_th=standard_th
        )

        yolo_grade, yolo_color, yolo_conf, yolo_detail = predict_with_yolo(img_p)
        # 分类器只在“AI 缺陷证据”一致时参与最终等级：
        #  - AI 检出了缺陷：等级以缺陷占比为准，分类器说“合格”也不能覆盖
        #    （防止 57% 缺陷占比被判 Premium 的 bug）；分类器说“不合格”且缺陷
        #    占比不小(>=0.02)时才采纳为 Reject。
        #  - AI 没检出缺陷：不采纳分类器的“不合格”，避免误伤好果；两者都说好才
        #    维持合格（本来就是合格）。
        yolo_is_bad = ("Reject" in str(yolo_grade) or "不合格" in str(yolo_grade))
        if yolo_grade is not None and yolo_conf > 0.5:
            if yolo_is_bad and ai_found_defect and defect_ratio >= 0.02:
                grade = yolo_grade
                grade_color = yolo_color
            elif not yolo_is_bad and not ai_found_defect:
                grade = yolo_grade
                grade_color = yolo_color

        # 特征只提取一次（内部含一次分割），SVM 与 RF 复用，省约 275ms/图
        ml_feats = extract_features_v2(img_p)
        svm_grade, svm_color, svm_detail = predict_with_svm(
            img_p, apple_mask, apple_contour, feats=ml_feats)
        rf_grade, rf_color, rf_detail = predict_with_rf(
            img_p, apple_mask, apple_contour, feats=ml_feats)
        dl_grade, dl_color, dl_conf, dl_detail = predict_with_dl(img_p)

        # 多模型加权融合 + 先验知识（缺陷比例）修正
        import re as _re
        def _pct(detail, kw):
            if not isinstance(detail, str):
                return 0.5
            m = _re.search(kw + r"[:：]\s*([\d.]+)%", detail)
            return float(m.group(1)) / 100.0 if m else 0.5
        from apple_defect_detector import ensemble_score, estimate_diameter_px
        fusion = ensemble_score(
            svm_grade, rf_grade, dl_grade, yolo_grade,
            _pct(svm_detail, "自信度"), _pct(rf_detail, "优质"),
            _pct(dl_detail, "优质"), yolo_conf,
            defect_ratio, premium_th, standard_th,
        )
        diam = estimate_diameter_px(apple_contour, img_p.shape[1])
        defect_blocks = analyze_defect_blocks(img_p, apple_mask, defect_mask)

        severity = severity_of(defect_ratio)
        reasons = review_reasons(
            yolo_conf=yolo_conf, dl_conf=dl_conf,
            grades=(svm_grade, rf_grade, dl_grade, yolo_grade),
            defect_ratio=defect_ratio,
            premium_th=premium_th, standard_th=standard_th,
        )
        if ("Premium" in str(fusion["grade"])) != ("Premium" in str(grade)):
            reasons.append("综合判定与等级不一致")
        needs_review = bool(reasons)
        review_reason = "；".join(reasons)

        result_img = visualize_result(img_p, apple_contour, defect_contours,
                                      defect_mask, grade, defect_ratio, grade_color,
                                      yolo_conf=yolo_conf,
                                      ai_boxes=ai_defect_boxes)
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

        detect_results = detect_boxes
        detect_viz = visualize_yolo_detect(img_p, detect_results)
        detect_viz_name = "detect_" + unique_name
        cv2.imwrite(str(config.UPLOAD_DIR / detect_viz_name), detect_viz)

        return {
            "unique_name": unique_name,
            "upload_path": upload_path,
            "grade": grade,
            "grade_color": grade_color,
            "defect_ratio": float(defect_ratio),
            "defect_count": defect_count,
            "apple_area": apple_pixels,
            "defect_area": defect_pixels,
            "severity": severity,
            "needs_review": needs_review,
            "review_reason": review_reason,
            "fusion": fusion,
            "diam": diam,
            "defect_blocks": defect_blocks[:10] if defect_blocks else [],
            "svm_detail": svm_detail if isinstance(svm_detail, str) else "",
            "rf_detail": rf_detail if isinstance(rf_detail, str) else "",
            "dl_detail": dl_detail if isinstance(dl_detail, str) else "",
            "yolo_detail": yolo_detail if isinstance(yolo_detail, str) else "",
            "viz_names": viz_names,
            "detect_viz_name": detect_viz_name,
            "result_filename": result_filename,
            "svm_grade": svm_grade, "rf_grade": rf_grade,
            "dl_grade": dl_grade, "dl_conf": dl_conf,
            "yolo_grade": yolo_grade, "yolo_conf": yolo_conf,
            "detect_count": len(detect_results) if detect_results else 0,
        }

    def _process_apple(self, upload_path: str, unique_name: str, tuning=None, auto_review=True) -> dict:
        """苹果: 四模型对比全功能（原有逻辑，单图存库）"""
        core = self._apple_detect_core(upload_path, unique_name, tuning)
        if "error" in core:
            self._remove_upload(upload_path)
            return {"error": core["error"]}

        record = DetectionRecord(
            filename=core["unique_name"], result_filename=core["result_filename"],
            category="apple",
            grade=core["grade"], grade_color=str(core["grade_color"]),
            defect_ratio=float(core["defect_ratio"]), defect_count=core["defect_count"],
            apple_area=core["apple_area"], defect_area=core["defect_area"],
            severity=core["severity"], needs_review=core["needs_review"], review_reason=core["review_reason"],
        )
        db.session.add(record)
        db.session.commit()
        if core["needs_review"] and auto_review:
            self._queue_for_review(core["unique_name"])

        return self._apple_result_payload(record, core)

    @staticmethod
    def _apple_result_payload(record, core) -> dict:
        """把核心检测结果组装成前端需要的 dict"""
        viz_names = core["viz_names"]
        fusion = core["fusion"]
        diam = core["diam"]
        return {
            "id": record.id,
            "category": "apple",
            "filename": core["unique_name"],
            "grade": core["grade"],
            "fusion_grade": fusion["grade"],
            "fusion_conf": round(fusion["conf"], 3),
            "fusion_score": round(fusion["score"], 3),
            "fusion_votes": fusion["votes"],
            "diameter_px": round(diam["diameter_px"], 1),
            "diameter_norm": round(diam["diameter_norm"], 3),
            "diameter_aspect": round(diam["aspect"], 3),
            "defect_blocks": core["defect_blocks"],
            "svm_detail": core["svm_detail"],
            "rf_detail": core["rf_detail"],
            "dl_detail": core["dl_detail"],
            "yolo_detail": core["yolo_detail"],
            "svm_image": url_for("static", filename="uploads/" + viz_names["svm"]),
            "rf_image": url_for("static", filename="uploads/" + viz_names["rf"]),
            "dl_image": url_for("static", filename="uploads/" + viz_names["dl"]),
            "yolo_image": url_for("static", filename="uploads/" + viz_names["yolo"]),
            "detect_image": url_for("static", filename="uploads/" + core["detect_viz_name"]),
            "detect_count": core["detect_count"],
            "svm_grade": core["svm_grade"], "rf_grade": core["rf_grade"],
            "dl_grade": core["dl_grade"],
            "dl_conf": f"{core['dl_conf'] * 100:.1f}%",
            "yolo_grade": core["yolo_grade"],
            "yolo_conf": f"{core['yolo_conf'] * 100:.1f}%",
            "defect_ratio": f"{core['defect_ratio'] * 100:.2f}%",
            "defect_count": core["defect_count"],
            "apple_area": core["apple_area"],
            "defect_area": core["defect_area"],
            "severity": core["severity"],
            "needs_review": core["needs_review"],
            "review_reason": core["review_reason"],
            "original_image": url_for("static", filename=f"uploads/{core['unique_name']}"),
            "result_image": url_for("static", filename=f"uploads/{core['result_filename']}"),
        }

    @staticmethod
    def _compose_multi_image(faces, prefix="multi"):
        """把各面的结果图横向拼接成一张综合图"""
        import cv2 as _cv2
        imgs = []
        for f in faces:
            p = config.UPLOAD_DIR / f["result_filename"]
            im = _cv2.imread(str(p))
            if im is not None:
                imgs.append(im)
        if not imgs:
            return None
        h = max(im.shape[0] for im in imgs)
        resized = []
        for im in imgs:
            scale = h / im.shape[0]
            resized.append(_cv2.resize(im, (int(im.shape[1] * scale), h)))
        w = sum(im.shape[1] for im in resized)
        canvas = np.full((h, w, 3), 24, dtype=np.uint8)
        x = 0
        for im in resized:
            canvas[:, x:x + im.shape[1]] = im
            x += im.shape[1]
        return canvas

    def process_multi_upload(self, file_storages, category: str = "apple", tuning=None, auto_review=True) -> dict:
        """多角度检测：同一苹果 2~8 个面，逐面检测，取最差判定（防漏检）"""
        tuning = tuning or {}
        if category != "apple":
            return {"error": "多角度检测当前仅支持苹果品类"}
        files = [f for f in (file_storages or []) if f and f.filename]
        if not files:
            return {"error": "未找到图片文件"}
        if len(files) > 8:
            return {"error": "一次最多上传 8 张图片"}
        if len(files) < 2:
            return {"error": "请至少上传 2 张（同一苹果的不同面）"}

        faces = []
        warnings = []
        worst_rank = -1
        worst_face_idx = -1
        worst_grade = None
        worst_color = None
        worst_ratio = 0.0
        worst_count = 0
        worst_severity = "轻度"
        total_apple = 0
        total_defect = 0
        any_review = False
        review_texts = []

        GRADE_ORDER = {"Premium": 0, "特级": 0, "一级": 1, "Standard": 1,
                       "二级": 2, "Reject": 3, "不合格": 3, "残次": 3}

        def _rank(g):
            return GRADE_ORDER.get(str(g), 2)

        for i, fs in enumerate(files, 1):
            unique_name, upload_path = self._save_upload(fs)
            core = self._apple_detect_core(upload_path, unique_name, tuning)
            if "error" in core:
                warnings.append({"index": i, "name": fs.filename, "error": core["error"]})
                self._remove_upload(upload_path)
                continue
            rk = _rank(core["grade"])
            if rk > worst_rank or (rk == worst_rank and core["defect_ratio"] > worst_ratio):
                worst_rank = rk
                worst_face_idx = i
                worst_grade = core["grade"]
                worst_color = core["grade_color"]
                worst_ratio = core["defect_ratio"]
                worst_count = core["defect_count"]
                worst_severity = core["severity"]
            total_apple += core["apple_area"]
            total_defect += core["defect_area"]
            if core["needs_review"]:
                any_review = True
                review_texts.append(f"面{i}:{core['review_reason']}")
            faces.append({
                "index": i,
                "name": fs.filename,
                "grade": core["grade"],
                "grade_en": str(core["grade"]),
                "defect_ratio": f"{core['defect_ratio'] * 100:.2f}%",
                "defect_count": core["defect_count"],
                "severity": core["severity"],
                "needs_review": core["needs_review"],
                "review_reason": core["review_reason"],
                "fusion_grade": core["fusion"]["grade"],
                "fusion_conf": round(core["fusion"]["conf"], 3),
                "svm_grade": core["svm_grade"], "rf_grade": core["rf_grade"],
                "dl_grade": core["dl_grade"], "yolo_grade": core["yolo_grade"],
                "original_image": url_for("static", filename=f"uploads/{core['unique_name']}"),
                "result_image": url_for("static", filename=f"uploads/{core['result_filename']}"),
                "result_filename": core["result_filename"],
                "detect_image": url_for("static", filename=f"uploads/{core['detect_viz_name']}"),
            })

        if not faces:
            return {"error": "所有图片均检测失败", "warnings": warnings}

        return self._finalize_multi(faces, warnings, worst_rank, worst_face_idx, worst_grade,
                                    worst_color, worst_ratio, worst_count, worst_severity,
                                    total_apple, total_defect, any_review, review_texts, auto_review)

    def _finalize_multi(self, faces, warnings, worst_rank, worst_face_idx, worst_grade,
                        worst_color, worst_ratio, worst_count, worst_severity,
                        total_apple, total_defect, any_review, review_texts, auto_review) -> dict:
        """多角度检测收尾：拼图 + 存一条综合记录 + 组装返回"""
        composed = self._compose_multi_image(faces)
        composed_name = None
        if composed is not None:
            composed_name = f"multi_{uuid.uuid4().hex[:8]}.jpg"
            cv2.imwrite(str(config.UPLOAD_DIR / composed_name), composed)

        # 第一张作为记录主图
        first_url = faces[0]["original_image"]
        first_name = os.path.basename(first_url) if first_url else faces[0]["name"]

        review_reason = "；".join(review_texts) if review_texts else ""
        record = DetectionRecord(
            filename=first_name, result_filename=composed_name or faces[0]["result_image"].split("/")[-1],
            category="apple",
            grade=worst_grade, grade_color=str(worst_color),
            defect_ratio=float(worst_ratio), defect_count=worst_count,
            apple_area=total_apple, defect_area=total_defect,
            severity=worst_severity, needs_review=any_review, review_reason=review_reason,
        )
        db.session.add(record)
        db.session.commit()
        return {
            "multi": True,
            "id": record.id,
            "faces": faces,
            "warnings": warnings,
            "fused": {
                "grade": worst_grade,
                "grade_color": str(worst_color),
                "defect_ratio": f"{worst_ratio * 100:.2f}%",
                "defect_count": worst_count,
                "severity": worst_severity,
                "needs_review": any_review,
                "review_reason": review_reason,
                "face_count": len(faces),
                "worst_face": worst_face_idx,
                "original_image": faces[0]["original_image"],
                "result_image": url_for("static", filename=f"uploads/{composed_name}") if composed_name else faces[0]["result_image"],
            },
        }
