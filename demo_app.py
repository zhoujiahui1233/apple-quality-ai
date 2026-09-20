# -*- coding: utf-8 -*-
"""临时演示版：只跑现有苹果检测模型，不依赖数据库 / 登录 / 数据集。

用法：python demo_app.py   （端口 5030）
"""
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.ultralytics')

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

import config
from apple_defect_detector import (
    preprocess, segment_apple, detect_defects, classify_quality,
    predict_with_svm, predict_with_rf, predict_with_dl, predict_with_yolo,
    predict_with_yolo_detect,
    visualize_result, visualize_svm, visualize_rf, visualize_dl,
    visualize_yolo, visualize_yolo_detect,
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

DEMO_DIR = config.INSTANCE_DIR / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def _clear_demo_dir():
    """临时目录：每次启动清空旧图"""
    try:
        for f in DEMO_DIR.glob("*"):
            if f.is_file():
                f.unlink()
    except Exception:
        pass


def warmup():
    from apple_defect_detector import (
        load_scaler, load_svm_model, load_rf_model,
        load_dl_model, load_yolo_model, load_yolo_detect_model,
    )
    print("正在加载模型...")
    for name, loader in [
        ("Scaler", load_scaler), ("SVM", load_svm_model), ("RF", load_rf_model),
        ("DL", load_dl_model), ("YOLO-CLS", load_yolo_model),
        ("YOLO-DET", load_yolo_detect_model),
    ]:
        try:
            loader()
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")
    print("模型加载完成")


def _pct(x):
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "N/A"


def _fmt_detail(d):
    return d if isinstance(d, str) else (str(d) if d else "")


@app.route('/')
def index():
    return render_template('demo.html')


@app.route('/demo_out/<path:name>')
def demo_out(name):
    return send_from_directory(DEMO_DIR, name)


@app.route('/api/detect', methods=['POST'])
def detect():
    f = request.files.get('file')
    if not f:
        return jsonify({"error": "未收到图片"}), 400
    ext = Path(f.filename or "a.jpg").suffix.lower() or ".jpg"
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "不支持的图片格式"}), 400

    uid = uuid.uuid4().hex
    in_name = f"in_{uid}{ext}"
    in_path = DEMO_DIR / in_name
    f.save(str(in_path))

    img = cv2.imread(str(in_path))
    if img is None:
        return jsonify({"error": "无法解析图片，请换一张"}), 400
    h, w = img.shape[:2]
    if max(w, h) > 1600:
        s = 1600.0 / max(w, h)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    img_p, _ = preprocess(img)
    apple_mask, apple_contour = segment_apple(img_p)
    if apple_mask is None:
        return jsonify({"error": "未检测到苹果区域，请换一张图片"}), 400

    apple_pixels = int(np.sum(apple_mask > 0))
    defect_mask, defect_contours = detect_defects(img_p, apple_mask)
    defect_pixels = int(np.sum(defect_mask > 0))
    defect_count = len(defect_contours)
    grade, defect_ratio, grade_color = classify_quality(defect_pixels, apple_pixels)

    yolo_grade, yolo_color, yolo_conf, yolo_detail = predict_with_yolo(img_p)
    if yolo_grade is not None and yolo_conf > 0.5:
        grade = yolo_grade
        grade_color = yolo_color

    svm_grade, svm_color, svm_detail = predict_with_svm(img_p, apple_mask, apple_contour)
    rf_grade, rf_color, rf_detail = predict_with_rf(img_p, apple_mask, apple_contour)
    dl_grade, dl_color, dl_conf, dl_detail = predict_with_dl(img_p)

    result_img = visualize_result(img_p, apple_contour, defect_contours, defect_mask,
                                  grade, defect_ratio, grade_color, yolo_conf=yolo_conf)
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
    detect_results = predict_with_yolo_detect(img_p)
    detect_viz = visualize_yolo_detect(img_p, detect_results)

    out = {}
    for key, viz in [("result", result_img), ("svm", svm_viz), ("rf", rf_viz),
                     ("dl", dl_viz), ("yolo", yolo_viz), ("detect", detect_viz)]:
        name = f"{key}_{uid}.jpg"
        cv2.imwrite(str(DEMO_DIR / name), viz)
        out[key] = f"/demo_out/{name}"

    # 四模型多数表决
    votes = [g for g in (svm_grade, rf_grade, dl_grade, yolo_grade) if g]
    consensus = None
    if votes:
        from collections import Counter
        consensus, cnt = Counter(votes).most_common(1)[0]
        consensus = {"grade": consensus, "votes": cnt, "total": len(votes)}

    return jsonify({
        "grade": grade,
        "grade_en": "Reject" if "Reject" in str(grade) else ("Premium" if "Premium" in str(grade) else "Standard"),
        "defect_ratio": _pct(defect_ratio),
        "defect_count": defect_count,
        "apple_area": apple_pixels,
        "defect_area": defect_pixels,
        "models": {
            "yolo": {"grade": str(yolo_grade) if yolo_grade else "N/A", "conf": _pct(yolo_conf),
                     "detail": _fmt_detail(yolo_detail), "image": out["yolo"]},
            "dl": {"grade": str(dl_grade) if dl_grade else "N/A", "conf": _pct(dl_conf),
                   "detail": _fmt_detail(dl_detail), "image": out["dl"]},
            "rf": {"grade": str(rf_grade) if rf_grade else "N/A", "conf": None,
                   "detail": _fmt_detail(rf_detail), "image": out["rf"]},
            "svm": {"grade": str(svm_grade) if svm_grade else "N/A", "conf": None,
                    "detail": _fmt_detail(svm_detail), "image": out["svm"]},
        },
        "consensus": consensus,
        "result_image": out["result"],
        "detect_image": out["detect"],
        "detect_count": len(detect_results) if detect_results else 0,
        "original_image": f"/demo_out/{in_name}",
    })


if __name__ == '__main__':
    _clear_demo_dir()
    warmup()
    print("演示版地址: http://127.0.0.1:5030/")
    app.run(host="0.0.0.0", port=5030, debug=False, threaded=True)