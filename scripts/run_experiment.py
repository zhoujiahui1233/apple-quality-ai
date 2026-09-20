# -*- coding: utf-8 -*-
"""
实验脚本：在收集的数据集上运行模型，生成显著性分析数据

用法:
  python scripts/run_experiment.py                  # 全部品类
  python scripts/run_experiment.py --category apple # 单品类
  python scripts/run_experiment.py --limit 30       # 每品类最多30张

输出:
  output/experiment_results.json   (显著性页面数据源)
  output/experiment_results.csv    (逐样本预测明细)

数据集目录结构: dataset_collected/{category}/{label}/*.jpg
"""
import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config
import cv2
import numpy as np

DATASET_DIR = config.ROOT_DIR / "dataset_collected"
OUTPUT_DIR = config.OUTPUT_DIR


def imread_unicode(path):
    """支持中文路径的图片读取（OpenCV imread 对 Windows 非·ASCII 路径支持不好）"""
    import numpy as np
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def label_classes(folder_name: str):
    """把文件夹名规范化为二分类与三分类标签"""
    low = folder_name.lower()
    if "不合格" in folder_name or "reject" in low:
        return "bad", "reject"
    if "优" in folder_name or "premium" in low:
        return "good", "premium"
    return "good", "standard"


def model_binary(grade_str):
    """苹果模型输出 -> good/bad"""
    if not grade_str:
        return None
    return "bad" if "Reject" in grade_str else "good"


def iter_samples(category: str, limit: int):
    cat_dir = DATASET_DIR / category
    if not cat_dir.is_dir():
        return
    for label_dir in sorted(cat_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        yb, y3 = label_classes(label_dir.name)
        files = sorted(label_dir.glob("*"))[:limit] if limit else sorted(label_dir.glob("*"))
        for f in files:
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                yield f, yb, y3


def _score_svm(img_p):
    """SVM 决策值 -> 优质概率分数（供 ROC/AUC）"""
    from apple_defect_detector import load_scaler, load_svm_model, extract_features_v2
    try:
        model = load_svm_model()
        scaler = load_scaler()
        if model is None:
            return None
        feats = extract_features_v2(img_p)
        if feats is None:
            return None
        if scaler is not None:
            feats = scaler.transform(feats.reshape(1, -1)).flatten()
        _, dist = model.predict(np.array([feats], dtype=np.float32),
                                flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
        d = float(dist[0, 0])
        return 1.0 / (1.0 + math.exp(-d))
    except Exception:
        return None


def _score_rf(img_p):
    """随机森林 predict_proba -> 优质概率分数"""
    from apple_defect_detector import load_rf_model, extract_features_v2
    try:
        model = load_rf_model()
        if model is None:
            return None
        feats = extract_features_v2(img_p)
        if feats is None:
            return None
        proba = model.predict_proba(np.array([feats], dtype=np.float32))[0]
        return float(proba[0])
    except Exception:
        return None


def _conf_to_score(conf, binary):
    """DL/YOLO 的 top1 置信度 -> 优质概率分数（避免二次推理）"""
    if conf is None or binary is None:
        return None
    try:
        conf = float(conf)
    except Exception:
        return None
    return conf if binary == "good" else 1.0 - conf


def run_apple(f, yb, y3):
    from apple_defect_detector import (
        preprocess, segment_apple, detect_defects, classify_quality,
        predict_with_svm, predict_with_rf, predict_with_dl, predict_with_yolo,
    )
    img = imread_unicode(f)
    if img is None:
        return None
    img_p, _ = preprocess(img)
    mask, contour = segment_apple(img_p)
    if mask is None:
        return None
    row = {"svm": None, "rf": None, "dl": None, "yolo": None, "pipeline3": None,
           "scores": {}, "latency": {}}
    _t = time.perf_counter()
    try:
        g, _, _ = predict_with_svm(img_p, mask, contour)
        row["svm"] = model_binary(g)
    except Exception:
        pass
    row["latency"]["svm"] = (time.perf_counter() - _t) * 1000
    row["scores"]["svm"] = _score_svm(img_p)

    _t = time.perf_counter()
    try:
        g, _, _ = predict_with_rf(img_p, mask, contour)
        row["rf"] = model_binary(g)
    except Exception:
        pass
    row["latency"]["rf"] = (time.perf_counter() - _t) * 1000
    row["scores"]["rf"] = _score_rf(img_p)

    _t = time.perf_counter()
    try:
        g, _, conf, _ = predict_with_dl(img_p)
        row["dl"] = model_binary(g)
        row["scores"]["dl"] = _conf_to_score(conf, row["dl"])
    except Exception:
        pass
    row["latency"]["dl"] = (time.perf_counter() - _t) * 1000

    _t = time.perf_counter()
    try:
        g, _, conf, _ = predict_with_yolo(img_p)
        row["yolo"] = model_binary(g)
        row["scores"]["yolo"] = _conf_to_score(conf, row["yolo"])
    except Exception:
        pass
    row["latency"]["yolo"] = (time.perf_counter() - _t) * 1000
    # 管线三级分级
    try:
        dm, dc = detect_defects(img_p, mask)
        dp = int(np.sum(dm > 0))
        ap = int(np.sum(mask > 0))
        g3, _, _ = classify_quality(dp, ap)
        row["pipeline3"] = (g3 or "").lower()  # premium/standard/reject
    except Exception:
        pass
    return row


def run_generic(f, category):
    from detector.pipeline import run_detection
    img = imread_unicode(f)
    if img is None:
        return None
    try:
        pipe = run_detection(img, category)
        if "error" in pipe:
            return None
        return pipe["grade"]  # premium/standard/reject
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="实验脚本: 运行模型并生成显著性数据")
    parser.add_argument("--category", default=None, help="apple/bread/beverage, 默认全部")
    parser.add_argument("--limit", type=int, default=None, help="每个标签最多运行的图片数")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    categories = [args.category] if args.category else ["apple", "bread", "beverage"]

    summary = {}
    samples = []
    for cat in categories:
        rows = []
        for f, yb, y3 in iter_samples(cat, args.limit):
            if cat == "apple":
                row = run_apple(f, yb, y3)
                if row is None:
                    continue
                row.update({"file": str(f.relative_to(ROOT)), "y_true": yb, "y3": y3})
            else:
                g3 = run_generic(f, cat)
                if g3 is None:
                    continue
                row = {"file": str(f.relative_to(ROOT)), "y_true": yb, "y3": y3,
                       "pipeline3": g3, "svm": None, "rf": None, "dl": None, "yolo": None}
            rows.append(row)
            samples.append({"category": cat, **row})
        if not rows:
            continue

        from services.experiment_service import summarize_category
        cat_sum = summarize_category(rows)
        summary[cat] = cat_sum
        print(f"[{cat}] 样本 {len(rows)} 张")

    if not summary:
        print("未找到数据集，请先在 dataset_collected/{category}/{label}/ 放入图片")
        sys.exit(1)

    # 汇总为 API 结构
    from services.experiment_service import build_result
    base = build_result(summary)
    result = {
        "source": "experiment",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "categories": base["categories"],
        "models": base["models"],
        "accuracy": base["accuracy"],
        "n": base["n"],
        "mcnemar": base["mcnemar"],
        "grade_accuracy": base["grade_accuracy"],
        "details": base["details"],
        "note": "实验数据: 自动生成于 " + datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    json_path = OUTPUT_DIR / "experiment_results.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = OUTPUT_DIR / "experiment_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["category", "file", "y_true", "y3", "svm", "rf", "dl", "yolo", "pipeline3"])
        for s in samples:
            w.writerow([s["category"], s["file"], s["y_true"], s["y3"],
                        s.get("svm", ""), s.get("rf", ""), s.get("dl", ""),
                        s.get("yolo", ""), s.get("pipeline3", "")])

    print(f"已生成: {json_path}")
    print(f"已生成: {csv_path}")
    print("显著性页面将自动读取实验数据。")


if __name__ == "__main__":
    main()
