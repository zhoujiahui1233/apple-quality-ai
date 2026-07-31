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
import sys
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
    row = {"svm": None, "rf": None, "dl": None, "yolo": None, "pipeline3": None}
    try:
        g, _, _ = predict_with_svm(img_p, mask, contour)
        row["svm"] = model_binary(g)
    except Exception:
        pass
    try:
        g, _, _ = predict_with_rf(img_p, mask, contour)
        row["rf"] = model_binary(g)
    except Exception:
        pass
    try:
        g, _, _, _ = predict_with_dl(img_p)
        row["dl"] = model_binary(g)
    except Exception:
        pass
    try:
        g, _, _, _ = predict_with_yolo(img_p)
        row["yolo"] = model_binary(g)
    except Exception:
        pass
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

        cat_sum = {"n": len(rows), "grade_accuracy": None}
        if cat == "apple":
            from services.stats_service import model_comparison_table
            preds = []
            for m in ["svm", "rf", "dl", "yolo"]:
                ys = [r[m] for r in rows if r[m] is not None]
                yt = [r["y_true"] for r in rows if r[m] is not None]
                preds.append({"name": m, "y_pred": ys, "y_true": yt})
            table = model_comparison_table(preds)
            cat_sum["models"] = [
                {"name": m["name"], "accuracy": round(m["accuracy"] * 100, 2),
                 "precision": [round(x, 3) for x in m["precision"]],
                 "recall": [round(x, 3) for x in m["recall"]],
                 "f1": [round(x, 3) for x in m["f1"]], "n": len(m["cm"][0])}
                for m in table["models"]
            ]
            cat_sum["mcnemar"] = table["mcnemar"]
        # 管线分级准确率（二分类：premium/standard -> good）
        g3_ok = [1 for r in rows
                 if (r["pipeline3"] in ("premium", "standard")) == (r["y_true"] == "good")]
        if g3_ok:
            cat_sum["grade_accuracy"] = round(len(g3_ok) / len(rows) * 100, 2)
        summary[cat] = cat_sum
        print(f"[{cat}] 样本 {len(rows)} 张")

    if not summary:
        print("未找到数据集，请先在 dataset_collected/{category}/{label}/ 放入图片")
        sys.exit(1)

    # 汇总为 API 结构
    models_union = []
    accuracy = {}
    n_map = {}
    mcnemar = {}
    for cat, cs in summary.items():
        n_map[cat] = cs["n"]
        accuracy[cat] = {}
        for m in cs.get("models", []):
            accuracy[cat][m["name"]] = m["accuracy"]
            if m["name"] not in models_union:
                models_union.append(m["name"])
        if "mcnemar" in cs:
            mcnemar[cat] = cs["mcnemar"]
        if cs["grade_accuracy"] is not None:
            accuracy[cat]["pipeline"] = cs["grade_accuracy"]
            if "pipeline" not in models_union:
                models_union.append("pipeline")

    result = {
        "source": "experiment",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "categories": list(summary.keys()),
        "models": models_union,
        "accuracy": accuracy,
        "n": n_map,
        "mcnemar": mcnemar,
        "grade_accuracy": {c: s["grade_accuracy"] for c, s in summary.items()},
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
