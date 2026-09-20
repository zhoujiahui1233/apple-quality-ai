# -*- coding: utf-8 -*-
"""
鲁棒性测试：图像扰动（模糊 / 亮度 / 旋转）对四模型准确率的影响

用法:
  python scripts/robustness_test.py                          # 全部扰动
  python scripts/robustness_test.py --perturbations "blur2,rot15"
  python scripts/robustness_test.py --limit 10               # 每标签最多 10 张，快速验证

扰动清单:
  clean          原始图像（基线）
  blur2 / blur4  高斯模糊 sigma=2 / 4
  bright+30 / -30  亮度 +30% / -30%
  rot5 / rot15   旋转 5° / 15°

原理:
  对同一批真实收集图像施加扰动后重新过四个模型，统计每个模型
  相对基线(clean)的准确率下降幅度，评估模型鲁棒性。

输出:
  output/robustness_test.json
  output/robustness_test.csv
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import config

DATASET_DIR = config.ROOT_DIR / "dataset_collected"
OUTPUT_DIR = config.OUTPUT_DIR
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
ALL_PERTURBATIONS = ["clean", "blur2", "blur4", "bright+30", "bright-30", "rot5", "rot15"]
MODELS = ["svm", "rf", "dl", "yolo"]


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def label_classes(folder_name):
    low = folder_name.lower()
    if "不合格" in folder_name or "reject" in low:
        return "bad"
    return "good"


def iter_samples(category, limit=None):
    cat_dir = DATASET_DIR / category
    if not cat_dir.is_dir():
        return
    for label_dir in sorted(cat_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        y = label_classes(label_dir.name)
        files = sorted(label_dir.glob("*"))
        if limit:
            files = files[:limit]
        for f in files:
            if f.suffix.lower() in IMG_EXTS:
                yield f, y


def rotate(img, deg):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def perturb(img, kind):
    if kind == "blur2":
        return cv2.GaussianBlur(img, (0, 0), 2)
    if kind == "blur4":
        return cv2.GaussianBlur(img, (0, 0), 4)
    if kind == "bright+30":
        return np.clip(img.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
    if kind == "bright-30":
        return np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
    if kind == "rot5":
        return rotate(img, 5)
    if kind == "rot15":
        return rotate(img, 15)
    return img


def run_models_on_image(img_bgr):
    """对一张 BGR 图运行四个模型，返回 {model: good/bad}"""
    from apple_defect_detector import (
        preprocess, segment_apple,
        predict_with_svm, predict_with_rf, predict_with_dl, predict_with_yolo,
    )
    img_p, _ = preprocess(img_bgr)
    mask, contour = segment_apple(img_p)
    if mask is None:
        return None
    preds = {}
    try:
        g, *_ = predict_with_svm(img_p, mask, contour)
        preds["svm"] = "bad" if "Reject" in (g or "") else "good"
    except Exception:
        preds["svm"] = None
    try:
        g, *_ = predict_with_rf(img_p, mask, contour)
        preds["rf"] = "bad" if "Reject" in (g or "") else "good"
    except Exception:
        preds["rf"] = None
    try:
        g, *_ = predict_with_dl(img_p)
        preds["dl"] = "bad" if "Reject" in (g or "") else "good"
    except Exception:
        preds["dl"] = None
    try:
        g, *_ = predict_with_yolo(img_p)
        preds["yolo"] = "bad" if "Reject" in (g or "") else "good"
    except Exception:
        preds["yolo"] = None
    return preds


def main():
    parser = argparse.ArgumentParser(description="鲁棒性测试")
    parser.add_argument("--category", default="apple")
    parser.add_argument("--perturbations", default=",".join(ALL_PERTURBATIONS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    perturbs = [p.strip() for p in args.perturbations.split(",") if p.strip()]
    if "clean" not in perturbs:
        perturbs = ["clean"] + perturbs

    samples = list(iter_samples(args.category, args.limit))
    if not samples:
        raise SystemExit(f"没有可用数据: dataset_collected/{args.category}/")
    print(f"[{args.category}] 样本 {len(samples)} 张，扰动 {len(perturbs)} 种，"
          f"模型 {len(MODELS)} 个，预计推理 {len(samples) * len(perturbs) * len(MODELS)} 次")

    acc = {m: {p: [] for p in perturbs} for m in MODELS}
    detail = []
    t0 = time.time()
    for i, (f, y) in enumerate(samples, 1):
        img = imread_unicode(f)
        if img is None:
            continue
        row = {"file": str(f.relative_to(ROOT)), "y_true": y}
        for p in perturbs:
            pimg = perturb(img, p)
            preds = run_models_on_image(pimg)
            for m in MODELS:
                hit = (preds or {}).get(m) == y
                acc[m][p].append(int(bool(hit)))
                row[f"{p}_{m}"] = preds.get(m) if preds else None
        detail.append(row)
        if i % 5 == 0:
            print(f"  已处理 {i}/{len(samples)}，耗时 {time.time() - t0:.0f}s")

    summary = {}
    for m in MODELS:
        summary[m] = {}
        base = np.mean(acc[m]["clean"]) * 100 if acc[m]["clean"] else 0.0
        summary[m]["clean"] = round(float(base), 2)
        for p in perturbs[1:]:
            v = np.mean(acc[m][p]) * 100 if acc[m][p] else 0.0
            summary[m][p] = round(float(v), 2)

    payload = {
        "source": "robustness_test",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category": args.category,
        "n": len(samples),
        "perturbations": perturbs,
        "models": MODELS,
        "accuracy": summary,
        "note": "准确率为二分类(合格/不合格)一致率",
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    json_path = OUTPUT_DIR / "robustness_test.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    csv_path = OUTPUT_DIR / "robustness_test.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["model"] + perturbs)
        for m in MODELS:
            w.writerow([m] + [summary[m][p] for p in perturbs])

    print(f"\n鲁棒性测试结果（{len(samples)} 张，准确率 %）:")
    print("  模型   " + "".join(f"{p:>11}" for p in perturbs))
    for m in MODELS:
        print(f"  {m:<6}" + "".join(f"{summary[m][p]:>10.2f}%" for p in perturbs))
    print(f"\n耗时 {time.time() - t0:.0f}s")
    print(f"已保存: {json_path} / {csv_path}")


if __name__ == "__main__":
    main()