# -*- coding: utf-8 -*-
"""
train_defect_type.py - 缺陷类型自动分类器训练（计划 8.1 第二步）

触发条件：数据集中"碰伤/腐烂/虫眼"每类累计 >= 300 张（前端保存时手动选类型已在使用）。
数据来源：dataset_collected/{category}/{label}/*.jpg 为合格/不合格；
         缺陷类型标注建议统一放在 dataset_collected/{category}/defect_types.csv：
         csv 列: filename,label(碰伤/腐烂/虫眼/斑点/畸形/其他)

流程：
  1) 读取标注 CSV，逐张图调用 analyze_defect_blocks() 提取缺陷块特征
  2) 每个缺陷块特征 + 该图类型标签 → 训练 RF / SVM 分类器
  3) 5 折交叉验证 + 测试集报告，保存模型为 models/defect_type_model.pkl

用法：
  python train_defect_type.py --csv dataset_collected/apple/defect_types.csv
  python train_defect_type.py --csv ... --test-size 0.2 --model rf
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

import apple_defect_detector as D


# 缺陷块特征 -> 扁平特征向量（与 FEATURE_KEYS 对齐，用于 sklearn）
NUMERIC_KEYS = ["area", "area_ratio", "circularity", "aspect", "dist_norm"]
HSV_KEYS = ["hsv_mean"]   # 3 维
GLCM_KEYS = ["glcm"]      # 4 维


def block_to_vector(block: dict) -> np.ndarray:
    """把 analyze_defect_blocks 的单块 dict 转成定长特征向量（TODO: 形状可扩展）。"""
    v = [float(block[k]) for k in NUMERIC_KEYS]
    v += [float(x) for x in block["hsv_mean"]]
    v += [float(x) for x in block["glcm"]]
    return np.array(v, dtype=np.float32)


def extract_blocks_from_labeled_csv(csv_path: str, img_root: Path):
    """??? CSV??? (X, y, filenames)?

    ???????
      1) ?????defect_blocks.csv?? area/circularity/aspect/hsv/glcm + type ???
         ???????????????
      2) ?????filename,label?????????????????????
    """
    import json as _json
    df = pd.read_csv(csv_path, encoding="utf-8")
    label_col = "label" if "label" in df.columns else ("type" if "type" in df.columns else None)
    if "filename" not in df.columns or label_col is None:
        raise SystemExit("CSV ??? filename ? label/type ?")
    has_block_feats = all(c in df.columns for c in ("area", "circularity", "aspect"))
    X, y, names = [], [], []
    for _, row in df.iterrows():
        if has_block_feats:
            # ?????????????? defect_blocks.csv?
            v = [float(row.get("area", 0) or 0), float(row.get("area_ratio", 0) or 0),
                 float(row.get("circularity", 0) or 0), float(row.get("aspect", 0) or 0),
                 float(row.get("dist_norm", 0) or 0)]
            hsv_raw = row.get("hsv") if "hsv" in df.columns else row.get("hsv_mean", "[]")
            glcm_raw = row.get("glcm", "[]")
            try:
                hsv_v = _json.loads(hsv_raw) if isinstance(hsv_raw, str) else list(hsv_raw)
                glcm_v = _json.loads(glcm_raw) if isinstance(glcm_raw, str) else list(glcm_raw)
            except Exception:
                hsv_v, glcm_v = [], []
            v += [float(x) for x in hsv_v[:3]]
            v += [float(x) for x in glcm_v[:4]]
            X.append(np.array(v, dtype=np.float32))
            y.append(str(row[label_col]).strip())
            names.append(str(row["filename"]))
            continue
        # ??????????????
        img_path = img_root / str(row["filename"])
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] ????: {img_path}")
            continue
        mask, _ = D.segment_apple(img)
        if mask is None:
            continue
        defect_mask, _ = D.detect_defects(img, mask)
        blocks = D.analyze_defect_blocks(img, mask, defect_mask)
        if not blocks:
            continue
        # TODO: ???"?????"?"??????"???????
        X.append(block_to_vector(blocks[0]))
        y.append(str(row[label_col]).strip())
        names.append(str(row["filename"]))
    return np.array(X, dtype=np.float32), np.array(y), names

def train(X, y, model_name="rf", cv=5):
    """训练并评估。TODO: 类别不均衡时加 class_weight / 重采样。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
    import joblib

    clf = (RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
           if model_name == "rf"
           else make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)))
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"[{model_name}] {cv}-折交叉验证准确率: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print("\n分类报告:")
    print(classification_report(yte, pred, zero_division=0))
    kappa = cohen_kappa_score(yte, pred)
    print(f"与人工标注 Kappa: {kappa:.3f}  (验收线 >=0.6)")
    return clf, kappa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="缺陷类型标注 CSV（filename,label）")
    ap.add_argument("--img-root", default="", help="图片根目录，默认取 CSV 所在目录")
    ap.add_argument("--model", choices=["rf", "svm"], default="rf")
    ap.add_argument("--out", default="models/defect_type_model.pkl")
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    img_root = Path(args.img_root) if args.img_root else Path(args.csv).parent
    X, y, names = extract_blocks_from_labeled_csv(args.csv, img_root)
    print(f"样本数: {len(X)}  类别: {sorted(set(y))}  特征维数: {X.shape[1]}")
    if len(X) < 20:
        print("[warn] 样本过少，结果仅供流程验证；达到每类 300 张后重训。")
    clf, kappa = train(X, y, args.model, args.cv)
    os.makedirs(Path(args.out).parent, exist_ok=True)
    import joblib
    joblib.dump(clf, args.out)
    print(f"\n模型已保存: {args.out}")


if __name__ == "__main__":
    main()
