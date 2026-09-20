# -*- coding: utf-8 -*-
"""
predict_freshness.py - 电子鼻单样本预测（分类：新鲜度等级 / 回归：剩余货架期）

输入：一条气味采集 CSV（collect.py 输出，如 data/raw/s001_slice_0min.csv）
流程：预处理（与 preprocess.py 相同的 S-G 滤波 + 60 维特征）→ 加载模型 → 预测

模型（joblib bundle：{"model", "feat_cols", "task", "model_name", ...}）：
  分类：enose/models/freshness_rf.pkl（train.py --mode classify --save）
  回归：enose/models/shelf_plsr.pkl（train.py --mode regress --save）

用法：
  python enose/predict_freshness.py data/raw/s001.csv
  python enose/predict_freshness.py data/raw/s001.csv --model enose/models/shelf_plsr.pkl
  python enose/predict_freshness.py --train-from enose/data/features.csv             # 训练分类再预测
  python enose/predict_freshness.py --train-from enose/data/features.csv --mode regress --target hours --kind plsr
"""
import argparse
import os
import sys
from pathlib import Path

ENOSE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENOSE.parent))  # 使 preprocess 能 import config（如需要）
sys.path.insert(0, str(ENOSE))

import numpy as np
import pandas as pd

try:
    import plsda  # noqa: F401  保证 joblib 反序列化 PLSDA 模型
    import train
except Exception:
    pass

# 复用 preprocess 的滤波/特征逻辑
import preprocess as PP

TARGET_UNITS = {"hours": "小时", "minutes": "分钟", "days": "天", "shelf_hours": "小时"}


def extract_single(path: str) -> np.ndarray:
    """对单条气味 CSV 提取 60 维特征（与 preprocess.py 一致）。"""
    df = pd.read_csv(path, encoding="utf-8")
    df = PP.savgol_filter_df(df)
    feats = []
    for ch in PP.CHANNELS:
        if ch not in df.columns:
            continue
        v = pd.to_numeric(df[ch], errors="coerce").values.astype(float)
        v = v[np.isfinite(v)]
        if len(v) < 15:
            feats.extend([0.0] * 6)
            continue
        start = float(np.mean(v[:5]))
        steady = float(np.mean(v[int(len(v) * 0.7):]))
        rng = float(np.max(v) - np.min(v))
        slope1 = float((v[5] - v[0]) / 5)
        _trapz = getattr(np, "trapezoid", np.trapz)
        auc = float(_trapz(v - start))
        ratio = float(steady / start) if start != 0 else 0.0
        feats += [start, steady, rng, slope1, auc, ratio]
    return np.array(feats, dtype=np.float32)


def train_classifier_from_features(csv_path: str, out_path: Path, kind: str = "rf"):
    """从特征 CSV 训练分类模型并保存（RF 默认；也可 plsda）。"""
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    from train import CLASSIFIERS
    df = pd.read_csv(csv_path, encoding="utf-8")
    feat_cols = [c for c in df.columns if c not in ("label", "source", "group")]
    X, y = df[feat_cols].fillna(0).values.astype(np.float32), df["label"].astype(str).values
    clf = CLASSIFIERS.get(kind, CLASSIFIERS["rf"])()
    clf.fit(X, y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feat_cols": feat_cols, "task": "classify",
                 "model_name": kind, "labels": sorted(set(y))}, out_path)
    print(f"[train] 分类模型已保存: {out_path}（样本 {len(X)}，类别 {sorted(set(y))}，模型 {kind}）")
    return clf, feat_cols


def train_regressor_from_features(csv_path: str, out_path: Path, target: str = "hours", kind: str = "plsr"):
    """从特征 CSV 训练回归模型（剩余货架期/放置时间）并保存。"""
    import joblib
    from train import REGRESSORS
    df = pd.read_csv(csv_path, encoding="utf-8")
    if target not in df.columns:
        sys.exit(f"目标列 {target} 不在特征表里：{list(df.columns)}")
    feat_cols = [c for c in df.columns if c not in ("label", "source", "group") and c != target]
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df[target].astype(float).values
    reg = REGRESSORS.get(kind, REGRESSORS["plsr"])()
    reg.fit(X, y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": reg, "feat_cols": feat_cols, "task": "regress",
                 "model_name": kind, "target": target}, out_path)
    print(f"[train] 回归模型已保存: {out_path}（样本 {len(X)}，目标 {target}，模型 {kind}）")
    return reg, feat_cols


def predict_bundle(feats: np.ndarray, bundle: dict):
    """按 bundle 任务类型预测：返回 (prediction, proba_or_None, meta_dict)。"""
    model = bundle.get("model")
    task = bundle.get("task", "classify")
    feat_cols = bundle.get("feat_cols")
    X = feats.reshape(1, -1)
    if feat_cols:
        X = pd.DataFrame([feats], columns=feat_cols).fillna(0)
    if task == "regress":
        value = float(model.predict(X)[0])
        target = bundle.get("target", "hours")
        unit = TARGET_UNITS.get(target, "小时")
        return value, None, {"task": "regress", "target": target, "unit": unit}
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    return pred, proba, {"task": "classify"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="", help="待预测的气味 CSV")
    ap.add_argument("--model", default="")
    ap.add_argument("--train-from", default="", help="特征 CSV，若提供则先训练模型")
    ap.add_argument("--mode", choices=["classify", "regress"], default="classify", help="训练任务类型")
    ap.add_argument("--target", default="hours", help="回归目标列名")
    ap.add_argument("--kind", default="", help="模型类型：classify 用 rf/plsda；regress 用 plsr/rf/svr")
    args = ap.parse_args()

    import joblib
    model_path = Path(args.model) if args.model else ENOSE / "models" / ("shelf_plsr.pkl" if args.mode == "regress" else "freshness_rf.pkl")

    if args.train_from and Path(args.train_from).exists():
        if args.mode == "regress":
            train_regressor_from_features(args.train_from, model_path, target=args.target, kind=args.kind or "plsr")
        else:
            train_classifier_from_features(args.train_from, model_path, kind=args.kind or "rf")

    if not args.csv:
        raise SystemExit("请提供待预测的 CSV 路径（或 --train-from 后接 CSV 路径）")

    feats = extract_single(args.csv)
    print(f"[feat] 特征维数: {len(feats)}")

    if not model_path.exists():
        raise SystemExit(f"模型不存在: {model_path}（先用 --train-from 训练，或 --model 指定模型）")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        bundle = {"model": bundle, "task": "classify"}
    pred, proba, meta = predict_bundle(feats, bundle)
    if meta["task"] == "regress":
        print(f"预测结果: {pred:.2f} {meta['unit']}")
        print(f"(目标: {meta['target']} | 模型: {bundle.get('model_name', '?')})")
    else:
        print(f"预测结果: {pred}")
        if proba is not None:
            classes = getattr(bundle["model"], "classes_", None)
            names = classes if classes is not None else range(len(proba))
            for k, v in zip(names, proba):
                print(f"  {k}: {v * 100:.1f}%")


if __name__ == "__main__":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()