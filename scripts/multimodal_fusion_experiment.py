# -*- coding: utf-8 -*-
"""
multimodal_fusion_experiment.py - 视觉 + 嗅觉(+理化) 多模态融合实验（完整版）

对比"单模态 vs 多模态融合"的分类性能，输出论文图表数据：
  - 单模态基线：visual / odor / phys
  - 融合：concat（特征拼接）/ late（概率平均）
  - 显著性：融合 vs 最优单模态的 McNemar 检验
  - 特征重要性：RF 的 Top 特征

输入（按 sample_id 对齐的 CSV）：
  --visual  视觉特征表（含 sample_id,label）
  --odor    气味特征表（enose/preprocess.py 输出 60 维，含 source/label；自动映射 source->sample_id）
  --phys    理化表（含 sample_id,label）

输出：output/fusion_experiment.json + .csv
用法：
  python scripts/multimodal_fusion_experiment.py --visual ... --odor enose/data/features.csv
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score


def _infer_label(sample_id: str) -> str:
    """? sample_id ????????? mock_fresh_00 -> fresh?????? unknown?"""
    s = str(sample_id).lower()
    for kw in ("rotten", "bad", "unqualified", "???", "mid", "fresh", "good", "??", "overripe", "ripe"):
        if kw in s:
            return kw
    return "unknown"


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    if "sample_id" not in df.columns:
        if "source" in df.columns:  # enose 特征表用 source 列
            df = df.rename(columns={"source": "sample_id"}); df["sample_id"] = df["sample_id"].astype(str).str.replace(r"\.[a-zA-Z]+$", "", regex=True)
        else:
            df["sample_id"] = [f"row_{i}" for i in range(len(df))]
    if "label" not in df.columns:
        df["label"] = df["sample_id"].map(_infer_label)
    return df


def feature_cols(df, label_col="label"):
    return [c for c in df.columns
            if c != "sample_id" and not c.startswith("label")]


def _train_eval(X, y, cv=5, seed=42):
    """5 折交叉验证，返回 (acc, std, clf, X_test, y_test, pred_test, proba_test)。"""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    accs, models = [], []
    # 保存最后一次 fold 的测试结果用于 McNemar
    last = None
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        models.append(clf)
        last = (X[te], y[te], pred, clf.predict_proba(X[te]))
    return float(np.mean(accs)), float(np.std(accs)), models, last


def mcnemar(y_true, p1, p2):
    """McNemar 检验（融合 vs 单模态）。返回 p 值（连续性校正）。"""
    from scipy.stats import binom
    b = int(np.sum((p1 != y_true) & (p2 == y_true)))
    c = int(np.sum((p1 == y_true) & (p2 != y_true)))
    n = b + c
    if n == 0:
        return 1.0
    # 二项检验（双边），等价 McNemar
    p = 2 * min(binom.cdf(min(b, c), n, 0.5), 1 - binom.cdf(min(b, c) - 1, n, 0.5))
    return float(p)


def late_fusion_proba(models_list, X):
    """概率平均 late fusion：每个模型 predict_proba -> 平均 -> argmax。"""
    probs = [m.predict_proba(X) for m in models_list]
    mean_p = np.mean(probs, axis=0)
    return mean_p.argmax(axis=1), mean_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual", default="")
    ap.add_argument("--odor", default="")
    ap.add_argument("--phys", default="")
    ap.add_argument("--fuse", choices=["concat", "late", "both"], default="both")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--out", default="output/fusion_experiment.json")
    args = ap.parse_args()

    inputs = []
    for name, key in [("visual", "visual"), ("odor", "odor"), ("phys", "phys")]:
        p = getattr(args, key)
        if p and Path(p).exists():
            inputs.append((name, load_features(Path(p))))
    if not inputs:
        raise SystemExit("至少提供一个特征 CSV（--visual/--odor/--phys）")
    for name, df in inputs:
        print(f"[{name}] 样本 {len(df)}，特征 {len(feature_cols(df))}")

    results = []

    def eval_one(X, y, name):
        acc, std, models, last = _train_eval(X, y, args.cv)
        r = {"modality": name, "acc": round(acc * 100, 2), "std": round(std * 100, 2),
             "n": len(X), "cv": args.cv}
        results.append(r)
        print(f"[{name}] acc={acc*100:.2f}% (+/-{std*100:.2f}%)")
        return acc, models, last

    # 单模态
    single = {}
    for name, df in inputs:
        X = df[feature_cols(df)].fillna(0).values.astype(np.float32)
        y = df["label"].astype(str).values
        single[name] = eval_one(X, y, name)

    # 融合
    if len(inputs) >= 2:
        # concat：按 sample_id 内连接
        merged = inputs[0][1]
        for name, df in inputs[1:]:
            keep = ["sample_id"] + feature_cols(df)
            merged = merged.merge(df[keep], on="sample_id", suffixes=("", f"_{name}"), how="inner")
        merged = merged.dropna(subset=["label"])
        Xc = merged[feature_cols(merged)].fillna(0).values.astype(np.float32)
        yc = merged["label"].astype(str).values
        print(f"[concat] 对齐样本 {len(Xc)}，特征数 {Xc.shape[1]}")
        mc, _, lastc = eval_one(Xc, yc, "fused_concat")

        # 特征重要性（最后一次 fold 的模型）
        clf_last = _train_eval(Xc, yc, args.cv)[2][-1]
        imp = pd.Series(clf_last.feature_importances_, index=feature_cols(merged)).sort_values(ascending=False)
        top = imp.head(10).to_dict()
        print("[concat] Top-10 特征:", list(top.keys()))

        # late fusion：用各单模态在同一测试集上的概率平均（构造对齐样本的 KFold）
        if args.fuse in ("late", "both"):
            ids = merged["sample_id"].astype(str).values
            # 每个模态按 ids 对齐
            mod_X = {}
            for name, df in inputs:
                m = df.merge(merged[["sample_id"]], on="sample_id", how="inner")
                mod_X[name] = m[feature_cols(m)].fillna(0).values.astype(np.float32)
            skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
            accs_late = []
            for tr, te in skf.split(Xc, yc):
                probs = []
                for name in mod_X:
                    Xm = mod_X[name]
                    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
                    clf.fit(Xm[tr], yc[tr])
                    probs.append(clf.predict_proba(Xm[te]))
                mean_p = np.mean(probs, axis=0)
                pred = np.array([clf.classes_[np.argmax(p)] for p in mean_p])
                accs_late.append(accuracy_score(yc[te], pred))
            r = {"modality": "fused_late", "acc": round(float(np.mean(accs_late)) * 100, 2),
                 "std": round(float(np.std(accs_late)) * 100, 2), "n": len(Xc), "cv": args.cv}
            results.append(r)
            print(f"[fused_late] acc={np.mean(accs_late)*100:.2f}%")

        # McNemar：融合 concat vs 最优单模态（用 concat 最后一次 fold 的测试集）
        best_single_name = max(single, key=lambda k: single[k][0])
        # 重新在 concat 对齐集上评估最优单模态
        mod_best = inputs[[n for n, _ in inputs].index(best_single_name)][1]
        m2 = mod_best.merge(merged[["sample_id"]], on="sample_id", how="inner")
        Xb = m2[feature_cols(m2)].fillna(0).values.astype(np.float32)
        # 最后一次 fold
        _, _, models_b, last_b = _train_eval(Xb, yc, args.cv)
        p_mc = mcnemar(lastc[1], last_b[2], lastc[2])
        results.append({"modality": "mcnemar", "fused_vs": best_single_name,
                        "p_value": p_mc, "significant": p_mc < 0.05})
        print(f"[McNemar] fused_concat vs {best_single_name}: p={p_mc:.4f}")

    out = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {n: str(getattr(args, k)) for n, k in [("visual", "visual"), ("odor", "odor"), ("phys", "phys")]},
        "results": results,
    }
    os.makedirs(Path(args.out).parent, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出:", args.out)


if __name__ == "__main__":
    main()
