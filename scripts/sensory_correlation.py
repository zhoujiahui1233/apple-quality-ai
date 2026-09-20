# -*- coding: utf-8 -*-
"""
sensory_correlation.py - 仪器指标 vs 人工感官评分（QDA）相关性分析（完整版）

输出论文图表数据：
  - QDA 内部一致性（各评分项均值/标准差）
  - 皮尔逊相关矩阵（仪器特征 x 感官评分，r 与 p）
  - 回归：RF 用仪器特征预测感官评分（R²、MAE）
  - PCA：特征载荷 Top，解释度
  - Kappa：机器分级 vs 人工分级一致性（需分级列）

输入：sensory_scores.csv（sample_id + QDA 评分列 1~9 分）
      --feature 仪器特征表（sample_id + 特征列）
      --grade-col / --true-col（可选，用于 Kappa）
输出：output/sensory_correlation.json + .csv
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


def pearson_matrix(df, feat_cols, score_cols):
    from scipy import stats
    mat = {}
    for sc in score_cols:
        row = {}
        for f in feat_cols:
            s = df[[f, sc]].dropna()
            if len(s) < 5 or s[f].std() == 0 or s[sc].std() == 0:
                row[f] = {"r": None, "p": None}
                continue
            r, p = stats.pearsonr(s[f], s[sc])
            row[f] = {"r": round(float(r), 3), "p": round(float(p), 4)}
        # 按 |r| 排序
        mat[sc] = dict(sorted(row.items(), key=lambda kv: abs(kv[1]["r"] or 0), reverse=True))
    return mat


def regression(df, feat_cols, score_cols):
    """RF 回归：仪器特征预测感官评分，输出 R²/MAE（5 折）。"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import mean_absolute_error
    res = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for sc in score_cols:
        sub = df[feat_cols + [sc]].dropna()
        if len(sub) < 10:
            res[sc] = {"note": "样本不足(<10)"}
            continue
        X, y = sub[feat_cols].values.astype(np.float32), sub[sc].values
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        r2s, maes = [], []
        for tr, te in kf.split(X):
            rf.fit(X[tr], y[tr])
            p = rf.predict(X[te])
            r2s.append(rf.score(X[te], y[te]))
            maes.append(mean_absolute_error(y[te], p))
        res[sc] = {"r2": round(float(np.mean(r2s)), 3), "mae": round(float(np.mean(maes)), 3),
                   "n": len(sub)}
    return res


def pca_loadings(df, feat_cols, n=6):
    """PCA 前 n 个主成分的载荷 Top 特征与解释度。"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = df[feat_cols].dropna()
    if len(X) < 5:
        return {"note": "样本不足"}
    Xs = StandardScaler().fit_transform(X)
    k = min(n, len(feat_cols), len(X) - 1)
    pca = PCA(n_components=k)
    pca.fit(Xs)
    evr = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    load = {}
    for i in range(k):
        top_idx = np.argsort(np.abs(pca.components_[i]))[::-1][:5]
        load[f"PC{i+1}"] = {
            "explained": evr[i],
            "top_features": [{"feature": feat_cols[j], "loading": round(float(pca.components_[i][j]), 3)} for j in top_idx],
        }
    return {"explained_variance": evr, "components": load}


def kappa_agreement(df, pred_col, true_col):
    from sklearn.metrics import cohen_kappa_score
    a = df[[pred_col, true_col]].dropna()
    if len(a) < 10:
        return None
    return round(float(cohen_kappa_score(a[pred_col].astype(str), a[true_col].astype(str))), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="QDA 评分 CSV（sample_id + 评分列）")
    ap.add_argument("--feature", default="", help="仪器特征 CSV（可选）")
    ap.add_argument("--score-cols", default="", help="评分列，逗号分隔；缺省自动取数值列")
    ap.add_argument("--grade-col", default="", help="机器分级列（可选，用于 Kappa）")
    ap.add_argument("--true-col", default="", help="人工分级列（可选，用于 Kappa）")
    ap.add_argument("--out", default="output/sensory_correlation.json")
    args = ap.parse_args()

    scores = pd.read_csv(args.scores, encoding="utf-8")
    if "sample_id" not in scores.columns:
        raise SystemExit("评分 CSV 需包含 sample_id 列")
    num_cols = scores.select_dtypes(include=[np.number]).columns.tolist()
    score_cols = [c.strip() for c in args.score_cols.split(",") if c.strip()] or num_cols

    internal = {}
    for c in score_cols:
        vals = scores[c].dropna()
        internal[c] = {"n": int(len(vals)), "mean": round(float(vals.mean()), 2),
                       "std": round(float(vals.std()), 2) if len(vals) > 1 else None}

    result = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "n_samples": int(len(scores)), "qda_internal": internal}

    if args.feature and Path(args.feature).exists():
        feat = pd.read_csv(args.feature, encoding="utf-8")
        # ?????? QDA ??????????? merge ?? _x/_y ???
        overlap = [c for c in score_cols if c in feat.columns and c != "sample_id"]
        if overlap:
            feat = feat.drop(columns=overlap)
        merged = scores.merge(feat, on="sample_id", how="inner")
        feat_cols = [c for c in merged.columns if c not in ("sample_id",) and c not in score_cols]
        feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(merged[c])]
        result["pearson"] = pearson_matrix(merged, feat_cols, score_cols)
        result["regression"] = regression(merged, feat_cols, score_cols)
        result["pca"] = pca_loadings(merged, feat_cols)
        if args.grade_col and args.true_col and args.grade_col in merged.columns and args.true_col in merged.columns:
            result["kappa"] = kappa_agreement(merged, args.grade_col, args.true_col)
        print(f"特征 x 评分矩阵/回归/PCA 完成（{len(feat_cols)} 特征 x {len(score_cols)} 评分项）")
    else:
        print("[info] 未提供特征表，仅输出 QDA 内部一致性")

    os.makedirs(Path(args.out).parent, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # CSV：每个评分项的相关性 Top10
    if "pearson" in result:
        rows = []
        for sc, items in result["pearson"].items():
            for f, v in list(items.items())[:10]:
                rows.append({"score": sc, "feature": f, "r": v["r"], "p": v["p"]})
        pd.DataFrame(rows).to_csv(args.out.replace(".json", ".csv"), index=False, encoding="utf-8-sig")
    print("已写出:", args.out)


if __name__ == "__main__":
    main()
