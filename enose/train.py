# -*- coding: utf-8 -*-
"""
电子鼻气味识别建模：多模型对比 + 分类/回归 + 防泄漏评估 + 模型保存

输入：preprocess.py 生成的特征 CSV（每行一个样本）
    分类：含 label 列（如 fresh / mid / rotten）
    回归：含连续目标列（如 hours / minutes，用 --target 指定）
    可选 group 列（采集批次/日期）：--cv group 时按批次分组交叉验证，防数据泄漏

用法：
    # 分类对比（RF/SVM/kNN/LR/PLS-DA，5 折交叉验证）
    python enose/train.py data/features.csv --compare
    # 分类 + 留一交叉验证 + 保存模型
    python enose/train.py data/features.csv --model rf --cv loo --save enose/models/freshness_rf.pkl
    # 分类 + 按批次分组交叉验证（防止同批次数据泄漏）
    python enose/train.py data/features.csv --model plsda --cv group --save enose/models/freshness_plsda.pkl
    # 回归：预测放置小时数（PLSR / RF）
    python enose/train.py data/features.csv --mode regress --model plsr --target hours --cv group --save enose/models/shelf_plsr.pkl
    python enose/train.py data/features.csv --mode regress --model rf --target hours --save enose/models/shelf_rf.pkl

评估指标：
    分类：交叉验证准确率、测试集准确率、Precision/Recall/F1、混淆矩阵
    回归：R^2、RMSE、MAE（LOOCV 或分组 K 折）
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GroupKFold, LeaveOneOut
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             r2_score, mean_squared_error, mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression
from plsda import PLSDA

# ---------------- 模型注册表 ----------------
CLASSIFIERS = {
    "rf": lambda: RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "svm": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "plsda": lambda: make_pipeline(StandardScaler(), PLSDA(n_components=10)),
}

REGRESSORS = {
    "plsr": lambda: PLSRegression(n_components=10, scale=True),
    "rf": lambda: RandomForestRegressor(n_estimators=300, random_state=42),
    "svr": lambda: make_pipeline(StandardScaler(), SVR(C=10, gamma="scale")),
}

CLASSIFY_MODEL_KEYS = {"rf", "svm", "knn", "lr", "plsda"}
REGRESS_MODEL_KEYS = {"plsr", "rf", "svr"}


METADATA_COLS = {"label", "source", "group", "hours", "minutes", "days", "shelf_hours"}

def load(path):
    df = pd.read_csv(path, encoding="utf-8")
    feat_cols = [c for c in df.columns if c not in METADATA_COLS]
    if df.empty:
        sys.exit("数据为空")
    df = df.dropna(subset=feat_cols).copy()
    if df.empty:
        sys.exit("清洗后无数据")
    return df, feat_cols


def make_cv(mode, args_cv, y, groups=None, n_splits=5):
    """按模式返回交叉验证器：loo / group / kfold（回归用 KFold，分类用分层 K 折）"""
    if args_cv == "loo":
        return LeaveOneOut()
    if args_cv == "group":
        if groups is None or len(set(groups)) < 2:
            print("[warn] 没有可用的 group 列，回退到 K 折", file=sys.stderr)
        else:
            n_groups = len(set(groups))
            return GroupKFold(n_splits=min(n_splits, n_groups))
    if mode == "regress":
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def compare_classify(df, feat_cols, args):
    X, y = df[feat_cols], df["label"].astype(str)
    groups = df["group"] if "group" in df.columns else None
    print("=" * 72)
    print("分类多模型对比")
    print(f"样本 {len(X)} | 类别 {sorted(set(y))} | 特征 {len(feat_cols)} | CV: {args.cv}")
    print("-" * 72)
    header = f"{'模型':<8}{'交叉验证准确率':>16}{'测试集准确率':>14}"
    print(header)
    rows = []
    for name in sorted(CLASSIFIERS):
        clf = CLASSIFIERS[name]()
        cv = make_cv("classify", args.cv, y, groups)
        if isinstance(cv, GroupKFold):
            scores = cross_val_score(clf, X, y, cv=cv, groups=groups)
        else:
            scores = cross_val_score(clf, X, y, cv=cv)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        clf.fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        rows.append((name, scores.mean(), acc))
        print(f"{name:<8}{scores.mean() * 100:>14.2f}%{acc * 100:>12.2f}%")
    print("=" * 72)
    best = max(rows, key=lambda r: r[1])
    print(f"最优: {best[0]}（交叉验证 {best[1] * 100:.2f}%）")


def compare_regress(df, feat_cols, args):
    target = args.target
    if target not in df.columns:
        sys.exit(f"目标列 {target} 不存在；可用的连续列: {[c for c in df.columns if c not in feat_cols]}")
    X, y = df[feat_cols], df[target].astype(float)
    groups = df["group"] if "group" in df.columns else None
    print("=" * 72)
    print("回归多模型对比（预测目标: %s）" % target)
    print(f"样本 {len(X)} | 特征 {len(feat_cols)} | CV: {args.cv}")
    print("-" * 72)
    print(f"{'模型':<8}{'R^2':>10}{'RMSE':>12}{'MAE':>12}")
    rows = []
    for name in sorted(REGRESSORS):
        reg = REGRESSORS[name]()
        cv = make_cv("regress", args.cv, y, groups)
        if isinstance(cv, GroupKFold):
            scores = cross_val_score(reg, X, y, cv=cv, groups=groups, scoring="r2")
        else:
            scores = cross_val_score(reg, X, y, cv=cv, scoring="r2")
        # 汇总 RMSE/MAE
        rmses, maes = [], []
        for tr_idx, te_idx in cv.split(X, y, groups):
            reg2 = REGRESSORS[name]()
            reg2.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            p = reg2.predict(X.iloc[te_idx])
            rmses.append(mean_squared_error(y.iloc[te_idx], p) ** 0.5)
            maes.append(mean_absolute_error(y.iloc[te_idx], p))
        rows.append((name, float(np.mean(scores)), float(np.mean(rmses)), float(np.mean(maes))))
        print(f"{name:<8}{np.mean(scores):>10.3f}{np.mean(rmses):>12.3f}{np.mean(maes):>12.3f}")
    print("=" * 72)
    best = max(rows, key=lambda r: r[1])
    print(f"最优: {best[0]}（R^2 = {best[1]:.3f}，RMSE = {best[2]:.3f}，MAE = {best[3]:.3f}）")


def single_classify(df, feat_cols, args):
    X, y = df[feat_cols], df["label"].astype(str)
    groups = df["group"] if "group" in df.columns else None
    clf = CLASSIFIERS[args.model]()
    cv = make_cv("classify", args.cv, y, groups)
    if isinstance(cv, GroupKFold):
        scores = cross_val_score(clf, X, y, cv=cv, groups=groups)
    else:
        scores = cross_val_score(clf, X, y, cv=cv)
    print(f"模型: {args.model} | CV: {args.cv} | 交叉验证准确率: {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print(f"测试集准确率: {accuracy_score(yte, pred) * 100:.2f}%")
    labels = sorted(set(y))
    print("\n混淆矩阵（行=真实，列=预测）：")
    print(pd.DataFrame(confusion_matrix(yte, pred, labels=labels), index=labels, columns=labels).to_string())
    print("\n分类报告：")
    print(classification_report(yte, pred, zero_division=0))
    if args.model == "rf" and hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
        print("\nTop-10 特征重要性：")
        print(imp.head(10).to_string())
    if args.save:
        clf.fit(X, y)  # 全量重训再保存
        joblib.dump({"model": clf, "feat_cols": feat_cols, "task": "classify",
                     "model_name": args.model, "labels": labels}, args.save)
        print(f"模型已保存: {args.save}")


def single_regress(df, feat_cols, args):
    target = args.target
    if target not in df.columns:
        sys.exit(f"目标列 {target} 不存在")
    X, y = df[feat_cols], df[target].astype(float)
    groups = df["group"] if "group" in df.columns else None
    reg = REGRESSORS[args.model]()
    cv = make_cv("regress", args.cv, y, groups)
    if isinstance(cv, GroupKFold):
        scores = cross_val_score(reg, X, y, cv=cv, groups=groups, scoring="r2")
    else:
        scores = cross_val_score(reg, X, y, cv=cv, scoring="r2")
    print(f"模型: {args.model} | 目标: {target} | CV: {args.cv} | R^2: {scores.mean():.3f} (+/- {scores.std():.3f})")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    reg.fit(Xtr, ytr)
    p = reg.predict(Xte)
    print(f"测试集: R^2 = {r2_score(yte, p):.3f} | RMSE = {mean_squared_error(yte, p) ** 0.5:.3f} | MAE = {mean_absolute_error(yte, p):.3f}")
    if args.model == "rf" and hasattr(reg, "feature_importances_"):
        imp = pd.Series(reg.feature_importances_, index=feat_cols).sort_values(ascending=False)
        print("\nTop-10 特征重要性：")
        print(imp.head(10).to_string())
    if args.save:
        reg.fit(X, y)
        joblib.dump({"model": reg, "feat_cols": feat_cols, "task": "regress",
                     "model_name": args.model, "target": target}, args.save)
        print(f"模型已保存: {args.save}")


def main():
    ap = argparse.ArgumentParser(description="电子鼻建模（分类/回归，防泄漏评估，模型保存）")
    ap.add_argument("csv", help="特征 CSV（preprocess.py 输出）")
    ap.add_argument("--compare", action="store_true", help="多模型对比模式")
    ap.add_argument("--mode", choices=["classify", "regress"], default="classify")
    ap.add_argument("--model", default="rf", help="classify: rf/svm/knn/lr/plsda；regress: plsr/rf/svr")
    ap.add_argument("--target", default="hours", help="回归目标列名（默认 hours）")
    ap.add_argument("--cv", choices=["kfold", "group", "loo"], default="group",
                    help="交叉验证方式：group=按批次分组（防泄漏，推荐）/ loo=留一 / kfold=K 折")
    ap.add_argument("--save", default="", help="保存模型路径（joblib bundle）")
    args = ap.parse_args()

    df, feat_cols = load(args.csv)
    if args.mode == "regress" and args.target in df.columns:
        # 回归目标列不得进入特征（防目标泄漏）
        feat_cols = [c for c in feat_cols if c != args.target]
    if args.mode == "classify" and "label" not in df.columns:
        sys.exit("分类模式需要 label 列")
    if args.cv == "group" and "group" not in df.columns:
        print("[提示] CSV 没有 group 列，将使用普通 K 折；建议在特征表里加 group（批次/日期）列以分组防泄漏", file=sys.stderr)
        args.cv = "kfold"

    if args.compare:
        if args.mode == "classify":
            compare_classify(df, feat_cols, args)
        else:
            compare_regress(df, feat_cols, args)
    elif args.mode == "classify":
        single_classify(df, feat_cols, args)
    else:
        single_regress(df, feat_cols, args)


if __name__ == "__main__":
    main()