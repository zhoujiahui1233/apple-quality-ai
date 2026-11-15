# -*- coding: utf-8 -*-
"""在清洗后的无泄漏数据集上重训 SVM / RF，并做超参数搜索。

要点
  * 特征提取严格对齐线上：先 preprocess 到 500 宽，再 extract_features_v2
    （原 train_ml_v3.py 直接对整图提特征，与线上不一致）。
  * 交叉验证按「同源簇」分组 (GroupKFold)，避免近重复图跨折泄漏。
  * 超参数用网格搜索选，按平衡准确率挑最优。
  * 新模型写到 *_clean.* 文件名，不覆盖现有模型。
  * 同一份干净验证集上对比「现役模型 vs 新模型」。

用法:
  python scripts/retrain_ml_clean.py [--jobs 8] [--search]
"""
import argparse
import csv
import hashlib
import os
import sys
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

import apple_defect_detector as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "clean_dataset_v1")
MANIFEST = os.path.join(CLEAN, "manifest.csv")
FEAT_CACHE = os.path.join(CLEAN, "features.npz")
# joblib 的默认临时目录在用户目录下（可能含中文），loky 的 resource_tracker
# 会用 ASCII 编码路径而崩溃；显式指到全 ASCII 目录规避。
JOBLIB_TMP = os.path.join(ROOT, "_joblib_tmp")
os.makedirs(JOBLIB_TMP, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = JOBLIB_TMP


def load_manifest():
    rows = []
    with open(MANIFEST, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def extract_one(path):
    """与线上一致：先 preprocess 到统一宽度，再提特征。"""
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        img_p, _ = D.preprocess(img)
        return D.extract_features_v2(img_p)
    except Exception:
        return None


def build_features(rows, jobs):
    # 清单指纹：文件名+切分+分组都要一致才能复用缓存，
    # 否则「切分变了但文件名没变」会被误判为命中，导致评估口径错乱。
    fp = hashlib.md5("|".join(
        "%s:%s:%s" % (r["filename"], r["split"], r["cluster"]) for r in rows
    ).encode("utf-8")).hexdigest()
    if os.path.exists(FEAT_CACHE):
        z = np.load(FEAT_CACHE, allow_pickle=True)
        if "fp" in z and str(z["fp"]) == fp:
            print("复用特征缓存:", FEAT_CACHE)
            return z["X"], z["y"], z["g"], z["split"]
        print("特征缓存与当前清单不一致(切分已变)，重新提取")
    from joblib import Parallel, delayed
    paths = [os.path.join(ROOT, r["path"]) for r in rows]
    print("提取特征 (%d 张, jobs=%d)... 这一步较慢" % (len(paths), jobs))
    t0 = time.time()
    feats = Parallel(n_jobs=jobs, verbose=5, temp_folder=os.path.join(ROOT, "_joblib_tmp"))(
        delayed(extract_one)(p) for p in paths)
    print("特征提取完成 %.0fs" % (time.time() - t0))

    X, y, g, split = [], [], [], []
    bad = 0
    for r, fe in zip(rows, feats):
        if fe is None:
            bad += 1
            continue
        X.append(fe)
        y.append(0 if r["class"] == "good" else 1)
        g.append(int(r["cluster"]))
        split.append(r["split"])
    print("有效特征 %d / %d (跳过 %d，多为未检出苹果)" % (len(X), len(rows), bad))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    g = np.asarray(g, dtype=np.int64)
    split = np.asarray(split)
    np.savez_compressed(FEAT_CACHE, X=X, y=y, g=g, split=split, fp=fp)
    print("特征已缓存:", FEAT_CACHE)
    return X, y, g, split


def bal_acc(yt, yp):
    a = np.mean(yp[yt == 0] == 0) if np.any(yt == 0) else 0.0
    b = np.mean(yp[yt == 1] == 1) if np.any(yt == 1) else 0.0
    return (a + b) / 2.0 * 100.0


def eval_arrays(yt, yp):
    acc = float(np.mean(yt == yp)) * 100.0
    return acc, bal_acc(yt, yp)


def _svm_cv_one(Xs, y, folds_idx, C, gm):
    """单个 (C, gamma) 组合的分组 CV 得分。cv2 SVM 单线程，靠多进程并行。"""
    cv2.setNumThreads(1)
    scores = []
    for tr, te in folds_idx:
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_RBF)
        svm.setGamma(gm)
        svm.setC(C)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                             100000, 1e-7))
        svm.train(Xs[tr].astype(np.float32), cv2.ml.ROW_SAMPLE, y[tr].astype(np.int32))
        _, p = svm.predict(Xs[te].astype(np.float32))
        scores.append(bal_acc(y[te], p.flatten().astype(int)))
    return C, gm, float(np.mean(scores))


def train_svm_cv(Xs, y, groups, folds=4, jobs=8):
    """OpenCV SVM + 网格搜索（多进程），按分组 CV 的平衡准确率选参。"""
    from sklearn.model_selection import GroupKFold
    from joblib import Parallel, delayed
    gkf = list(GroupKFold(n_splits=folds).split(Xs, y, groups))
    grid = [(C, gm) for C in (10.0, 50.0, 100.0) for gm in (0.005, 0.01, 0.02)]
    print("    %d 个组合 x %d 折，%d 进程并行" % (len(grid), folds, jobs), flush=True)
    res = Parallel(n_jobs=jobs, verbose=0, temp_folder=JOBLIB_TMP)(
        delayed(_svm_cv_one)(Xs, y, gkf, C, gm) for C, gm in grid)
    best = None
    for C, gm, m in sorted(res, key=lambda t: -t[2]):
        print("    SVM C=%-5g gamma=%-6g bal_acc=%.2f%%" % (C, gm, m), flush=True)
        if best is None or m > best[0]:
            best = (m, C, gm)
    print("  最优 SVM: C=%g gamma=%g (CV bal_acc=%.2f%%)" % (best[1], best[2], best[0]))
    return best[1], best[2], best[0]


def train_rf_cv(X, y, groups, folds=4):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    gkf = list(GroupKFold(n_splits=folds).split(X, y, groups))
    grid = [(400, None, 1), (700, None, 2), (700, 30, 1)]
    best = None
    for n, md, msl in grid:
        scores = []
        for tr, te in gkf:
            rf = RandomForestClassifier(n_estimators=n, max_depth=md,
                                        min_samples_leaf=msl, random_state=42,
                                        n_jobs=-1, class_weight="balanced")
            rf.fit(X[tr], y[tr])
            scores.append(bal_acc(y[te], rf.predict(X[te])))
        m = float(np.mean(scores))
        print("    RF n=%-4d depth=%-5s leaf=%-2d bal_acc=%.2f%%" % (n, md, msl, m), flush=True)
        if best is None or m > best[0]:
            best = (m, n, md, msl)
    print("  最优 RF: n=%d depth=%s leaf=%d (CV bal_acc=%.2f%%)"
          % (best[1], best[2], best[3], best[0]))
    return best[1], best[2], best[3], best[0]


def predict_old(X, split, y):
    """把现役模型在同一份干净 val 上跑一遍（基线）。"""
    from sklearn.preprocessing import StandardScaler
    out = {}
    idx = np.where(split == "val")[0]
    Xv, yv = X[idx], y[idx]
    try:
        svm = cv2.ml.SVM_load(os.path.join(ROOT, "apple_svm_model.dat"))
        with open(os.path.join(ROOT, "apple_scaler.pkl"), "rb") as f:
            sc = pickle.load(f)
        _, p = svm.predict(sc.transform(Xv).astype(np.float32))
        out["SVM(现役)"] = eval_arrays(yv, p.flatten().astype(int))
    except Exception as e:
        print("  现役 SVM 评估失败:", e)
    try:
        with open(os.path.join(ROOT, "apple_rf_model.pkl"), "rb") as f:
            rf = pickle.load(f)
        out["RF(现役)"] = eval_arrays(yv, rf.predict(Xv))
    except Exception as e:
        print("  现役 RF 评估失败:", e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--search", action="store_true", help="执行网格搜索(较慢)")
    args = ap.parse_args()

    from sklearn.preprocessing import StandardScaler
    rows = load_manifest()
    X, y, g, split = build_features(rows, args.jobs)
    print("特征维度:", X.shape, "| train %d, val %d"
          % (int((split == "train").sum()), int((split == "val").sum())))

    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    Xtr, ytr, gtr = X[tr], y[tr], g[tr]
    Xva, yva = X[va], y[va]

    print("\n[基线] 现役模型在同一份干净 val 上:")
    base = predict_old(X, split, y)
    for k, (a, b) in base.items():
        print("  %-10s acc=%.2f%%  bal_acc=%.2f%%" % (k, a, b))

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)

    if args.search:
        print("\n[1/2] SVM 超参数搜索...")
        C, gm, cv_svm = train_svm_cv(Xtr_s, ytr, gtr, jobs=args.jobs)
        print("\n[2/2] RF 超参数搜索...")
        n, md, msl, cv_rf = train_rf_cv(Xtr, ytr, gtr)
    else:
        C, gm, cv_svm = 10.0, 0.01, float("nan")
        n, md, msl, cv_rf = 600, None, 1, float("nan")

    print("\n训练最终模型...")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setGamma(gm)
    svm.setC(C)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100000, 1e-7))
    svm.train(Xtr_s.astype(np.float32), cv2.ml.ROW_SAMPLE, ytr.astype(np.int32))
    _, p = svm.predict(Xva_s.astype(np.float32))
    svm_res = eval_arrays(yva, p.flatten().astype(int))

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=n, max_depth=md, min_samples_leaf=msl,
                                random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(Xtr, ytr)
    rf_res = eval_arrays(yva, rf.predict(Xva))

    print("\n===== 干净验证集结果 (train 按簇切分, val 完全未见过的簇) =====")
    print("%-12s %10s %12s" % ("model", "acc", "bal_acc"))
    for k, (a, b) in base.items():
        print("%-12s %9.2f%% %11.2f%%" % (k, a, b))
    print("%-12s %9.2f%% %11.2f%%" % ("SVM(新)", svm_res[0], svm_res[1]))
    print("%-12s %9.2f%% %11.2f%%" % ("RF(新)", rf_res[0], rf_res[1]))

    svm.save(os.path.join(ROOT, "apple_svm_model_clean.dat"))
    with open(os.path.join(ROOT, "apple_rf_model_clean.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(ROOT, "apple_scaler_clean.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("\n新模型已保存（未覆盖现役模型）:")
    print("  apple_svm_model_clean.dat / apple_rf_model_clean.pkl / apple_scaler_clean.pkl")
    print("  切换方式：改 config.MODEL_PATHS 指向 *_clean.* 即可")


if __name__ == "__main__":
    main()
