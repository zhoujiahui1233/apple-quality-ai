# -*- coding: utf-8 -*-
"""
train_ml_v3.py - 论文吸收版特征工程重训（170维 -> 185维）
新增特征：着色率/红度(4) + GLCM纹理(8) + 果径果形(3)，
参考《刘佳浩》《贾馥谦》《赵树昌》《仇德凯》。
先备份旧模型，再训练 scaler/SVM/RF 并对比 v2 准确率。
"""
import os, sys, time, shutil, io
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\apple")

import numpy as np
import cv2
import pickle
import glob
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import apple_defect_detector as D

DATA = r"E:\apple\kaggle_data\extracted"
OUT = r"E:\apple"

def extract_one(path):
    img = cv2.imread(path)
    if img is None:
        return None
    try:
        return D.extract_features_v2(img)
    except Exception:
        return None

def load_split(sub):
    fresh = sorted(glob.glob(os.path.join(DATA, sub, "freshapples", "*.png"))
                   + glob.glob(os.path.join(DATA, sub, "freshapples", "*.jpg"))
                   + glob.glob(os.path.join(DATA, sub, "freshapples", "*.jpeg")))
    rotten = sorted(glob.glob(os.path.join(DATA, sub, "rottenapples", "*.png"))
                    + glob.glob(os.path.join(DATA, sub, "rottenapples", "*.jpg"))
                    + glob.glob(os.path.join(DATA, sub, "rottenapples", "*.jpeg")))
    paths = fresh + rotten
    labels = np.array([0]*len(fresh) + [1]*len(rotten))
    t0 = time.time()
    os.makedirs(r"E:\apple\_joblib_tmp", exist_ok=True)
    feats = Parallel(n_jobs=4, verbose=5, temp_folder=r"E:\apple\_joblib_tmp")(delayed(extract_one)(p) for p in paths)
    feats = np.array([f for f in feats if f is not None], dtype=np.float32)
    print(f"[{sub}] {len(paths)} imgs -> {len(feats)} feats, {time.time()-t0:.0f}s, dim={feats.shape[1]}")
    return feats, labels[:len(feats)]

if __name__ == "__main__":
    # 备份旧模型
    for name in ["apple_scaler.pkl", "apple_svm_model.dat", "apple_rf_model.pkl"]:
        src = os.path.join(OUT, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT, name.replace(".pkl", "_v2.pkl").replace(".dat", "_v2.dat")))
            print("backup ->", name.replace(".pkl", "_v2.pkl").replace(".dat", "_v2.dat"))

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    print("train:", X_train.shape, "test:", X_test.shape)
    if X_train.shape[1] != 185:
        print("WARN: feature dim =", X_train.shape[1], "(expect 185)")

    # 平衡采样（与 v2 一致）
    n0 = int(np.sum(y_train == 0)); n1 = int(np.sum(y_train == 1))
    print(f"train fresh={n0} rotten={n1}")
    rng = np.random.RandomState(42)
    if n0 != n1:
        m = min(n0, n1)
        idx0 = rng.choice(np.where(y_train == 0)[0], m, replace=False)
        idx1 = rng.choice(np.where(y_train == 1)[0], m, replace=False)
        idx = np.concatenate([idx0, idx1]); rng.shuffle(idx)
        X_bal, y_bal = X_train[idx], y_train[idx]
    else:
        X_bal, y_bal = X_train, y_train

    print("\n[1/3] StandardScaler...")
    scaler = StandardScaler().fit(X_bal)
    X_bal_s = scaler.transform(X_bal)
    X_test_s = scaler.transform(X_test)

    print("[2/3] SVM (OpenCV RBF, C=50, gamma=0.01)...")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC); svm.setKernel(cv2.ml.SVM_RBF)
    svm.setGamma(0.01); svm.setC(50.0)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100000, 1e-7))
    svm.train(X_bal_s.astype(np.float32), cv2.ml.ROW_SAMPLE, y_bal.astype(np.int32))
    _, pred = svm.predict(X_test_s.astype(np.float32))
    pred = pred.flatten()
    svm_acc = accuracy_score(y_test, pred) * 100
    svm_fa = accuracy_score(y_test[y_test == 0], pred[y_test == 0]) * 100
    svm_ra = accuracy_score(y_test[y_test == 1], pred[y_test == 1]) * 100
    print(f"  SVM v3: acc={svm_acc:.2f}% fresh={svm_fa:.2f}% rotten={svm_ra:.2f}%")
    svm.save(os.path.join(OUT, "apple_svm_model.dat"))

    print("[3/3] RandomForest...")
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_bal, y_bal)
    pred_rf = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, pred_rf) * 100
    rf_fa = accuracy_score(y_test[y_test == 0], pred_rf[y_test == 0]) * 100
    rf_ra = accuracy_score(y_test[y_test == 1], pred_rf[y_test == 1]) * 100
    print(f"  RF  v3: acc={rf_acc:.2f}% fresh={rf_fa:.2f}% rotten={rf_ra:.2f}%")

    with open(os.path.join(OUT, "apple_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUT, "apple_rf_model.pkl"), "wb") as f:
        pickle.dump(rf, f)

    print("\n===== 对比 v2 (170维) =====")
    print("v2: SVM 99.93% / RF 99.80%")
    print(f"v3: SVM {svm_acc:.2f}% / RF {rf_acc:.2f}%")
    print("完成：新模型已保存（旧模型已备份为 *_v2.*）")
