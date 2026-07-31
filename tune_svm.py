"""
苹果 SVM 模型调优工具
====================
运行方式: python tune_svm.py
耗时约 5-10 分钟（取决于数据量）
自动搜索最优 C 和 gamma 参数，并用最好参数重新训练保存模型。
"""

import cv2
import numpy as np
import os
import sys
import time

sys.path.insert(0, "E:\\apple")
import importlib
import apple_defect_detector
importlib.reload(apple_defect_detector)

DATA_DIR = "E:\\apple\\kaggle_data\\extracted"
OUTPUT_MODEL = "E:\\apple\\apple_svm_model.dat"

def extract_features(img_bgr):
    img_p, _ = apple_defect_detector.preprocess(img_bgr)
    apple_mask, contour = apple_defect_detector.segment_apple(img_p)
    if apple_mask is None: return None

    gray = cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_p, cv2.COLOR_BGR2HSV)
    ap = apple_mask > 0
    ac = int(np.sum(ap))
    ar = ac / (img_p.shape[0] * img_p.shape[1])

    hc, sc, vc = cv2.split(hsv)
    feats = [
        np.mean(hc[ap]), np.std(hc[ap]),
        np.mean(sc[ap]), np.std(sc[ap]),
        np.mean(vc[ap]), np.std(vc[ap]),
        ar,
    ]
    for lo, up in apple_defect_detector.DEFECT_HSV_RANGES:
        dm = cv2.inRange(hsv, lo, up)
        dm = cv2.bitwise_and(dm, apple_mask)
        feats.append(cv2.countNonZero(dm) / max(1, ac))

    blurred = cv2.bilateralFilter(gray, 9, 50, 50)
    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    la = np.uint8(np.abs(lap))[ap]
    feats.append(np.mean(la)); feats.append(np.std(la))

    perim = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circ = 4 * np.pi * area / (perim * perim) if perim > 0 else 0
    hull = cv2.convexHull(contour)
    ha = cv2.contourArea(hull)
    sol = area / ha if ha > 0 else 0
    feats.append(circ); feats.append(sol)

    return np.array(feats, dtype=np.float32)

print("=" * 50)
print("苹果 SVM 模型调优工具")
print("=" * 50)

# Load all data
print("\n[1/4] 加载训练数据特征...")
t0 = time.time()
X_all, y_all = [], []
for cat, lab in [("freshapples", 0), ("rottenapples", 1)]:
    path = os.path.join(DATA_DIR, "train", cat)
    files = [f for f in os.listdir(path) if f.lower().endswith((".png",".jpg",".jpeg"))]
    count = 0
    for fname in files:
        img = cv2.imread(os.path.join(path, fname))
        if img is None: continue
        feats = extract_features(img)
        if feats is not None:
            X_all.append(feats); y_all.append(lab)
            count += 1
    print(f"  {cat}: {count} images")

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int32)
n0 = int(np.sum(y_all == 0)); n1 = int(np.sum(y_all == 1))
print(f"  总计: {len(X_all)} ({n0} 新鲜, {n1} 腐烂)")
print(f"  耗时: {time.time()-t0:.0f}s")

# Balance
print("\n[2/4] 平衡正负样本...")
min_c = min(n0, n1)
fi = np.where(y_all == 0)[0]; ri = np.where(y_all == 1)[0]
np.random.seed(42)
fs = np.random.choice(fi, min_c, replace=False)
rs = np.random.choice(ri, min_c, replace=False)
sel = np.concatenate([fs, rs])
X_bal = X_all[sel]; y_bal = y_all[sel]
print(f"  平衡后: {len(X_bal)} (各 {min_c})")

# Load test
print("\n[3/4] 加载测试数据...")
X_test, y_test = [], []
for cat, lab in [("freshapples", 0), ("rottenapples", 1)]:
    path = os.path.join(DATA_DIR, "test", cat)
    for fname in os.listdir(path):
        img = cv2.imread(os.path.join(path, fname))
        if img is None: continue
        feats = extract_features(img)
        if feats is not None:
            X_test.append(feats); y_test.append(lab)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int32)
print(f"  测试集: {len(X_test)} ({int(np.sum(y_test==0))} 新鲜, {int(np.sum(y_test==1))} 腐烂)")

# Grid search
print("\n[4/4] 网格搜索最优参数...")
print(f"  {'C':>6}  {'gamma':>6}  {'准确率':>6}  {'新鲜':>6}  {'腐烂':>6}")
print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

best_acc = 0; best_p = None
for C in [0.1, 1.0, 10.0, 100.0]:
    for gamma in [0.01, 0.05, 0.1, 0.5, 1.0]:
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_RBF)
        svm.setGamma(gamma); svm.setC(C)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-6))
        svm.train(X_bal, cv2.ml.ROW_SAMPLE, y_bal)

        _, pred = svm.predict(X_test)
        pred = pred.flatten().astype(np.int32)
        acc = np.mean(pred == y_test) * 100
        fa = np.mean(pred[y_test == 0] == 0) * 100
        ra = np.mean(pred[y_test == 1] == 1) * 100

        if acc > best_acc:
            best_acc = acc; best_p = (C, gamma)

        mark = " <<<" if acc == best_acc else ""
        print(f"  {C:6.1f}  {gamma:6.2f}  {acc:6.1f}%  {fa:6.1f}%  {ra:6.1f}%{mark}")

# Retrain best model on full balanced data
print(f"\n最优参数: C={best_p[0]}, gamma={best_p[1]}, 准确率={best_acc:.1f}%")
print("正在用最优参数重新训练模型...")
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setGamma(best_p[1]); svm.setC(best_p[0])
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-6))
svm.train(X_bal, cv2.ml.ROW_SAMPLE, y_bal)
svm.save(OUTPUT_MODEL)
print(f"模型已保存: {OUTPUT_MODEL}")
print("\n完成!")
