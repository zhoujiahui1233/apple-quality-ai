"""
ML Model v2 - 升级版特征工程 + 重新训练 SVM & RandomForest
"""
import sys, os, cv2, numpy as np, pickle, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "E:\\apple")
import importlib, apple_defect_detector
importlib.reload(apple_defect_detector)

DATA_DIR = "E:\\apple\\kaggle_data\\extracted"
OUTPUT_DIR = "E:\\apple"
SCALER_PATH = os.path.join(OUTPUT_DIR, "apple_scaler.pkl")

def calc_hist(region, mask, channel, bins, range_val):
    hist = cv2.calcHist([region], [channel], mask, [bins], range_val).flatten()
    hist = hist / (hist.sum() + 1e-8)
    return hist

def lbp_features(gray_region):
    """Simple LBP-like texture features"""
    h, w = gray_region.shape
    if h < 3 or w < 3: return [0, 0, 0]
    center = gray_region[1:-1, 1:-1]
    neighbors = [
        gray_region[:-2, :-2], gray_region[:-2, 1:-1], gray_region[:-2, 2:],
        gray_region[1:-1, 2:],  gray_region[2:, 2:],    gray_region[2:, 1:-1],
        gray_region[2:, :-2],   gray_region[1:-1, :-2]
    ]
    lbp = sum((n > center).astype(np.uint8) * (1 << i) for i, n in enumerate(neighbors))
    hist = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [32], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    return hist.tolist()

def extract_features_v2(img_bgr):
    """新版特征提取：~140维特征"""
    img_p, _ = apple_defect_detector.preprocess(img_bgr)
    apple_mask, contour = apple_defect_detector.segment_apple(img_p)
    if apple_mask is None: return None
    
    h, w = img_p.shape[:2]
    gray = cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_p, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_p, cv2.COLOR_BGR2LAB)
    
    ap = apple_mask > 0
    ac = int(np.sum(ap))
    ar = ac / (h * w)
    
    feats = []
    
    # ============================
    # 1. COLOR HISTOGRAMS (96维)
    # ============================
    hc, sc, vc = cv2.split(hsv)
    for channel, bins, rng in [(hc, 32, [0, 180]), (sc, 32, [0, 256]), (vc, 32, [0, 256])]:
        hist = calc_hist(channel, apple_mask, 0, bins, rng)
        feats.extend(hist.tolist())
    
    # ============================
    # 2. COLOR MOMENTS (9维)
    # ============================
    for ch in [hc[ap], sc[ap], vc[ap]]:
        if len(ch) == 0:
            feats.extend([0, 0, 0])
        else:
            mean = float(np.mean(ch))
            std = float(np.std(ch))
            skew = float(np.mean(((ch - mean) / (std + 1e-8)) ** 3))
            feats.extend([mean, std, skew])
    
    # ============================
    # 3. LAB COLOR SPACE (6维)
    # ============================
    lc, ac_ch, bc = cv2.split(lab)
    for ch in [lc[ap], ac_ch[ap], bc[ap]]:
        if len(ch) > 0:
            feats.extend([float(np.mean(ch)), float(np.std(ch))])
        else:
            feats.extend([0, 0])
    
    # ============================
    # 4. DEFECT HSV RATIOS (6维)
    # ============================
    for lo, up in apple_defect_detector.DEFECT_HSV_RANGES:
        dm = cv2.inRange(hsv, lo, up)
        dm = cv2.bitwise_and(dm, apple_mask)
        feats.append(float(cv2.countNonZero(dm)) / max(1, ac))
    
    # ============================
    # 5. EXTRA DEFECT RANGES (6维)
    # ============================
    extra_ranges = [
        (np.array([0, 0, 80]), np.array([180, 50, 200])),   # desaturated areas
        (np.array([0, 50, 150]), np.array([180, 255, 255])), # bright saturated
        (np.array([100, 30, 0]), np.array([130, 255, 100])),  # purple/bruise
        (np.array([40, 50, 30]), np.array([80, 255, 150])),   # green/unripe
        (np.array([0, 0, 180]), np.array([180, 30, 255])),    # white bright spots
        (np.array([20, 50, 100]), np.array([40, 200, 200])),  # yellow/brown
    ]
    for lo, up in extra_ranges:
        dm = cv2.inRange(hsv, lo, up)
        dm = cv2.bitwise_and(dm, apple_mask)
        feats.append(float(cv2.countNonZero(dm)) / max(1, ac))
    
    # ============================
    # 6. TEXTURE FEATURES (12维)
    # ============================
    # Multi-scale Laplacian
    for ksize in [3, 7, 15]:
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        lap = cv2.Laplacian(blurred, cv2.CV_64F)
        la = np.uint8(np.abs(lap))[ap]
        if len(la) > 0:
            feats.extend([float(np.mean(la)), float(np.std(la)), float(np.percentile(la, 90))])
        else:
            feats.extend([0, 0, 0])
    
    # Sobel edge density
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_density = float(np.mean(sobel_mag[ap] > 50)) if np.sum(ap) > 0 else 0
    feats.append(edge_density)
    
    # OTSU adaptive threshold defect ratio
    apple_region = cv2.bitwise_and(gray, gray, mask=apple_mask)
    if np.sum(apple_mask > 0) > 0:
        _, otsu = cv2.threshold(apple_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_defect = 1.0 - float(np.sum(otsu[ap] > 0)) / max(1, ac)
        feats.append(otsu_defect)
    else:
        feats.append(0)
    
    # Local Binary Pattern features
    apple_gray_full = cv2.bitwise_and(gray, gray, mask=apple_mask)
    if np.sum(ap) > 100:
        lbp_feats = lbp_features(apple_gray_full)
        feats.extend(lbp_feats)
    else:
        feats.extend([0]*3)
    
    # ============================
    # 7. SHAPE FEATURES (3维)
    # ============================
    perim = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circ = 4 * np.pi * area / (perim * perim) if perim > 0 else 0
    hull = cv2.convexHull(contour)
    ha = cv2.contourArea(hull)
    sol = area / ha if ha > 0 else 0
    x, y, w_box, h_box = cv2.boundingRect(contour)
    aspect = float(w_box) / max(h_box, 1)
    feats.extend([float(circ), float(sol), float(aspect)])
    
    # ============================
    # 8. AREA RATIO (1维)
    # ============================
    feats.append(float(ar))
    
    return np.array(feats, dtype=np.float32)

# ============ MAIN ============
print("=" * 55)
print("  ML Model v2 - 升级特征 + 重新训练")
print("=" * 55)

print("\n[1/5] 提取训练特征...")
t0 = time.time()
X_train, y_train = [], []
for cat, lab in [("freshapples", 0), ("rottenapples", 1)]:
    path = os.path.join(DATA_DIR, "train", cat)
    files = [f for f in os.listdir(path) if f.lower().endswith((".png",".jpg",".jpeg"))]
    count = 0; errs = 0
    for fname in files:
        img = cv2.imread(os.path.join(path, fname))
        if img is None: errs += 1; continue
        feats = extract_features_v2(img)
        if feats is not None:
            X_train.append(feats); y_train.append(lab); count += 1
    print(f"  {cat}: {count} OK, {errs} errors")

X_train = np.array(X_train); y_train = np.array(y_train)
n0 = int(np.sum(y_train==0)); n1 = int(np.sum(y_train==1))
print(f"  总计: {len(X_train)} 样本 ({n0} fresh, {n1} rotten)")
print(f"  特征维度: {X_train.shape[1]}")

# Balance
print("\n[2/5] 平衡样本...")
min_c = min(n0, n1)
fi = np.where(y_train==0)[0]; ri = np.where(y_train==1)[0]
np.random.seed(42)
fs = np.random.choice(fi, min_c, replace=False)
rs = np.random.choice(ri, min_c, replace=False)
sel = np.concatenate([fs, rs])
X_bal, y_bal = X_train[sel], y_train[sel]
print(f"  平衡后: {len(X_bal)} 样本")

# Extract test features
print("\n[3/5] 提取测试特征...")
X_test, y_test = [], []
for cat, lab in [("freshapples",0), ("rottenapples",1)]:
    path = os.path.join(DATA_DIR, "test", cat)
    for fname in os.listdir(path):
        img = cv2.imread(os.path.join(path, fname))
        if img is None: continue
        feats = extract_features_v2(img)
        if feats is not None:
            X_test.append(feats); y_test.append(lab)
X_test = np.array(X_test); y_test = np.array(y_test)
print(f"  测试集: {len(X_test)} 样本")

# StandardScaler (important for SVM)
print("\n[4/5] 训练 StandardScaler...")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_bal_scaled = scaler.fit_transform(X_bal)
X_test_scaled = scaler.transform(X_test)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print(f"  Scaler saved to {SCALER_PATH}")

# ============ SVM ============
print("\n" + "=" * 55)
print("  训练 SVM (OpenCV)")
print("=" * 55)
t0 = time.time()
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setGamma(0.01)
svm.setC(50.0)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100000, 1e-7))
svm.train(X_bal_scaled, cv2.ml.ROW_SAMPLE, y_bal)
train_t = time.time() - t0

_, pred = svm.predict(X_test_scaled)
pred = pred.flatten().astype(np.int32)
svm_acc = np.mean(pred == y_test) * 100
svm_fa = np.mean(pred[y_test==0] == 0) * 100
svm_ra = np.mean(pred[y_test==1] == 1) * 100
print(f"  训练耗时: {train_t:.1f}s")
print(f"  总准确率: {svm_acc:.2f}%")
print(f"  新鲜识别: {svm_fa:.2f}%")
print(f"  腐烂识别: {svm_ra:.2f}%")

svm_path = os.path.join(OUTPUT_DIR, "apple_svm_model.dat")
svm.save(svm_path)
print(f"  已保存: {svm_path}")

# ============ RandomForest ============
print("\n" + "=" * 55)
print("  训练 RandomForest (sklearn)")
print("=" * 55)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=500, max_depth=30, min_samples_leaf=2,
    random_state=42, n_jobs=-1, class_weight="balanced"
)
rf.fit(X_bal, y_bal)  # RF doesn't need scaling
train_t = time.time() - t0

pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, pred) * 100
rf_fa = accuracy_score(y_test[y_test==0], pred[y_test==0]) * 100
rf_ra = accuracy_score(y_test[y_test==1], pred[y_test==1]) * 100
print(f"  训练耗时: {train_t:.1f}s")
print(f"  总准确率: {rf_acc:.2f}%")
print(f"  新鲜识别: {rf_fa:.2f}%")
print(f"  腐烂识别: {rf_ra:.2f}%")

rf_path = os.path.join(OUTPUT_DIR, "apple_rf_model.pkl")
with open(rf_path, "wb") as f:
    pickle.dump(rf, f)
print(f"  已保存: {rf_path}")

# Top features
importances = rf.feature_importances_
top10 = np.argsort(importances)[-10:][::-1]
print("\n  最重要的10个特征索引:", top10.tolist())
print(f"  Top10 累计重要性: {importances[top10].sum():.3f}")

# ============ COMPARISON ============
print("\n" + "=" * 55)
print("  新旧模型对比")
print("=" * 55)
print(f"  特征维度: 15 -> {X_train.shape[1]}")
print(f"  SVM 原版: ?%  -> 新版: {svm_acc:.2f}% (F:{svm_fa:.1f}% R:{svm_ra:.1f}%)")
print(f"  RF  原版: ?%  -> 新版: {rf_acc:.2f}% (F:{rf_fa:.1f}% R:{rf_ra:.1f}%)")
print("\n完成!")
