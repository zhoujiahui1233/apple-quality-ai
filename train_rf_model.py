import sys
sys.stdout.reconfigure(encoding="utf-8")
import cv2, numpy as np, os, pickle, time
sys.path.insert(0, "E:\\apple")
import importlib, apple_defect_detector
importlib.reload(apple_defect_detector)

DATA_DIR = "E:\\apple\\kaggle_data\\extracted"
OUTPUT_MODEL = "E:\\apple\\apple_rf_model.pkl"

def extract_features(img_bgr):
    img_p, _ = apple_defect_detector.preprocess(img_bgr)
    apple_mask, contour = apple_defect_detector.segment_apple(img_p)
    if apple_mask is None: return None
    gray = cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_p, cv2.COLOR_BGR2HSV)
    ap = apple_mask > 0; ac = int(np.sum(ap))
    ar = ac / (img_p.shape[0] * img_p.shape[1])
    hc, sc, vc = cv2.split(hsv)
    feats = [float(np.mean(hc[ap])), float(np.std(hc[ap])), float(np.mean(sc[ap])), float(np.std(sc[ap])), float(np.mean(vc[ap])), float(np.std(vc[ap])), float(ar)]
    for lo, up in apple_defect_detector.DEFECT_HSV_RANGES:
        dm = cv2.inRange(hsv, lo, up)
        dm = cv2.bitwise_and(dm, apple_mask)
        feats.append(float(cv2.countNonZero(dm)) / max(1, ac))
    blurred = cv2.bilateralFilter(gray, 9, 50, 50)
    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    la = np.uint8(np.abs(lap))[ap]
    feats.append(float(np.mean(la))); feats.append(float(np.std(la)))
    perim = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circ = 4 * np.pi * area / (perim * perim) if perim > 0 else 0
    hull = cv2.convexHull(contour)
    ha = cv2.contourArea(hull)
    sol = area / ha if ha > 0 else 0
    feats.append(float(circ)); feats.append(float(sol))
    return feats

print("=" * 50)
print("苹果 Random Forest 模型训练")
print("=" * 50)

# Step 1: Load features
print("\n[1/4] 加载训练数据...")
t0 = time.time()
X_train, y_train = [], []
for cat, lab in [("freshapples", 0), ("rottenapples", 1)]:
    path = os.path.join(DATA_DIR, "train", cat)
    files = [f for f in os.listdir(path) if f.lower().endswith((".png",".jpg",".jpeg"))]
    count = 0
    for fname in files:
        img = cv2.imread(os.path.join(path, fname))
        if img is None: continue
        feats = extract_features(img)
        if feats is not None:
            X_train.append(feats); y_train.append(lab)
            count += 1
    print(f"  {cat}: {count}")

X_train = np.array(X_train); y_train = np.array(y_train)
n0 = int(np.sum(y_train==0)); n1 = int(np.sum(y_train==1))
print(f"  总计: {len(X_train)} ({n0} fresh, {n1} rotten) [{time.time()-t0:.0f}s]")

# Step 2: Balance
print("\n[2/4] 平衡样本...")
min_c = min(n0, n1)
fi = np.where(y_train==0)[0]; ri = np.where(y_train==1)[0]
np.random.seed(42)
fs = np.random.choice(fi, min_c, replace=False)
rs = np.random.choice(ri, min_c, replace=False)
sel = np.concatenate([fs, rs])
X_bal, y_bal = X_train[sel], y_train[sel]
print(f"  平衡后: {len(X_bal)}")

# Step 3: Load test
print("\n[3/4] 加载测试数据...")
X_test, y_test = [], []
for cat, lab in [("freshapples",0), ("rottenapples",1)]:
    path = os.path.join(DATA_DIR, "test", cat)
    for fname in os.listdir(path):
        img = cv2.imread(os.path.join(path, fname))
        if img is None: continue
        feats = extract_features(img)
        if feats is not None:
            X_test.append(feats); y_test.append(lab)
X_test = np.array(X_test); y_test = np.array(y_test)
print(f"  测试集: {len(X_test)}")

# Step 4: Train Random Forest
print("\n[4/4] 训练 Random Forest...")
from sklearn.ensemble import RandomForestClassifier
t0 = time.time()
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_bal, y_bal)
train_t = time.time() - t0

# Evaluate
pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
acc = accuracy_score(y_test, pred) * 100
fa = accuracy_score(y_test[y_test==0], pred[y_test==0]) * 100
ra = accuracy_score(y_test[y_test==1], pred[y_test==1]) * 100

print(f"\n  训练耗时: {train_t:.1f}s")
print(f"  准确率: {acc:.1f}%")
print(f"  新鲜识别: {fa:.1f}%")
print(f"  腐烂识别: {ra:.1f}%")

# Feature importance
feat_names = ["H_mean","H_std","S_mean","S_std","V_mean","V_std","area_ratio","def1","def2","def3","def4","lap_mean","lap_std","circularity","solidity"]
importances = model.feature_importances_
top5 = np.argsort(importances)[-5:][::-1]
print("\n  最重要特征:")
for i in top5:
    print(f"    {feat_names[i]}: {importances[i]:.3f}")

# Save
print(f"\n  保存模型: {OUTPUT_MODEL}")
with open(OUTPUT_MODEL, "wb") as f:
    pickle.dump(model, f)

print("\n完成!")
