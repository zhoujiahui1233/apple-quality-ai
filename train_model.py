import cv2
import numpy as np
import os
import sys
import json

sys.path.insert(0, "E:\\apple")
import importlib
import apple_defect_detector
importlib.reload(apple_defect_detector)

DATA_DIR = "E:\\apple\\kaggle_data\\extracted"
OUTPUT_MODEL = "E:\\apple\\apple_svm_model.dat"

def extract_features(img_bgr):
    h, w = img_bgr.shape[:2]
    img_p, _ = apple_defect_detector.preprocess(img_bgr)
    apple_mask, contour = apple_defect_detector.segment_apple(img_p)
    if apple_mask is None: return None
    
    gray = cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_p, cv2.COLOR_BGR2HSV)
    apple_pixels = apple_mask > 0
    apple_px_count = int(np.sum(apple_pixels))
    apple_ratio = apple_px_count / (img_p.shape[0] * img_p.shape[1])
    
    h_ch, s_ch, v_ch = cv2.split(hsv)
    apple_h = h_ch[apple_pixels]
    apple_s = s_ch[apple_pixels]
    apple_v = v_ch[apple_pixels]
    
    feats = [
        np.mean(apple_h), np.std(apple_h),
        np.mean(apple_s), np.std(apple_s),
        np.mean(apple_v), np.std(apple_v),
        apple_ratio,
    ]
    
    for lower, upper in apple_defect_detector.DEFECT_HSV_RANGES:
        dm = cv2.inRange(hsv, lower, upper)
        dm = cv2.bitwise_and(dm, apple_mask)
        feats.append(cv2.countNonZero(dm) / max(1, apple_px_count))
    
    blurred = cv2.bilateralFilter(gray, 9, 50, 50)
    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    lap_abs = np.uint8(np.abs(lap))[apple_pixels]
    feats.append(np.mean(lap_abs))
    feats.append(np.std(lap_abs))
    
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circ = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    feats.append(circ)
    feats.append(solidity)
    
    return np.array(feats, dtype=np.float32)

print("Extracting features...")
X_train, y_train = [], []

# Process fresh and rotten
for category, label in [("freshapples", 0), ("rottenapples", 1)]:
    path = os.path.join(DATA_DIR, "train", category)
    files = [f for f in os.listdir(path) if f.lower().endswith((".png",".jpg",".jpeg"))]
    print(f"  {category}: {len(files)} images...")
    
    for i, fname in enumerate(files):
        img = cv2.imread(os.path.join(path, fname))
        if img is None: continue
        feats = extract_features(img)
        if feats is not None:
            X_train.append(feats)
            y_train.append(label)
        if (i+1) % 500 == 0: print(f"    {i+1}/{len(files)}")

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
n_fresh = int(np.sum(y_train==0))
n_rotten = int(np.sum(y_train==1))
print(f"Fresh: {n_fresh}, Rotten: {n_rotten}")

# Balance: use equal number of fresh and rotten
min_count = min(n_fresh, n_rotten)
fresh_idx = np.where(y_train == 0)[0]
rotten_idx = np.where(y_train == 1)[0]
np.random.seed(42)
fresh_sel = np.random.choice(fresh_idx, min_count, replace=False)
rotten_sel = np.random.choice(rotten_idx, min_count, replace=False)
sel = np.concatenate([fresh_sel, rotten_sel])
X_bal = X_train[sel]
y_bal = y_train[sel]
print(f"Balanced: {len(X_bal)} samples ({int(np.sum(y_bal==0))} fresh, {int(np.sum(y_bal==1))} rotten)")

# Train SVM with balanced data
print("Training SVM...")
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setGamma(0.1)
svm.setC(10.0)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-6))
svm.train(X_bal, cv2.ml.ROW_SAMPLE, y_bal)

# Save using svm.save()
svm.save(OUTPUT_MODEL)
print(f"Model saved to {OUTPUT_MODEL}")

# Test
print("\nTesting...")
X_test, y_test = [], []
for category, label in [("freshapples", 0), ("rottenapples", 1)]:
    path = os.path.join(DATA_DIR, "test", category)
    for fname in os.listdir(path):
        img = cv2.imread(os.path.join(path, fname))
        if img is None: continue
        feats = extract_features(img)
        if feats is not None:
            X_test.append(feats)
            y_test.append(label)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int32)

_, y_pred = svm.predict(X_test)
y_pred = y_pred.flatten().astype(np.int32)

acc = np.mean(y_pred == y_test) * 100
fresh_acc = np.mean(y_pred[y_test==0]==0) * 100
rotten_acc = np.mean(y_pred[y_test==1]==1) * 100
print(f"Accuracy: {acc:.1f}%  Fresh: {fresh_acc:.1f}%  Rotten: {rotten_acc:.1f}%")
