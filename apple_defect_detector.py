"""
苹果表面缺陷识别程序 — Apple Surface Defect Detector
基于 OpenCV + NumPy，通过图像处理技术检测苹果表面缺陷并进行品质分级。
"""
import cv2
import numpy as np
import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None

import config

import config

logger = logging.getLogger("detector")

# ==================== 全局配置参数 ====================
IMAGE_WIDTH = 500
PREMIUM_THRESHOLD = 0.05
STANDARD_THRESHOLD = 0.15

APPLE_HSV_RANGES = [(np.array(l), np.array(u)) for l, u in config.APPLE_HSV_RANGES]

# skin color range
SKIN_HSV_LOWER = np.array([0, 18, 60])
SKIN_HSV_UPPER = np.array([20, 120, 255])


DEFECT_HSV_RANGES = [
    # Dark brown / black deep rot
    (np.array([0, 30, 0]), np.array([30, 255, 100])),
    # Very dark areas (shadows, deep rot)
    (np.array([0, 0, 0]), np.array([180, 255, 60])),
    # Black/dark gray
    (np.array([0, 0, 0]), np.array([180, 255, 40])),
    # Brown / rust colored defects
    (np.array([5, 40, 40]), np.array([25, 255, 180])),
    # True rot spots only: dark yellow/brown (low brightness), NOT normal yellow apple
    (np.array([15, 40, 0]), np.array([40, 255, 80])),
    # Grayish defects (low saturation, mid brightness) - tightened
    (np.array([0, 0, 50]), np.array([180, 30, 150])),
]

# ==================== 工具函数 ====================
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法加载图片: {path}")
    return img

def ensure_output_dir(path: str = "output") -> str:
    os.makedirs(path, exist_ok=True)


# ==================== 图像预处理 ====================

def preprocess(img: np.ndarray) -> np.ndarray:
    """
    图像预处理：
    1. 等比例缩放到统一宽度 (IMAGE_WIDTH)
    2. 高斯模糊降噪
    返回: (img_blurred, img_resized)
    """
    h, w = img.shape[:2]
    scale = IMAGE_WIDTH / w
    new_size = (IMAGE_WIDTH, int(round(h * scale)))
    img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # 高斯模糊去噪（5×5 核）
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)

    return img_blurred, img_resized


# ==================== 苹果区域分割 ====================

def segment_apple(img_bgr: np.ndarray) -> tuple:
    """Improved color-based apple segmentation with expanded color ranges."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Step 1: Try HoughCircles to find approximate apple location
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                param1=100, param2=35, minRadius=30, maxRadius=250)
    
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Step 2: Color-based apple detection (expanded ranges)
    color_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for lo, up in APPLE_HSV_RANGES:
        color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lo, up))
    color_mask = cv2.dilate(color_mask, k5, iterations=2)
    
    if circles is not None:
        # Use circle to focus the color mask
        cx, cy, r = np.round(circles[0][0]).astype("int")
        mask_r = min(int(r * 1.3), cx, cy, w - cx, h - cy)
        circle_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), mask_r, 255, -1)
        mask = cv2.bitwise_and(circle_mask, color_mask)
        
        # If color mask is too sparse, fall back to circle alone
        if np.sum(mask > 0) < np.sum(circle_mask > 0) * 0.3:
            mask = circle_mask
    else:
        # No circle found, try contour-based approach on color mask
        mask = color_mask
        co_try, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if co_try:
            largest = max(co_try, key=cv2.contourArea)
            if cv2.contourArea(largest) > 0.005 * h * w:
                # Keep only the largest color region
                mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [largest], -1, 255, -1)
            else:
                # Try edge-based fallback
                edges = cv2.Canny(blurred, 20, 80)
                closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k5, iterations=5)
                co_edge, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if co_edge:
                    largest = max(co_edge, key=cv2.contourArea)
                    if cv2.contourArea(largest) > 0.02 * h * w:
                        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
                        cv2.drawContours(mask, [largest], -1, 255, -1)
                    else:
                        return None, None
                else:
                    return None, None
        else:
            return None, None
    
    # Step 3: Clean up mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    mask = cv2.dilate(mask, k5, iterations=1)  # Expand to edge
    
    # Step 4: Remove skin (hand) pixels
    skin = cv2.inRange(hsv, SKIN_HSV_LOWER, SKIN_HSV_UPPER)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin))
    
    # Step 5: Find largest contour
    co, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not co: return None, None
    apple_contour = max(co, key=cv2.contourArea)
    if cv2.contourArea(apple_contour) < 0.005 * h * w: return None, None
    
    # Final clean mask
    clean = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(clean, [apple_contour], -1, 255, -1)
    return clean, apple_contour


def _is_shadow(contour: np.ndarray, img_bgr: np.ndarray, apple_mask: np.ndarray) -> bool:
    """Check if a detected region is likely a shadow rather than a real defect."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:,:,0].astype(np.float32)
    s_ch = hsv[:,:,1].astype(np.float32)
    v_ch = hsv[:,:,2].astype(np.float32)
    
    cmask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(cmask, [contour], -1, 255, -1)
    pts = np.where(cmask > 0)
    if len(pts[0]) < 10:
        return False
    
    dh = np.median(h_ch[pts])
    ds = np.median(s_ch[pts])
    dv = np.median(v_ch[pts])
    
    # Get apple-average HSV (excluding this region)
    apple_only = cv2.bitwise_and(apple_mask.astype(np.uint8), 255 - cmask)
    apple_pts = np.where(apple_only > 0)
    if len(apple_pts[0]) < 50:
        return False
    ah = np.median(h_ch[apple_pts])
    av = np.median(v_ch[apple_pts])
    
    hue_diff = min(abs(dh - ah), 180 - abs(dh - ah))
    apple_area = int(np.sum(apple_mask > 0))
    contour_area = cv2.contourArea(contour)
    area_ratio = contour_area / max(apple_area, 1)
    
    # Edge proximity
    apple_edge = cv2.Canny(apple_mask.astype(np.uint8), 0, 1)
    edge_dist_map = cv2.distanceTransform((255 - apple_edge).astype(np.uint8), cv2.DIST_L2, 3)
    min_edge_dist = np.min(edge_dist_map[pts])
    near_edge = min_edge_dist < 20
    
    # Real defect checks
    is_brown = 10 < dh < 40 and ds > 40
    is_very_dark = dv < 30
    is_gray = ds < 30 and dv > 40
    
    # Large area = likely real defect (shadows are usually small strips)
    if area_ratio > 0.03:
        return False
    
    # Clear defect characteristics
    if is_brown or is_very_dark or is_gray:
        return False
    
    # Shadow: small area, similar hue, at edge, and darker
    is_similar_hue = hue_diff < 15
    is_darker = dv < av - 15
    if near_edge and is_similar_hue and is_darker and ds < 100:
        return True
    
    return False
def detect_defects(img_bgr: np.ndarray, apple_mask: np.ndarray, sensitivity: int = 5, min_area: int = 200) -> tuple:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_scale = 0.6 + (sensitivity - 1) * 0.1
    defect_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, up in DEFECT_HSV_RANGES:
        sl = np.array([lo[0], lo[1], int(lo[2] * v_scale)])
        su = np.array([up[0], up[1], min(255, int(up[2] * v_scale))])
        defect_mask = cv2.bitwise_or(defect_mask, cv2.inRange(hsv, sl, su))
    defect_mask = cv2.bitwise_and(defect_mask, apple_mask)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, k5)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, k3)
    co, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dc = [c for c in co if cv2.contourArea(c) >= min_area]
    # Filter out shadows
    dc = [c for c in dc if not _is_shadow(c, img_bgr, apple_mask)]
    filtered = np.zeros(defect_mask.shape, dtype=np.uint8)
    cv2.drawContours(filtered, dc, -1, 255, -1)
    return filtered, dc
def classify_quality(defect_pixels: int, apple_pixels: int, premium_th: float = None, standard_th: float = None) -> tuple:
    if apple_pixels == 0: return "Unknown", 0.0, (128, 128, 128)
    if premium_th is None: premium_th = PREMIUM_THRESHOLD
    if standard_th is None: standard_th = STANDARD_THRESHOLD
    defect_ratio = defect_pixels / apple_pixels
    if defect_ratio < premium_th: return "Premium", defect_ratio, (0, 200, 0)
    elif defect_ratio <= standard_th: return "Standard", defect_ratio, (0, 200, 200)
    else: return "Reject", defect_ratio, (0, 0, 200)
def visualize_result(img_bgr, apple_contour, defect_contours, defect_mask, grade, defect_ratio, grade_color, yolo_conf=None):
    result = img_bgr.copy()
    cv2.drawContours(result, [apple_contour], -1, (0, 220, 0), 2)
    # Rebuild filtered mask from filtered contours only (ignore raw defect_mask)
    filtered_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(filtered_mask, defect_contours, -1, 255, -1)
    overlay = result.copy()
    overlay[filtered_mask > 0] = (0, 0, 255)
    result = cv2.addWeighted(overlay, 0.5, result, 0.5, 0)
    for i, contour in enumerate(defect_contours):
        cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
            cv2.putText(result, f"#{i+1}", (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Minimal overlay using alpha blending instead of PIL RGBA
    try:
        from PIL import Image, ImageDraw, ImageFont
        result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        # Create transparent overlay
        overlay = Image.new('RGBA', result_pil.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        fp = "C:/Windows/Fonts/msyh.ttc"
        fs = ImageFont.truetype(fp, 13)
        fl = ImageFont.truetype(fp, 15)
        gc = (int(grade_color[2]), int(grade_color[1]), int(grade_color[0]), 220)
        # Dark pill background - 40% opacity
        draw.rectangle([(4, 4), (220, 54)], fill=(0, 0, 0, 100))
        # Border line
        draw.rectangle([(4, 4), (220, 54)], outline=(0, 212, 255, 60), width=1)
        # Text
        draw.text((12, 8), str(grade), font=fl, fill=gc)
        draw.text((12, 28), "缺陷: {:.1f}% | {}个".format(defect_ratio*100, len(defect_contours)), font=fs, fill=(200, 200, 200, 200))
        # Composite
        result_pil = Image.alpha_composite(result_pil.convert('RGBA'), overlay).convert('RGB')
        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    except:
        # Fallback: no overlay, just putText directly
        cv2.putText(result, str(grade), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grade_color, 1)
        cv2.putText(result, "{:.1f}%".format(defect_ratio*100), (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return result
def save_result(result, original_path, output_dir='output'):
    """保存结果图片到输出目录。"""
    ensure_output_dir(output_dir)
    stem = Path(original_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(output_dir, f'{stem}_result_{timestamp}.jpg')
    cv2.imwrite(out_path, result)
    print(f'  [输出] 结果已保存: {out_path}')
    return out_path


# ==================== 主函数 ====================
# ==================== (v2) 新版特征提取 170维 ====================
SCALER_PATH = str(config.MODEL_PATHS["scaler"])
_scaler = None

def _calc_hist(region, mask, channel, bins, range_val):
    hist = cv2.calcHist([region], [channel], mask, [bins], range_val).flatten()
    hist = hist / (hist.sum() + 1e-8)
    return hist

def _lbp_features(gray_region):
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

def extract_features_v2(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    mask, _ = segment_apple(img_bgr)
    if mask is None: return None
    ap = mask > 0; ac = int(np.sum(ap))
    hc, sc, vc = cv2.split(hsv)
    feats = []
    # 1. HSV histograms (96)
    for channel, bins, rng in [(hc, 32, [0, 180]), (sc, 32, [0, 256]), (vc, 32, [0, 256])]:
        feats.extend(_calc_hist(channel, mask, 0, bins, rng).tolist())
    # 2. Color moments (9)
    for ch in [hc[ap], sc[ap], vc[ap]]:
        if len(ch) == 0: feats.extend([0, 0, 0])
        else:
            m = float(np.mean(ch)); s = float(np.std(ch))
            sk = float(np.mean(((ch - m) / (s + 1e-8)) ** 3))
            feats.extend([m, s, sk])
    # 3. LAB (6)
    for ch in cv2.split(lab):
        c = ch[ap]
        if len(c) > 0: feats.extend([float(np.mean(c)), float(np.std(c))])
        else: feats.extend([0, 0])
    # 4. Defect HSV ratios (6)
    for lo, up in DEFECT_HSV_RANGES:
        dm = cv2.inRange(hsv, lo, up); dm = cv2.bitwise_and(dm, mask)
        feats.append(float(cv2.countNonZero(dm)) / max(1, ac))
    # 5. Extra defect ranges (6)
    for lo, up in [(np.array([0,0,80]),np.array([180,50,200])),(np.array([0,50,150]),np.array([180,255,255])),
                   (np.array([100,30,0]),np.array([130,255,100])),(np.array([40,50,30]),np.array([80,255,150])),
                   (np.array([0,0,180]),np.array([180,30,255])),(np.array([20,50,100]),np.array([40,200,200]))]:
        dm = cv2.inRange(hsv, lo, up); dm = cv2.bitwise_and(dm, mask)
        feats.append(float(cv2.countNonZero(dm)) / max(1, ac))
    # 6. Multi-scale texture (12)
    for ksize in [3, 7, 15]:
        bf = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        lap = np.uint8(np.abs(cv2.Laplacian(bf, cv2.CV_64F)))[ap]
        if len(lap) > 0:
            feats.extend([float(np.mean(lap)), float(np.std(lap)), float(np.percentile(lap, 90))])
        else: feats.extend([0,0,0])
    # Sobel
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    feats.append(float(np.mean(np.sqrt(sx**2+sy**2)[ap] > 50)))
    # OTSU
    apple_gray = cv2.bitwise_and(gray, gray, mask=mask)
    if np.sum(mask) > 0:
        _, otsu = cv2.threshold(apple_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        feats.append(1.0 - float(np.sum(otsu[ap] > 0)) / max(1, ac))
    else: feats.append(0)
    feats.extend(_lbp_features(apple_gray))
    # 7. Shape (3)
    cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    p = cv2.arcLength(cnt, True); a = cv2.contourArea(cnt)
    circ = 4*np.pi*a/(p*p) if p > 0 else 0
    hull = cv2.convexHull(cnt); ha = cv2.contourArea(hull)
    xb, yb, wb, hb = cv2.boundingRect(cnt)
    feats.extend([float(circ), float(a/ha) if ha > 0 else 0, float(wb)/max(hb,1)])
    # 8. Area ratio (1)
    feats.append(float(ac) / (img_bgr.shape[0] * img_bgr.shape[1]))
    return np.array(feats, dtype=np.float32)

def load_scaler():
    global _scaler
    if _scaler is None and os.path.exists(SCALER_PATH):
        import pickle
        with open(SCALER_PATH, "rb") as f: _scaler = pickle.load(f)
    return _scaler


SVM_MODEL_PATH = str(config.MODEL_PATHS["svm"])
_svm_model = None
def load_svm_model():
    global _svm_model
    if _svm_model is None and os.path.exists(SVM_MODEL_PATH):
        _svm_model = cv2.ml.SVM_load(SVM_MODEL_PATH)
    return _svm_model
def predict_with_svm(img_bgr: np.ndarray, apple_mask: np.ndarray, contour: np.ndarray) -> tuple:
    model = load_svm_model()
    scaler = load_scaler()
    if model is None: return None, (0,0,0), 0.0, ""
    feats = extract_features_v2(img_bgr)
    if feats is None: return None, (0,0,0), 0.0, ""
    if scaler is not None:
        feats = scaler.transform(feats.reshape(1, -1)).flatten()
    _, pred = model.predict(np.array([feats], dtype=np.float32))
    pc = int(pred.flatten()[0])
    # Get decision distance from SVM
    _, dist = model.predict(np.array([feats], dtype=np.float32), flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
    decision_val = float(dist[0, 0]) if dist is not None else 0.0
    grade = (chr(20248)+chr(36136)+" Premium") if pc==0 else (chr(19981)+chr(21512)+chr(26684)+" Reject")
    color = (0,200,0) if pc==0 else (0,0,200)
    # Build judgment detail
    confidence = abs(decision_val) / (1 + abs(decision_val))
    detail = "决策距离: {:.3f} | 自信度: {:.1f}%".format(decision_val, confidence * 100)
    return grade, color, detail


RF_MODEL_PATH = str(config.MODEL_PATHS["rf"])
_rf_model = None
def load_rf_model():
    global _rf_model
    if _rf_model is None and os.path.exists(RF_MODEL_PATH):
        import pickle
        with open(RF_MODEL_PATH,"rb") as f: _rf_model = pickle.load(f)
    return _rf_model
def predict_with_rf(img_bgr: np.ndarray, apple_mask: np.ndarray, contour: np.ndarray) -> tuple:
    model = load_rf_model()
    if model is None: return None, (0,0,0), 0.0, ""
    feats = extract_features_v2(img_bgr)
    if feats is None: return None, (0,0,0), 0.0, ""
    pred = model.predict(np.array([feats], dtype=np.float32))[0]
    grade = (chr(20248)+chr(36136)+" Premium") if pred==0 else (chr(19981)+chr(21512)+chr(26684)+" Reject")
    color = (0,200,0) if pred==0 else (0,0,200)
    # Get class probabilities
    try:
        proba = model.predict_proba(np.array([feats], dtype=np.float32))[0]
        prem_conf = float(proba[0]) * 100
        rej_conf = float(proba[1]) * 100
        detail = "优质: {:.1f}% / 不合格: {:.1f}%".format(prem_conf, rej_conf)
    except:
        detail = ""
    return grade, color, detail


DL_MODEL_PATH = str(config.MODEL_PATHS["dl"])
_dl_model = None; _dl_transform = None; _dl_device = None
def load_dl_model():
    global _dl_model, _dl_transform, _dl_device
    if _dl_model is not None: return _dl_model
    if not os.path.exists(DL_MODEL_PATH) or torch is None: return None
    from torchvision import transforms, models; import torch.nn as nn
    _dl_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,2))
    model.load_state_dict(torch.load(DL_MODEL_PATH, map_location=_dl_device, weights_only=True))
    model = model.to(_dl_device); model.eval()
    _dl_model = model
    _dl_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    return _dl_model
def predict_with_dl(img_bgr: np.ndarray) -> tuple:
    model = load_dl_model()
    if model is None: return None, (0,0,0), 0.0, 0.0, ""
    from PIL import Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); img_pil = Image.fromarray(img_rgb)
    img_t = _dl_transform(img_pil).unsqueeze(0).to(_dl_device)
    with torch.no_grad():
        out = model(img_t); prob = torch.softmax(out, dim=1)[0]; pred = out.argmax(dim=1).item()
    conf = float(prob[pred].item())
    prem_p = float(prob[0].item()) * 100
    rej_p = float(prob[1].item()) * 100
    grade = (chr(20248)+chr(36136)+" Premium") if pred==0 else (chr(19981)+chr(21512)+chr(26684)+" Reject")
    color = (0,200,0) if pred==0 else (0,0,200)
    detail = "优质: {:.1f}% / 不合格: {:.1f}%".format(prem_p, rej_p)
    return grade, color, conf, detail



# ==================== YOLO ====================
YOLO_MODEL_PATH = str(config.MODEL_PATHS["yolo_cls"])
_yolo_model = None
def load_yolo_model():
    global _yolo_model
    if _yolo_model is not None: return _yolo_model
    if not os.path.exists(YOLO_MODEL_PATH): return None
    try:
        os.environ['YOLO_CONFIG_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.ultralytics')
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL_PATH)
        return _yolo_model
    except Exception:
        return None
def predict_with_yolo(img_bgr: np.ndarray) -> tuple:
    model = load_yolo_model()
    if model is None: return None, (0,0,0), 0.0, ""
    try:
        results = model(img_bgr, verbose=False)
        r = results[0]
        pred = r.probs.top1
        conf = float(r.probs.top1conf)
        probs = r.probs.data.cpu().numpy() if hasattr(r.probs, 'data') else None
        prem_p = float(probs[0]) * 100 if probs is not None else conf * 100
        rej_p = float(probs[1]) * 100 if probs is not None else (1 - conf) * 100
        result = (chr(20248)+chr(36136)+" Premium") if pred == 0 else (chr(19981)+chr(21512)+chr(26684)+" Reject")
        color = (0, 200, 0) if pred == 0 else (0, 0, 200)
        detail = "优质: {:.1f}% / 不合格: {:.1f}%".format(prem_p, rej_p)
        return result, color, conf, detail
    except Exception:
        return None, 0.0, 0.0, ""

# ==================== YOLO Detection ====================
YOLO_DETECT_MODEL_PATH = str(config.MODEL_PATHS["yolo_detect"])
_yolo_detect_model = None
def load_yolo_detect_model():
    global _yolo_detect_model
    if _yolo_detect_model is not None: return _yolo_detect_model
    if not os.path.exists(YOLO_DETECT_MODEL_PATH): return None
    try:
        os.environ['YOLO_CONFIG_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.ultralytics')
        from ultralytics import YOLO
        _yolo_detect_model = YOLO(YOLO_DETECT_MODEL_PATH)
        return _yolo_detect_model
    except Exception:
        return None

def predict_with_yolo_detect(img_bgr: np.ndarray, conf_th: float = 0.15) -> list:
    model = load_yolo_detect_model()
    if model is None: return None
    try:
        results = model(img_bgr, verbose=False, conf=conf_th)
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return None
        detections = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i])
            detections.append({'bbox': xyxy.tolist(), 'confidence': conf})
        return detections
    except Exception:
        return None

def visualize_yolo_detect(img_bgr, detections):
    result = img_bgr.copy()
    if detections is None:
        h, w = result.shape[:2]
        cv2.putText(result, 'No apples detected', (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return result
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = 'apple {:.0%}'.format(conf)
        cv2.putText(result, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    return result



def _draw_model_result(img_bgr, model_name, grade_text, confidence, grade_color, heatmap=None, defect_ratio=None):
    h, w = img_bgr.shape[:2]
    result = cv2.resize(img_bgr, (300, 300))
    if heatmap is not None:
        hm = cv2.resize(heatmap, (300, 300))
        hm = (hm * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        result = cv2.addWeighted(result, 0.6, hm_color, 0.4, 0)
    rh, rw = result.shape[:2]
    try:
        from PIL import Image, ImageDraw, ImageFont
        result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(result_pil)
        fp = "C:/Windows/Fonts/msyh.ttc"
        fn = ImageFont.truetype(fp, 14)
        fs = ImageFont.truetype(fp, 11)
        gc = (int(grade_color[2]), int(grade_color[1]), int(grade_color[0])) if isinstance(grade_color, tuple) else (0, 200, 0)
        # Small pill overlay
        draw.rectangle([(4, rh-34), (290, rh-2)], fill=(0, 0, 0, 90))
        draw.text((8, rh-30), model_name, font=fn, fill=(255, 255, 255))
        draw.text((100, rh-30), grade_text, font=fn, fill=gc)
        if confidence is not None:
            draw.text((8, rh-14), "Conf: {:.1f}%".format(confidence*100), font=fs, fill=(200, 200, 200))
        if defect_ratio is not None:
            draw.text((130, rh-14), "Defect: {:.2f}%".format(defect_ratio*100), font=fs, fill=(200, 100, 100))
        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    except:
        cv2.rectangle(result, (4, rh-34), (290, rh-2), (0, 0, 0), -1)
        cv2.putText(result, model_name, (8, rh-14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(result, grade_text, (100, rh-14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, gc, 1)
        if confidence is not None:
            cv2.putText(result, "{:.0f}%".format(confidence*100), (8, rh-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    return result

def _gradcam_dl(img_bgr):
    "Generate Grad-CAM heatmap for DL model"
    model = load_dl_model_raw()
    if model is None:
        return None
    
    from PIL import Image
    device = next(model.parameters()).device
    
    # Store activations and gradients
    activations = None
    gradients = None
    
    def forward_hook(m, i, o):
        nonlocal activations
        activations = o.detach()
    
    def backward_hook(m, gi, go):
        nonlocal gradients
        gradients = go[0].detach()
    
    # Hook the last conv layer (features.12 is Conv2dNormActivation, access [0] for Conv2d)
    # Actually use the whole features.12 for its output (after BN+activation)
    fwd_handle = model.features[12].register_forward_hook(forward_hook)
    bwd_handle = model.features[12].register_full_backward_hook(backward_hook)
    
    # Forward pass
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((224, 224))
    img_t = _dl_transform(img_pil).unsqueeze(0).to(device)
    img_t.requires_grad_(True)
    
    out = model(img_t)
    score = out[0, out.argmax(dim=1).item()]
    
    # Backward
    model.zero_grad()
    score.backward(retain_graph=True)
    
    fwd_handle.remove()
    bwd_handle.remove()
    
    if activations is None or gradients is None:
        return None
    
    # Grad-CAM: weights = global avg pool of gradients
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))  # [1, 1, H, W]
    
    # Resize to original image size
    cam = cam.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam

def load_dl_model_raw():
    global _dl_model, _dl_device, _dl_transform
    if _dl_model is not None: return _dl_model
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apple_dl_model.pth")
    if not os.path.exists(path): return None
    try:
        from torchvision import transforms, models
        import torch.nn as nn
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.mobilenet_v3_large(weights=None)
        in_feat = model.classifier[0].in_features
        model.classifier = nn.Sequential(nn.Linear(in_feat,512), nn.Hardswish(), nn.Dropout(0.3), nn.Linear(512,128), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(128,2))
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model = model.to(device); model.eval()
        _dl_model = model; _dl_device = device
        _dl_transform = transforms.Compose([transforms.Resize((384,384)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
        return model
    except: return None

def visualize_svm(img_bgr, grade_text, grade_color, defect_ratio=None):
    return _draw_model_result(img_bgr, "SVM", grade_text, None, grade_color, defect_ratio=defect_ratio)

def visualize_rf(img_bgr, grade_text, grade_color, defect_ratio=None):
    return _draw_model_result(img_bgr, "RandomForest", grade_text, None, grade_color, defect_ratio=defect_ratio)

def visualize_dl(img_bgr, grade_text, confidence, grade_color, defect_ratio=None):
    heatmap = _gradcam_dl(img_bgr)
    return _draw_model_result(img_bgr, "Deep Learning", grade_text, confidence, grade_color, heatmap, defect_ratio=defect_ratio)

def visualize_yolo(img_bgr, grade_text, confidence, grade_color, defect_ratio=None):
    return _draw_model_result(img_bgr, "YOLO", grade_text, confidence, grade_color, defect_ratio=defect_ratio)

def main():
    parser = argparse.ArgumentParser(description="苹果表面缺陷识别程序")
    parser.add_argument("input", help="输入图片路径或包含图片的目录")
    args = parser.parse_args()
    input_path = args.input
    if os.path.isdir(input_path):
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [f for f in os.listdir(input_path) if Path(f).suffix.lower() in image_exts]
        if not files:
            print(f"错误: 目录 '{input_path}' 中没有找到图片文件")
            sys.exit(1)
        for fname in files:
            process_single_image(os.path.join(input_path, fname))
    else:
        process_single_image(input_path)

def process_single_image(image_path):
    """处理单张图片的完整流程。"""
    
    print(f"处理: {image_path}")
    
    try:
        # 1. 加载 & 预处理
        img_original = load_image(image_path)
        img_processed, img_resized = preprocess(img_original)
        print(f"  [预处理] 完成 → 尺寸: {img_processed.shape[1]}x{img_processed.shape[0]}")

        # 2. 分割苹果区域
        apple_mask, apple_contour = segment_apple(img_processed)
        if apple_mask is None:
            print(f"  [分割] ❌ 未检测到苹果区域，跳过")
            return
        apple_pixels = np.sum(apple_mask > 0)
        total_pixels = apple_mask.shape[0] * apple_mask.shape[1]
        area_ratio = apple_pixels / total_pixels * 100
        print(f"  [分割] 完成 → 苹果区域: {apple_pixels} 像素 ({area_ratio:.1f}% 画面占比)")

        # 3. 缺陷检测
        defect_mask, defect_contours = detect_defects(img_processed, apple_mask)
        defect_pixels = np.sum(defect_mask > 0)
        defect_count = len(defect_contours)
        print(f"  [检测] 完成 → 发现 {defect_count} 个缺陷区域，共 {defect_pixels} 像素")
        # 4. 品质分级
        apple_pixels = np.sum(apple_mask > 0)
        grade, defect_ratio, grade_color = classify_quality(defect_pixels, apple_pixels)
        print(f"  [分级] 缺陷占比: {defect_ratio*100:.2f}%")
        print(f"  [分级] 品质等级: {grade}")

        # 5. 结果可视化 & 输出
        result_img = visualize_result(
            img_processed, apple_contour, defect_contours,
            defect_mask, grade, defect_ratio, grade_color
        )
        save_result(result_img, image_path)
        print(f"  [完成] ✅")

    except Exception as e:
        print(f"  [??] ❌ {e}")

if __name__ == "__main__":
    main()