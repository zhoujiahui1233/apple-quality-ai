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

# ==================== (v4) 缺陷块分析（8.1 第一步） ====================

def _glcm_props(gray_block):
    """对单块灰度区域计算 GLCM 四特征（对比度/能量/同质性/相关性）。
    区域过小或无灰度变化时返回全 0。"""
    import numpy as _np
    try:
        from skimage.feature import graycomatrix, graycoprops
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]
    g = _np.uint8(_np.floor(gray_block.astype(_np.float32) / 256.0 * 32.0))
    if g.size < 64 or int(g.max()) - int(g.min()) < 2:
        return [0.0, 0.0, 0.0, 0.0]
    try:
        glcm = graycomatrix(g, distances=[1], angles=[0, _np.pi/4, _np.pi/2, 3*_np.pi/4],
                            levels=32, symmetric=True, normed=True)
        feats = []
        for prop in ("contrast", "energy", "homogeneity", "correlation"):
            vals = graycoprops(glcm, prop)
            feats.append(float(_np.nanmean(vals)))
        return feats
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


def analyze_defect_blocks(img_bgr, apple_mask, defect_mask, min_area=60):
    """缺陷块切分 + 特征提取（参考《刘佳浩》区域生长+GLCM、《贾馥谦》纹理特征）。

    对缺陷掩码的每个连通域独立成块，提取 4 组特征：
      - 位置：块中心相对苹果中心的归一化坐标（0~1，果萼/果梗常在两端）
      - 形状：面积、圆形度、长宽比（虫眼小而圆、碰伤片状）
      - 颜色：HSV 均值（腐烂偏黄褐、碰伤偏暗、虫眼偏黑）
      - 纹理：GLCM 对比度/能量/同质性/相关性（腐烂纹理乱、碰伤边界清晰）

    返回 list[dict]，每块一个字典（键见 FEATURE_KEYS），按面积降序。
    """
    if img_bgr is None or apple_mask is None or defect_mask is None:
        return []
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hc, sc, vc = cv2.split(hsv)

    # 苹果中心/半径（用于位置归一化）
    co, _ = cv2.findContours(apple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not co:
        return []
    cnt = max(co, key=cv2.contourArea)
    (cx, cy), cr = cv2.minEnclosingCircle(cnt)
    ac = int(np.sum(apple_mask > 0)) or 1

    blk_co, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blk_co = [c for c in blk_co if cv2.contourArea(c) >= min_area]
    blk_co.sort(key=cv2.contourArea, reverse=True)

    blocks = []
    for i, c in enumerate(blk_co, 1):
        x, y, w, h = cv2.boundingRect(c)
        bx, by = x + w / 2, y + h / 2
        area = float(cv2.contourArea(c))
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0.0
        bmask = np.zeros_like(apple_mask)
        cv2.drawContours(bmask, [c], -1, 255, -1)
        # 位置：距苹果中心归一化（0=中心，1=边缘）
        dist_norm = float(np.hypot(bx - cx, by - cy) / max(1e-6, cr))
        # 颜色均值（块内）
        seg = bmask > 0
        if np.sum(seg) > 0:
            h_mean = float(np.mean(hc[seg])); s_mean = float(np.mean(sc[seg])); v_mean = float(np.mean(vc[seg]))
        else:
            h_mean = s_mean = v_mean = 0.0
        # 纹理（块内灰度，含小块形态学）
        block_gray = cv2.bitwise_and(gray, gray, mask=bmask)
        glcm = _glcm_props(block_gray)
        blocks.append({
            "index": i,
            "area": round(area, 1),
            "area_ratio": round(area / ac, 4),
            "circularity": round(circ, 3),
            "aspect": round(max(w, h) / max(1e-6, min(w, h)), 3),
            "center_norm": [round(bx / max(1, img_bgr.shape[1]), 3),
                            round(by / max(1, img_bgr.shape[0]), 3)],
            "dist_norm": round(dist_norm, 3),
            "hsv_mean": [round(h_mean, 1), round(s_mean, 1), round(v_mean, 1)],
            "glcm": [round(v, 4) for v in glcm],
            "bbox": [int(x), int(y), int(w), int(h)],
        })
    return blocks


FEATURE_KEYS = ["index", "area", "area_ratio", "circularity", "aspect",
                "center_norm", "dist_norm", "hsv_mean", "glcm", "bbox"]

# ==================== (v3) 论文吸收函数 ====================

def auto_white_balance(img_bgr: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """灰世界白平衡（参考《仇德凯》颜色校正：麦克白色卡 20.9->13.4）。
    光照偏色时按各通道均值对齐，strength 控制校正强度，0=关闭。"""
    if strength <= 0:
        return img_bgr
    out = img_bgr.astype(np.float32)
    for i in range(3):
        ch = out[:, :, i]
        mean = float(np.mean(ch))
        if mean < 1e-6:
            continue
        gain = 128.0 / mean
        out[:, :, i] = np.clip(ch * (1 + (gain - 1) * strength), 0, 255)
    return np.uint8(out)


def _illumination_correct_v(img_bgr: np.ndarray, strength: float = 0.85) -> np.ndarray:
    """Multi-scale Retinex on the HSV Value channel to flatten spherical
    shading before defect detection. H/S are kept unchanged so genuine
    brown/dark defects stay detectable while smooth illumination gradients
    (edge shadows, vignetting) are suppressed. strength=0 disables.

    Reference idea: Retinex luminance correction is widely used in fruit
    surface defect detection to reduce shadow false positives (apple defect
    literature: adaptive bilateral-filter SSR + gamma, 农业工程学报 光照无关图).
    """
    if strength <= 0:
        return img_bgr
    try:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)
        v = np.maximum(v, 1.0)
        log_v = np.log(v)
        h_img, w_img = img_bgr.shape[:2]
        base = max(h_img, w_img)
        # Multi-scale surround: small/medium/large to flatten local & global
        # shading. Weighted combination approximates MSR.
        reflect = np.zeros_like(log_v)
        total_w = 0.0
        for sigma_ratio, wgt in [(1 / 30.0, 1.0), (1 / 80.0, 1.0), (1 / 200.0, 1.0)]:
            sigma = max(2.0, base * sigma_ratio)
            ksize = int(max(3, sigma * 6))
            if ksize % 2 == 0:
                ksize += 1
            surround = cv2.GaussianBlur(v, (ksize, ksize), sigma)
            surround = np.maximum(surround, 1.0)
            reflect += wgt * (log_v - np.log(surround))
            total_w += wgt
        reflect /= total_w
        rmin, rmax = float(reflect.min()), float(reflect.max())
        if rmax - rmin < 1e-6:
            return img_bgr
        corrected = (reflect - rmin) / (rmax - rmin) * 255.0
        # blend: keep some original so color shading cues remain for real
        # defects, but strongly flatten smooth illumination.
        v_new = corrected * strength + v * (1.0 - strength)
        v_new = np.clip(v_new, 0, 255).astype(np.uint8)
        hsv[:, :, 2] = v_new
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    except Exception:
        return img_bgr


def ensemble_score(svm_grade, rf_grade, dl_grade, yolo_grade,
                   svm_conf, rf_conf, dl_conf, yolo_conf,
                   defect_ratio, premium_th, standard_th) -> dict:
    """多模型加权融合 + 先验知识修正（参考《贾馥谦》先验知识多模型融合、《张瑞琪》CNN-SVM）。
    每模型给出合格倾向分 s∈[0,1]（判合格=置信度，判不合格=1-置信度）。
    先验：缺陷比例 < premium_th -> 1；> standard_th -> 0；中间线性。
    返回 {score, grade, conf, votes}。"""
    def _s(grade, conf):
        if not grade:
            return 0.5, 0.0
        ok = ("Premium" in str(grade)) or ("合格" in str(grade))
        try:
            c = float(conf)
        except (TypeError, ValueError):
            c = 0.5
        c = min(max(c, 0.0), 1.0)
        return (c if ok else 1.0 - c), c
    svm_s, svm_c = _s(svm_grade, svm_conf)
    rf_s, rf_c = _s(rf_grade, rf_conf)
    dl_s, dl_c = _s(dl_grade, dl_conf)
    yolo_s, yolo_c = _s(yolo_grade, yolo_conf)
    try:
        ratio = float(defect_ratio)
        if ratio <= float(premium_th):
            prior = 1.0
        elif ratio >= float(standard_th):
            prior = 0.0
        else:
            prior = 1.0 - (ratio - float(premium_th)) / max(1e-6, float(standard_th) - float(premium_th))
    except (TypeError, ValueError):
        prior = 0.5
    w = {"svm": 0.2, "rf": 0.2, "dl": 0.3, "yolo": 0.3}
    score = (w["svm"] * svm_s + w["rf"] * rf_s + w["dl"] * dl_s + w["yolo"] * yolo_s) * 0.85 + prior * 0.15
    conf = abs(score - 0.5) * 2.0
    grade = "合格 Premium" if score >= 0.5 else "不合格 Reject"
    def _txt(n, s, c):
        return f"{n} {'合格' if s >= 0.5 else '不合格'} {c * 100:.0f}%"
    votes = " | ".join([
        _txt("SVM", svm_s, svm_c), _txt("RF", rf_s, rf_c),
        _txt("DL", dl_s, dl_c), _txt("YOLO", yolo_s, yolo_c),
        f"先验(缺陷{ratio * 100:.1f}%) {'合格' if prior >= 0.5 else '不合格'}",
    ])
    return {"score": float(score), "grade": grade, "conf": float(conf), "votes": votes}


def estimate_diameter_px(contour, img_width=None) -> dict:
    """果径参考（参考《赵树昌》GB/T10651-2008 果径分级、《仇德凯》最小外接矩形）。
    返回外接圆直径(像素)、归一化直径(相对图宽)、长轴/短轴比。"""
    if contour is None or len(contour) < 5:
        return {"diameter_px": 0.0, "diameter_norm": 0.0, "aspect": 1.0}
    (_, _), r = cv2.minEnclosingCircle(contour)
    rect = cv2.minAreaRect(contour)
    (wa, ha) = rect[1]
    aspect = max(wa, ha) / max(1e-6, min(wa, ha))
    d = 2.0 * r
    norm = d / img_width if img_width else 0.0
    return {"diameter_px": float(d), "diameter_norm": float(norm), "aspect": float(aspect)}


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

_YOLO_DETECT_CACHE = {"model": None}
_YOLO_DETECT_PATH = str(config.MODEL_PATHS["yolo_detect"])


def _yolo_detect_fallback(img_bgr):
    """Use the YOLO apple-detector box as the apple region when color
    segmentation fails (background shares apple hue). Returns (x1,y1,x2,y2)
    in pixel coords, or None on any failure. Loaded lazily to avoid import
    cycles and to keep it out of the hot path."""
    try:
        if _YOLO_DETECT_CACHE["model"] is None:
            if not os.path.exists(_YOLO_DETECT_PATH):
                return None
            os.environ.setdefault(
                "YOLO_CONFIG_DIR",
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             ".ultralytics"))
            from ultralytics import YOLO
            _YOLO_DETECT_CACHE["model"] = YOLO(_YOLO_DETECT_PATH)
        model = _YOLO_DETECT_CACHE["model"]
        r = model.predict(img_bgr, conf=0.5, imgsz=640, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        # prefer the highest-confidence apple box
        best = int(np.argmax(confs))
        x1, y1, x2, y2 = xyxy[best]
        return (float(x1), float(y1), float(x2), float(y2))
    except Exception:
        return None


def mask_from_yolo_boxes(img_bgr, boxes, conf_th=0.5):
    """用已算好的 YOLO 苹果框构造 (mask, contour)，避免重复推理。

    只采纳置信度 >= conf_th 的框，取面积最大者；矩形轻微腐蚀以免框边被判为缺陷。
    无可用框时返回 (None, None)。"""
    h, w = img_bgr.shape[:2]
    usable = [d for d in (boxes or []) if float(d.get("confidence", 0.0)) >= conf_th]
    if not usable:
        return None, None
    x1, y1, x2, y2 = max(
        (d["bbox"] for d in usable),
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None, None
    clean = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(clean, (x1, y1), (x2, y2), 255, -1)
    # Erode a bit so the rectangle border itself is not treated as a defect
    clean = cv2.erode(clean, np.ones((7, 7), np.uint8), iterations=2)
    co, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not co:
        return None, None
    contour = max(co, key=cv2.contourArea)
    return clean, contour


def apple_region_from_yolo(img_bgr, conf_th=0.5):
    """Return (mask, contour) using the YOLO apple-detector box as the apple
    region. Falls back to (None, None) when no box is found. The mask is the
    box rectangle slightly eroded so the box edge is not seen as a defect."""
    boxes = predict_with_yolo_detect(img_bgr, conf_th=conf_th)
    return mask_from_yolo_boxes(img_bgr, boxes, conf_th=conf_th)


def segment_apple(img_bgr: np.ndarray) -> tuple:
    """Improved color-based apple segmentation with expanded color ranges."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    hsv0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    col0 = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for lo, up in APPLE_HSV_RANGES:
        col0 = cv2.bitwise_or(col0, cv2.inRange(hsv0, lo, up))
    # If almost the whole frame looks like apple color, color segmentation is
    # unreliable (background shares the same hue). Fall back to the YOLO
    # detector box so defects are searched inside the real apple only.
    if (col0 > 0).sum() > 0.78 * h * w:
        yolo_box = _yolo_detect_fallback(img_bgr)
        if yolo_box is not None:
            x1, y1, x2, y2 = yolo_box
            clean = np.zeros((h, w), np.uint8)
            cv2.rectangle(clean, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
            # shrink a little so the box edge doesn't count as defect
            clean = cv2.erode(clean, np.ones((7, 7), np.uint8), iterations=2)
            cof, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
            if cof:
                cnt = max(cof, key=cv2.contourArea)
                return clean, cnt
    
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

    # Step 5b: Trim mask that touches image border (background bleed-in).
    # A real single apple normally does NOT touch all four image edges; if the
    # largest region reaches the border, background pixels were mis-segmented.
    cm = np.zeros((h, w), np.uint8)
    cv2.drawContours(cm, [apple_contour], -1, 255, -1)
    touches = (cm[0, :].sum() + cm[-1, :].sum() +
               cm[:, 0].sum() + cm[:, -1].sum()) > 0
    if touches:
        # Erode just enough to disconnect border pixels, then keep the
        # largest remaining inner component as the apple.
        eroded = cv2.erode(cm, np.ones((7, 7), np.uint8), iterations=2)
        # Drop any remaining connection to the border
        border_mask = np.zeros((h, w), np.uint8)
        cv2.rectangle(border_mask, (0, 0), (w - 1, h - 1), 255, 3)
        eroded = cv2.bitwise_and(eroded, cv2.bitwise_not(border_mask))
        co2, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if co2:
            inner = max(co2, key=cv2.contourArea)
            if cv2.contourArea(inner) > 0.01 * h * w:
                # Re-dilate to restore the apple edge lost by erosion
                restored = cv2.dilate(
                    cv2.drawContours(np.zeros((h, w), np.uint8), [inner], -1, 255, -1),
                    np.ones((5, 5), np.uint8), iterations=3)
                apple_contour = max(
                    cv2.findContours(restored, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[0],
                    key=cv2.contourArea, default=apple_contour)
    
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


def _region_texture_energy(img_gray: np.ndarray, contour) -> float:
    """Mean absolute Laplacian inside a contour. Shadows are smooth (low
    energy); real defects (rot/bruise/scab) have rough texture (high energy)."""
    h, w = img_gray.shape[:2]
    cm = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(cm, [contour], -1, 255, -1)
    pts = np.where(cm > 0)
    if len(pts[0]) < 16:
        return 0.0
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    vals = lap[pts]
    return float(np.mean(np.abs(vals)))


def detect_defects(img_bgr: np.ndarray, apple_mask: np.ndarray, sensitivity: int = 5, min_area: int = 200) -> tuple:
    # Flatten spherical shading first so edge shadows are not seen as defects.
    img_bgr = _illumination_correct_v(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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

    # Texture gate: shadows are smooth gradients (low Laplacian energy);
    # real defects are rough. Drop smooth regions regardless of color.
    tex_min = 2.5
    dc = [c for c in dc if _region_texture_energy(gray, c) >= tex_min]

    # Filter out regions that sit on the apple boundary or image border:
    # true surface defects are surrounded by healthy skin.
    apple_eroded = cv2.erode(apple_mask.astype(np.uint8),
                             np.ones((5, 5), np.uint8), iterations=4)
    def _inside_ratio(c):
        cm = np.zeros_like(apple_mask)
        cv2.drawContours(cm, [c], -1, 255, -1)
        inter = cv2.countNonZero(cv2.bitwise_and(cm, apple_eroded))
        total = cv2.countNonZero(cm)
        return inter / max(total, 1)

    inner = [c for c in dc if _inside_ratio(c) >= 0.5]
    dc = inner

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
def visualize_result(img_bgr, apple_contour, defect_contours, defect_mask,
                     grade, defect_ratio, grade_color, yolo_conf=None,
                     ai_boxes=None):
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
    # AI defect boxes drawn in orange/amber so they are distinguishable from
    # the HSV-rule defects (red).
    if ai_boxes:
        for d in ai_boxes:
            x1, y1, x2, y2 = d["bbox"]
            conf = d["confidence"]
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(result, "AI {:.0%}".format(conf), (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
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
    # ==================== (v3) 论文吸收特征 ====================
    # 9. 着色率/红度特征 (4) —— 参考《刘佳浩》HSI 红色着色率、《贾馥谦》H 分量色泽特征
    red_mask = cv2.bitwise_and(mask, cv2.inRange(hsv, np.array([0, 60, 80]), np.array([10, 255, 255])))
    red_mask = cv2.bitwise_or(red_mask, cv2.bitwise_and(mask, cv2.inRange(hsv, np.array([170, 60, 80]), np.array([180, 255, 255]))))
    red_cnt = int(cv2.countNonZero(red_mask))
    red_ratio = red_cnt / max(1, ac)
    # 深红(着色充分)区域占比
    deep_red = cv2.bitwise_and(mask, cv2.inRange(hsv, np.array([0, 100, 100]), np.array([8, 255, 255])))
    deep_red = cv2.bitwise_or(deep_red, cv2.bitwise_and(mask, cv2.inRange(hsv, np.array([172, 100, 100]), np.array([180, 255, 255]))))
    feats.append(red_ratio)
    feats.append(float(cv2.countNonZero(deep_red)) / max(1, ac))
    # 红色区域色调均值/离散度（红苹果应集中在小 H 区间）
    if red_cnt > 50:
        red_h = hc[red_mask > 0]
        feats.append(float(np.mean(red_h)))
        feats.append(float(np.std(red_h)))
    else:
        feats.extend([0.0, 0.0])
    # 10. GLCM 纹理特征 (8) —— 参考《刘佳浩》灰度共生矩阵区分缺陷类型、《贾馥谦》纹理特征
    def _glcm_feats(region_gray, roi_mask):
        try:
            from skimage.feature import graycomatrix, graycoprops
            gl = cv2.bitwise_and(region_gray, region_gray, mask=roi_mask)
            if roi_mask is None or cv2.countNonZero(roi_mask) < 100:
                return [0.0, 0.0, 0.0, 0.0]
            g = np.uint8(np.floor(gl.astype(np.float32) / 256.0 * 32.0))  # 缩放到 0-31 匹配 levels
            glcm = graycomatrix(g, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=32, symmetric=True, normed=True)
            feats4 = []
            for prop in ("contrast", "energy", "homogeneity", "correlation"):
                vals = graycoprops(glcm, prop)
                feats4.append(float(np.nanmean(vals)))
            return feats4
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]
    apple_gray8 = np.uint8(apple_gray)
    feats.extend(_glcm_feats(apple_gray8, mask))
    # 缺陷区域（HSV 阈值组合，与 detect_defects 一致的口径）
    defect_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for lo, up in DEFECT_HSV_RANGES[:4]:
        defect_mask = cv2.bitwise_or(defect_mask, cv2.inRange(hsv, lo, up))
    defect_mask = cv2.bitwise_and(defect_mask, mask)
    feats.extend(_glcm_feats(apple_gray8, defect_mask))
    # 11. 果径/果形参考 (3) —— 参考《赵树昌》GB/T10651-2008 果径分级、《仇德凯》最小外接矩形
    (ex, ey), er = cv2.minEnclosingCircle(cnt)
    feats.append(float(2 * er) / max(1, img_bgr.shape[1]))   # 外接圆直径(归一化像素)
    feats.append(float(max(wb, hb)) / max(1, min(wb, hb)))   # 长轴/短轴比
    feats.append(float(cv2.arcLength(hull, True)) / max(1, p))  # 凸包周长/轮廓周长(凹凸度)
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
def predict_with_svm(img_bgr: np.ndarray, apple_mask: np.ndarray, contour: np.ndarray,
                     feats: np.ndarray = None) -> tuple:
    model = load_svm_model()
    scaler = load_scaler()
    if model is None: return None, (0,0,0), ""
    if feats is None:
        feats = extract_features_v2(img_bgr)
    if feats is None: return None, (0,0,0), ""
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
def predict_with_rf(img_bgr: np.ndarray, apple_mask: np.ndarray, contour: np.ndarray,
                    feats: np.ndarray = None) -> tuple:
    model = load_rf_model()
    if model is None: return None, (0,0,0), ""
    if feats is None:
        feats = extract_features_v2(img_bgr)
    if feats is None: return None, (0,0,0), ""
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
    """DL 分类模型统一入口：与 load_dl_model_raw 使用同一结构
    (MobileNetV3-Large @384)，避免两处结构不一致导致权重加载失败。"""
    return load_dl_model_raw()
def predict_with_dl(img_bgr: np.ndarray) -> tuple:
    model = load_dl_model()
    if model is None: return None, (0,0,0), 0.0, ""
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


# ==================== YOLO Defect Detection ====================
YOLO_DEFECT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "defect_detect_best.pt")
YOLO_DEFECT_MODEL_PATH2 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "defect_detect_best_v2.pt")
_yolo_defect_model = None
_yolo_defect_model2 = None


def load_yolo_defect_model():
    """Load the YOLO defect-detection model (defect_detect_best.pt)."""
    global _yolo_defect_model
    if _yolo_defect_model is not None:
        return _yolo_defect_model
    if not os.path.exists(YOLO_DEFECT_MODEL_PATH):
        return None
    try:
        os.environ.setdefault(
            "YOLO_CONFIG_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         ".ultralytics"))
        from ultralytics import YOLO
        _yolo_defect_model = YOLO(YOLO_DEFECT_MODEL_PATH)
        return _yolo_defect_model
    except Exception:
        return None


def _load_yolo_model_cached(path, cache_holder):
    """按路径加载 YOLO 模型并缓存。cache_holder 是全局变量容器名(字符串)。"""
    global _yolo_defect_model, _yolo_defect_model2
    cached = _yolo_defect_model if cache_holder == "_yolo_defect_model" else _yolo_defect_model2
    if cached is not None:
        return cached
    if not os.path.exists(path):
        return None
    try:
        os.environ.setdefault(
            "YOLO_CONFIG_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         ".ultralytics"))
        from ultralytics import YOLO
        model = YOLO(path)
        if cache_holder == "_yolo_defect_model":
            _yolo_defect_model = model
        else:
            _yolo_defect_model2 = model
        return model
    except Exception:
        return None


def load_yolo_defect_model_v2():
    """加载第二缺陷模型 defect_detect_best_v2.pt（存在才返回）。"""
    if not os.path.exists(YOLO_DEFECT_MODEL_PATH2):
        return None
    return _load_yolo_model_cached(YOLO_DEFECT_MODEL_PATH2, "_yolo_defect_model2")


def _run_defect_model(model, img_bgr, conf_th, region):
    """单个缺陷模型推理，返回 [{bbox, confidence}]（原始坐标空间）。"""
    if model is None:
        return []
    try:
        if region is not None:
            x1, y1, x2, y2 = [int(v) for v in region]
            if x2 - x1 < 20 or y2 - y1 < 20:
                return []
            crop = img_bgr[y1:y2, x1:x2]
            r = model(crop, verbose=False, conf=conf_th)[0]
            detections = []
            if r.boxes is not None and len(r.boxes):
                for i in range(len(r.boxes)):
                    cx1, cy1, cx2, cy2 = r.boxes.xyxy[i].cpu().numpy()
                    detections.append({
                        "bbox": [int(cx1) + x1, int(cy1) + y1,
                                 int(cx2) + x1, int(cy2) + y1],
                        "confidence": float(r.boxes.conf[i]),
                    })
            return detections
        r = model(img_bgr, verbose=False, conf=conf_th)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        detections = []
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy().astype(int)
            detections.append({"bbox": [int(x1), int(y1), int(x2), int(y2)],
                               "confidence": float(r.boxes.conf[i])})
        return detections
    except Exception:
        return []


def _merge_detections(det_a, det_b, iou_th=0.5):
    """两个模型的检测并集 + NMS 去重，返回合并后的检测列表。"""
    if not det_b:
        return det_a
    if not det_a:
        return det_b
    all_det = det_a + det_b
    boxes = []
    scores = []
    for d in all_det:
        x1, y1, x2, y2 = d["bbox"]
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(d["confidence"])
    keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0,
                            nms_threshold=iou_th)
    if keep is None or len(keep) == 0:
        return all_det
    keep = list(keep)
    if isinstance(keep[0], (list, tuple)):
        keep = [k[0] for k in keep]
    merged = [all_det[i] for i in keep]
    merged.sort(key=lambda d: d["confidence"], reverse=True)
    return merged


def predict_with_yolo_defect(img_bgr: np.ndarray,
                             conf_th: float = 0.25,
                             region=None) -> list:
    """Run YOLO defect detection（默认双模型融合）。
    同时跑 defect_detect_best.pt 与 defect_detect_best_v2.pt（存在时），
    并集 + NMS 去重。region=(x1,y1,x2,y2) 可选先裁剪到苹果区域。
    Returns list of {'bbox': [x1,y1,x2,y2], 'confidence': float} 原始坐标，
    或 [] 表示无检出 / 模型缺失。"""
    model = load_yolo_defect_model()
    if model is None:
        return []
    det_a = _run_defect_model(model, img_bgr, conf_th, region)
    model2 = load_yolo_defect_model_v2()
    det_b = _run_defect_model(model2, img_bgr, conf_th, region) if model2 else []
    return _merge_detections(det_a, det_b)

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
    # _dl_transform 已含 Resize((384,384))，不要再额外缩一次，否则多损失一次细节
    img_pil = Image.fromarray(img_rgb)
    img_t = _dl_transform(img_pil).unsqueeze(0).to(device)
    
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
    heatmap = _gradcam_dl(img_bgr) if getattr(config, "ENABLE_GRADCAM", True) else None
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
