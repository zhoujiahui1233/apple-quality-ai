# -*- coding: utf-8 -*-
"""Dataset enrichment tools: quality check, dedup, subject crop, augmentation.

All functions are ASCII-safe internally; Chinese UI strings are produced by
the caller (app.py / frontend) based on the numeric scores returned here.
"""
import cv2
import numpy as np
from pathlib import Path

# ---------- quality ----------

def quality_check(img):
    """Return dict with score(0-100), blur_var, brightness, ok(bool)."""
    if img is None:
        return {"score": 0, "blur_var": 0.0, "brightness": 0.0, "ok": False}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    # blur: variance < 60 clearly blurry, > 400 sharp
    blur_score = min(100.0, max(0.0, blur_var / 4.0))
    # exposure: prefer 60~200 mean brightness
    if 60 <= brightness <= 200:
        expo_score = 100.0
    else:
        expo_score = max(0.0, 100.0 - abs(brightness - 130.0) * 0.8)
    score = int(0.6 * blur_score + 0.4 * expo_score)
    return {"score": score, "blur_var": round(blur_var, 1),
            "brightness": round(brightness, 1), "ok": score >= 40}


# ---------- perceptual hash dedup ----------

def dhash(img, size=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size))
    diff = small[:, 1:] > small[:, :-1]
    h = 0
    for bit in diff.flatten().astype(np.uint8):
        h = (h << 1) | int(bit)
    return h


def hamming(a, b):
    return bin(a ^ b).count("1")


def _iter_dataset_images(base_dir):
    base = Path(base_dir)
    if not base.exists():
        return
    for cat in sorted(p for p in base.iterdir() if p.is_dir()):
        if cat.name in ("unlabeled", "_skipped"):
            continue
        for label in sorted(p for p in cat.iterdir() if p.is_dir()):
            for f in sorted(label.iterdir()):
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    yield f


def find_duplicates(img, base_dir, max_results=3, threshold=0.86):
    """Compare img against dataset images via dHash.

    Returns {"duplicates": [{"name","sim"}], "max_sim": float}
    """
    if img is None:
        return {"duplicates": [], "max_sim": 0.0}
    try:
        h_query = dhash(img)
    except Exception:
        return {"duplicates": [], "max_sim": 0.0}
    hits = []
    for f in _iter_dataset_images(base_dir):
        try:
            data = np.fromfile(str(f), dtype=np.uint8)
            if data.size == 0:
                continue
            other = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if other is None:
                continue
            sim = 1.0 - hamming(h_query, dhash(other)) / 64.0
            if sim >= threshold:
                hits.append((sim, f.name))
        except Exception:
            continue
    hits.sort(reverse=True)
    return {"duplicates": [{"name": n, "sim": round(s, 3)} for s, n in hits[:max_results]],
            "max_sim": round(hits[0][0], 3) if hits else 0.0}


# ---------- subject crop ----------

def crop_subject(img, category):
    """Crop the main subject region. Returns (cropped_img, ratio) or (None, 0.0)."""
    if img is None:
        return None, 0.0
    hh, ww = img.shape[:2]
    if category == "apple":
        try:
            from apple_defect_detector import segment_apple
            mask, contour = segment_apple(img)
            if mask is None or contour is None or len(contour) == 0:
                return None, 0.0
            x, y, w, h = cv2.boundingRect(contour)
            if w < 10 or h < 10:
                return None, 0.0
            pad = int(max(w, h) * 0.05)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(ww, x + w + pad), min(hh, y + h + pad)
            ratio = float(np.sum(mask > 0)) / float(hh * ww)
            return img[y0:y1, x0:x1], round(ratio, 3)
        except Exception:
            return None, 0.0
    # bread / beverage: keep central 80%
    y0, y1 = int(hh * 0.1), int(hh * 0.9)
    x0, x1 = int(ww * 0.1), int(ww * 0.9)
    return img[y0:y1, x0:x1], 0.64


def save_cropped(crop_img, dst_path, target_size=480):
    """Pad crop to a square and resize to target_size for consistent training."""
    if crop_img is None:
        return False
    h, w = crop_img.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 255, np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop_img
    if side != target_size:
        canvas = cv2.resize(canvas, (target_size, target_size),
                            interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return False
    buf.tofile(str(dst_path))
    return True


# ---------- augmentation ----------

def augment(img, n=4):
    """Return up to n variants (original + flips/rotation/brightness)."""
    if img is None or n < 1:
        return []
    outs = [img]
    if n > 1:
        outs.append(cv2.flip(img, 1))                      # horizontal flip
    if n > 2:
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 4.0, 1.0)
        outs.append(cv2.warpAffine(img, m, (w, h),
                                   borderMode=cv2.BORDER_REPLICATE))
    if n > 3:
        outs.append(cv2.convertScaleAbs(img, alpha=0.88, beta=12))
    if n > 4:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * 0.85, 0, 255)
        outs.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    return outs[:n]


def save_augmented(img, stem, dst_dir, n=4):
    """Save n augmented variants, return list of saved filenames."""
    saved = []
    for i, variant in enumerate(augment(img, n)):
        fname = f"{stem}_aug{i}.jpg"
        ok, buf = cv2.imencode(".jpg", variant, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            buf.tofile(str(Path(dst_dir) / fname))
            saved.append(fname)
    return saved
