# -*- coding: utf-8 -*-
"""
Defect sample augmentation & realistic synthesis tool
======================================================
Purpose: relieve the shortage of defective (unqualified) apple samples.
It ONLY processes the defect folder (unqualified); by default results go to a
separate output folder and never touch the validation set. Use --merge to
copy them back into the original defect folder when you are ready.

Two augmentation modes:
  1) basic      - flip / rotate / brightness-contrast / light noise (keeps the
                  defect nature unchanged, purely more variants)
  2) synthetic  - cut real defect texture from defect photos and feather-blend
                  it onto a good apple photo, so we get NEW "good apple with a
                  realistic defect" images (not flat color blocks).

Usage:
  python scripts/augment_defects.py
  python scripts/augment_defects.py --mode synthetic --per-image 6
  python scripts/augment_defects.py --merge
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def imread_unicode(path):
    """cv2.imread wrapper that supports non-ASCII (Chinese) file paths."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img, params=None):
    """cv2.imwrite wrapper that supports non-ASCII (Chinese) file paths."""
    ext = Path(str(path)).suffix or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if ok:
        buf.tofile(str(path))
    return ok


# ------------------- defect texture extraction -------------------

def extract_defect_mask(img):
    """Extract defect mask (brown bruises / dark rot) from a defect photo.

    Adaptive approach: within the apple region, pixels that deviate strongly
    from the apple's own median color (saturation / value / hue distance) are
    considered defects, with highlight pixels excluded. This adapts to
    different shooting conditions better than fixed HSV thresholds.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    am = apple_region_mask(img)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    h = hsv[:, :, 0].astype(np.float32)
    ap_s, ap_v, ap_h = s[am > 0], v[am > 0], h[am > 0]
    if len(ap_s) < 200:
        return np.zeros(img.shape[:2], np.uint8)
    s_med, v_med, h_med = np.median(ap_s), np.median(ap_v), np.median(ap_h)
    hd = np.minimum(np.abs(h - h_med), 180.0 - np.abs(h - h_med))
    dist = np.abs(s - s_med) + np.abs(v - v_med) + hd * 0.8
    thr = np.percentile(dist[am > 0], 88)
    mask = ((dist > thr) & (am > 0) & (v < 235) & (s > 25)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return mask


def biggest_defect_patches(img, mask, max_patches=3, min_area=400,
                           max_ratio=0.18):
    """Return up to max_patches (patch, patch_mask, bbox) sorted by area desc."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    img_area = img.shape[0] * img.shape[1]
    comps = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area and area / img_area <= max_ratio:
            comps.append((area, i))
    comps.sort(reverse=True)
    patches = []
    for area, i in comps[:max_patches]:
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        x0, y0 = max(0, x - 6), max(0, y - 6)
        x1 = min(img.shape[1], x + w + 6)
        y1 = min(img.shape[0], y + h + 6)
        patch = img[y0:y1, x0:x1].copy()
        pmask = mask[y0:y1, x0:x1].copy()
        patches.append((patch, pmask, (x0, y0, x1 - x0, y1 - y0)))
    return patches


def apple_region_mask(img):
    """Rough apple region mask (reuse the HSV ranges from config.py)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2], np.uint8)
    for lo, hi in [
        ((0, 20, 30), (10, 255, 255)),
        ((156, 20, 30), (180, 255, 255)),
        ((30, 20, 30), (85, 255, 255)),
        ((15, 20, 80), (35, 255, 255)),
    ]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    return mask


def match_brightness(patch, target):
    """Linearly scale patch brightness so it fits the target region."""
    src = patch.mean()
    dst = target.mean()
    if src < 1:
        return patch
    scale = float(np.clip(dst / src, 0.6, 1.6))
    return np.clip(patch.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def feather_blend(base, patch, pmask, x, y, rng):
    """Blend patch onto base at (x, y) with a feathered mask + brightness match."""
    ph, pw = patch.shape[:2]
    bh, bw = base.shape[:2]
    x = int(np.clip(x, 0, bw - pw))
    y = int(np.clip(y, 0, bh - ph))
    roi = base[y:y + ph, x:x + pw]
    patch_m = match_brightness(patch, roi)
    # slight local contrast / color jitter on the pasted defect for realism
    if rng.random() < 0.6:
        alpha = rng.uniform(0.9, 1.12)
        beta = rng.uniform(-8, 8)
        patch_m = cv2.convertScaleAbs(patch_m, alpha=alpha, beta=beta)
    m = cv2.GaussianBlur(pmask, (0, 0), sigmaX=3)
    m3 = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    blended = (patch_m.astype(np.float32) * m3
               + roi.astype(np.float32) * (1.0 - m3))
    base[y:y + ph, x:x + pw] = np.clip(blended, 0, 255).astype(np.uint8)
    return base


def pick_pos_in_region(rng, region_mask, pw, ph, bw, bh):
    """Random position inside the (approximate) apple region."""
    ys, xs = np.where(region_mask > 0)
    if len(xs) < 50:
        return rng.randint(0, max(1, bw - pw)), rng.randint(0, max(1, bh - ph))
    cx, cy = int(xs.mean()), int(ys.mean())
    rx = max(int((xs.max() - xs.min()) * 0.45), pw)
    ry = max(int((ys.max() - ys.min()) * 0.45), ph)
    x = int(np.clip(rng.randint(int(cx - rx), int(cx + rx)), 0, bw - pw))
    y = int(np.clip(rng.randint(int(cy - ry), int(cy + ry)), 0, bh - ph))
    return x, y


# ------------------- basic augmentation -------------------

def basic_augment(img, rng):
    """Random flip / rotation / brightness / contrast / light noise."""
    out = img.copy()
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    angle = rng.uniform(-18, 18)
    h, w = out.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    alpha = rng.uniform(0.75, 1.25)
    beta = rng.uniform(-20, 20)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if rng.random() < 0.5:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16)
                        + rng.randint(-8, 8)).clip(0, 179).astype(np.uint8)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32)
                               * rng.uniform(0.9, 1.1), 0, 255).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if rng.random() < 0.4:
        noise = np.random.normal(0, rng.uniform(2, 6), out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


def synthesize_defect(base_img, defect_patches, rng):
    """Paste 1-2 real defect textures onto a good apple photo."""
    out = base_img.copy()
    amask = apple_region_mask(out)
    n = min(rng.randint(1, 2), len(defect_patches))
    used = rng.sample(defect_patches, n)
    base_area = out.shape[0] * out.shape[1]
    for patch, pmask, _ in used:
        ph, pw = patch.shape[:2]
        # keep the pasted defect small relative to the target image
        max_pw = int(np.sqrt(base_area) * 0.30)
        scale = min(rng.uniform(0.7, 1.3), max(0.4, max_pw / max(pw, 1)))
        if scale != 1.0:
            patch = cv2.resize(patch, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)
            pmask = cv2.resize(pmask, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)
        ph, pw = patch.shape[:2]
        if pw < 12 or ph < 12 or pw >= out.shape[1] or ph >= out.shape[0]:
            continue
        x, y = pick_pos_in_region(rng, amask, pw, ph, out.shape[1], out.shape[0])
        out = feather_blend(out, patch, pmask, x, y, rng)
    return out


def synthesize_from_annotations(base_img, ann_defects, rng):
    """Paste real defect textures (cropped from user-annotated boxes) onto a
    good apple photo. ann_defects: list of (patch_img, patch_mask)."""
    out = base_img.copy()
    amask = apple_region_mask(out)
    n = min(rng.randint(1, 2), len(ann_defects))
    used = rng.sample(ann_defects, n)
    base_area = out.shape[0] * out.shape[1]
    for patch, pmask in used:
        ph, pw = patch.shape[:2]
        max_pw = int(np.sqrt(base_area) * 0.28)
        scale = min(rng.uniform(0.8, 1.4), max(0.4, max_pw / max(pw, 1)))
        if scale != 1.0:
            patch = cv2.resize(patch, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)
            pmask = cv2.resize(pmask, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)
        ph, pw = patch.shape[:2]
        if pw < 12 or ph < 12 or pw >= out.shape[1] or ph >= out.shape[0]:
            continue
        x, y = pick_pos_in_region(rng, amask, pw, ph, out.shape[1], out.shape[0])
        out = feather_blend(out, patch, pmask, x, y, rng)
    return out


def load_annotation_defects(ann_path, defect_root, max_side):
    """Read defect-box JSON, crop textures, return list of (patch, mask)."""
    ann_path = Path(ann_path)
    if not ann_path.exists():
        print(f"[error] annotations not found: {ann_path}")
        return None
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    defects = []
    for img_name, boxes in ann.items():
        img_path = defect_root / img_name
        if not img_path.exists():
            img_path = next((p for p in defect_root.rglob(img_name)
                             if p.is_file()), None)
        if img_path is None:
            continue
        img = imread_unicode(img_path)
        if img is None:
            continue
        img = resize_longest(img, max_side)
        h, w = img.shape[:2]
        for (x1, y1, x2, y2) in boxes:
            x1 = max(0, min(int(x1), w)); y1 = max(0, min(int(y1), h))
            x2 = max(0, min(int(x2), w)); y2 = max(0, min(int(y2), h))
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            patch = img[y1:y2, x1:x2].copy()
            # mask = anything inside the box that is not pure background edge;
            # simplest robust approach: full box with soft edge via feather.
            pmask = np.full(patch.shape[:2], 255, np.uint8)
            defects.append((patch, pmask))
    print(f"loaded {len(defects)} defect textures from annotations")
    return defects or None


# ------------------- main -------------------

def collect_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in Path(root).rglob("*")
                  if p.is_file() and p.suffix.lower() in exts)


def resize_longest(img, max_side=1024):
    """Downscale so the longest side <= max_side (keeps aspect ratio)."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def locate_apple_ellipse(img):
    """Locate the apple region via HSV color segmentation + ellipse fitting.
    Returns (ellipse_params, ellipse_mask) or (None, None)."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((h, w), np.uint8)
    for lo, hi in [
        ((0, 20, 30), (10, 255, 255)),
        ((156, 20, 30), (180, 255, 255)),
        ((30, 20, 30), (85, 255, 255)),
        ((15, 20, 80), (35, 255, 255)),
    ]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None, None
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n)]
    bi = int(np.argmax(areas)) + 1
    comp = (labels == bi).astype(np.uint8) * 255
    pts = np.column_stack(np.where(comp > 0))[:, ::-1]
    if len(pts) < 20:
        return None, None
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(pts)
    em = np.zeros((h, w), np.uint8)
    cv2.ellipse(em, (int(cx), int(cy)), (int(MA / 2), int(ma / 2)),
                ang, 0, 360, 255, -1)
    return (int(cx), int(cy), int(MA / 2), int(ma / 2), float(ang)), em


def locate_apple_mask(img):
    """Keep old name as a thin wrapper returning only the mask."""
    _, em = locate_apple_ellipse(img)
    return em


def select_defect_boxes(img_path, max_side=1024):
    """Interactive defect box selection (multiple boxes per image supported).

    - left-drag: draw a box around a defect; release to confirm it
    - keep drawing more boxes for the same image as needed
    - 'n' / space / Enter: finish this image, move to the next one
    - 'u' / Backspace: undo the last box on this image
    - 'd': clear all boxes of this image
    - 'q' / Esc: quit the whole annotation session (keep what is already done)

    Returns {image_name: [(x1, y1, x2, y2), ...]} or None when the user quits.
    """
    img = imread_unicode(img_path)
    if img is None:
        print(f"  [skip] cannot read {img_path}")
        return None
    img = resize_longest(img, max_side)
    boxes = []
    current = None
    drawing = False

    win_name = "mark defects: drag=box n=next u=undo d=clear q=quit"

    def on_mouse(event, x, y, flags, param):
        nonlocal current, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x1, y1, _, _ = current
            current = (x1, y1, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if current is not None:
                x1, y1, x2, y2 = current
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))
                if x2 - x1 > 8 and y2 - y1 > 8:
                    boxes.append((x1, y1, x2, y2))
            current = None

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)
    quit_session = False
    while not quit_session:
        view = img.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if current is not None:
            x1, y1, x2, y2 = current
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 255), 2)
        info = f"{len(boxes)} box(es): n=next u=undo d=clear q=quit"
        cv2.putText(view, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2)
        cv2.imshow(win_name, view)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("n"), ord(" "), 13):
            quit_session = True
        elif k in (ord("u"), 8):
            if boxes:
                boxes.pop()
        elif k in (ord("d"), ord("r")):
            boxes = []
        elif k in (ord("q"), 27):
            return None
    cv2.destroyWindow(win_name)
    return {Path(str(img_path)).name: boxes}


def annotate_mode(defect_imgs, ann_path, max_side, redo=False):
    """Run interactive annotation over defect photos, save JSON."""
    ann = {}
    if ann_path.exists():
        try:
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:
            ann = {}
    for p in defect_imgs:
        if not redo and p.name in ann and ann[p.name]:
            print(f"  [skip] already annotated: {p.name}")
            continue
        print(f"  annotate: {p.name}")
        res = select_defect_boxes(p, max_side)
        if res is None:
            print("  user quit; saving what is done so far")
            break
        if res:
            ann.update(res)
    ann_path.write_text(json.dumps(ann, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    n_total = sum(len(v) for v in ann.values())
    print(f"annotations saved: {len(ann)} images, {n_total} boxes -> {ann_path}")


def main():
    ap = argparse.ArgumentParser(description="Defect sample augmentation tool")
    ap.add_argument("--defect-dir", default=r"E:\apple\dataset_collected\apple\不合格")
    ap.add_argument("--good-dir", default=r"E:\apple\dataset_collected\apple\合格")
    ap.add_argument("--output", default=r"E:\apple\dataset_collected_augmented\apple\不合格")
    ap.add_argument("--mode", choices=["all", "basic", "synthetic"], default="all")
    ap.add_argument("--per-image", type=int, default=5,
                    help="generated variants per defect photo")
    ap.add_argument("--max-side", type=int, default=1024,
                    help="downscale photos so longest side <= this (speed/memory)")
    ap.add_argument("--preview", action="store_true",
                    help="only visualize extracted defect masks on the defect "
                         "photos (no augmentation), then exit")
    ap.add_argument("--annotate", action="store_true",
                    help="interactively mark defect boxes on the defect photos "
                         "(draw box, n=next, r=redo, q=quit); saved to JSON")
    ap.add_argument("--redo", action="store_true",
                    help="with --annotate: re-annotate every defect photo, "
                         "overwriting previous boxes")
    ap.add_argument("--annotations", default=None,
                    help="path to defect-box JSON (default: "
                         "<output>/_defect_boxes.json)")
    ap.add_argument("--ann-preview", action="store_true",
                    help="draw saved annotation boxes onto the defect photos "
                         "and save preview images (no augmentation)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge", action="store_true",
                    help="copy generated samples back into defect-dir (careful)")
    args = ap.parse_args()

    defect_root = Path(args.defect_dir)
    good_root = Path(args.good_dir)
    out_root = Path(args.output)
    if not defect_root.is_dir():
        print(f"[error] defect dir not found: {defect_root}")
        return

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    defect_imgs = collect_images(defect_root)
    good_imgs = collect_images(good_root)
    print(f"defect photos: {len(defect_imgs)}")
    print(f"good photos:   {len(good_imgs)}")
    print(f"mode: {args.mode}, per-image: {args.per_image}")

    if args.annotate:
        ann_path = Path(args.annotations) if args.annotations else \
            Path(args.output).parent / "_defect_boxes.json"
        annotate_mode(defect_imgs, ann_path, args.max_side, redo=args.redo)
        return

    if args.ann_preview:
        ann_path = Path(args.annotations) if args.annotations else \
            Path(args.output).parent / "_defect_boxes.json"
        if not ann_path.exists():
            print(f"[error] annotations not found: {ann_path}")
            return
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        prev_root = Path(args.output).parent / "_ann_preview"
        prev_root.mkdir(parents=True, exist_ok=True)
        for img_name, boxes in ann.items():
            img_path = next((p for p in defect_root.rglob(img_name)
                             if p.is_file()), None)
            if img_path is None:
                continue
            img = imread_unicode(img_path)
            if img is None:
                continue
            img = resize_longest(img, args.max_side)
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 0, 255), 3)
            out_path = prev_root / f"{Path(img_name).stem}__ann.png"
            imwrite_unicode(out_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            print(f"preview {img_name}: {len(boxes)} boxes -> {out_path}")
        print("annotation previews done ->", prev_root)
        return

    ann_path = Path(args.annotations) if args.annotations else \
        Path(args.output).parent / "_defect_boxes.json"
    ann_defects = None
    if ann_path.exists():
        ann_defects = load_annotation_defects(ann_path, defect_root,
                                              args.max_side)
        if ann_defects is None:
            print("[warn] annotation loading failed, fall back to auto masks")

    if args.preview:
        prev_root = Path(args.output).parent / "_preview"
        prev_root.mkdir(parents=True, exist_ok=True)
        for p in defect_imgs:
            img = imread_unicode(p)
            if img is None:
                continue
            img = resize_longest(img, args.max_side)
            mask = extract_defect_mask(img)
            overlay = img.copy()
            overlay[mask > 0] = (0, 0, 255)  # red = detected defect
            side = np.concatenate([img, overlay], axis=1)
            out_path = prev_root / f"{p.stem}__mask.png"
            imwrite_unicode(out_path, side, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            pct = (mask > 0).sum() / mask.size * 100
            print(f"preview {p.name}: defect area {pct:.1f}% -> {out_path}")
        print("previews done ->", prev_root)
        return

    # pre-extract defect textures once
    if ann_defects:
        defect_patches = ann_defects
        print(f"using annotated defect textures: {len(defect_patches)}")
    else:
        defect_patches = []
        for p in defect_imgs:
            img = imread_unicode(p)
            if img is None:
                continue
            img = resize_longest(img, args.max_side)
            mask = extract_defect_mask(img)
            defect_patches.extend(biggest_defect_patches(img, mask))
        print(f"auto defect textures: {len(defect_patches)}")
        if not defect_patches:
            print("[warn] no defect texture found - try annotate mode")

    if not good_imgs and args.mode in ("all", "synthetic"):
        print("[warn] no good photos -> synthetic mode disabled")

    out_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for p in defect_imgs:
        img = imread_unicode(p)
        if img is None:
            continue
        img = resize_longest(img, args.max_side)
        for k in range(args.per_image):
            mode = args.mode
            if mode == "all":
                mode = rng.choice(["basic", "synthetic"] if good_imgs else ["basic"])
            if mode == "basic":
                aug = basic_augment(img, rng)
            else:
                if not good_imgs or not defect_patches:
                    aug = basic_augment(img, rng)
                else:
                    base = imread_unicode(rng.choice(good_imgs))
                    if base is None:
                        aug = basic_augment(img, rng)
                    else:
                        base = resize_longest(base, args.max_side)
                        if ann_defects:
                            aug = synthesize_from_annotations(
                                base, defect_patches, rng)
                        else:
                            aug = synthesize_defect(base, defect_patches, rng)
            suffix = "_aug.png"
            out_path = out_root / f"{p.stem}__{mode}{k}{suffix}"
            imwrite_unicode(out_path, aug, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            total += 1
    print(f"generated {total} augmented samples -> {out_root}")

    if args.merge and total:
        n = 0
        for f in out_root.iterdir():
            if f.is_file():
                shutil.copy2(str(f), defect_root / f.name)
                n += 1
        print(f"merged {n} samples into defect dir: {defect_root}")


if __name__ == "__main__":
    main()
