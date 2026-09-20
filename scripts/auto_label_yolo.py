# -*- coding: utf-8 -*-
"""
Auto-label apple detection images using the current YOLO model.
- Runs the model over candidate images
- High-confidence boxes (>=0.85) -> copied to yolo_pseudo/images+labels (auto-pass)
- Low-confidence boxes        -> drawn onto review sheets for eyeballing

Usage: python scripts/auto_label_yolo.py
"""
import os, sys, glob, random, math
os.environ["YOLO_CONFIG_DIR"] = r"E:\apple\.ultralytics"
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
from ultralytics import YOLO

def imread_u(p):
    d = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(d, cv2.IMREAD_COLOR)

def save_label(img, boxes, out_txt):
    h, w = img.shape[:2]
    lines = []
    for x1, y1, x2, y2, conf in boxes:
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(out_txt).write_text("\n".join(lines), encoding="utf-8")

def main():
    model = YOLO(r"E:\apple\yolo_detect_best.pt")
    random.seed(1)
    cands = []
    sources = [
        (r"E:\apple\dataset_collected\apple\合格", 800, "dc_good"),
        (r"E:\apple\dataset_collected\apple\不合格", 800, "dc_bad"),
        (r"E:\新数据集\archive (2)\apple_images\apple_real_images\fresh", 400, "a2_fresh"),
        (r"E:\新数据集\archive (2)\apple_images\apple_real_images\rot_defect", 300, "a2_rot"),
        (r"E:\新数据集\archive (3)\dataset\dataset\Dataset_4_clases\sanas_1000", 300, "a3_sanas"),
        (r"E:\新数据集\archive (3)\dataset\dataset\Dataset_4_clases\podrida", 300, "a3_podrida"),
    ]
    for folder, n, tag in sources:
        files = sorted(glob.glob(folder + r"\*"))
        if not files:
            continue
        pick = random.sample(files, min(n, len(files)))
        for f in pick:
            cands.append((f, tag))
    print(f"candidates: {len(cands)}")

    OUT_IMG = Path(r"E:\apple\yolo_pseudo\images")
    OUT_LBL = Path(r"E:\apple\yolo_pseudo\labels")
    REVIEW = Path(r"E:\apple\yolo_pseudo\review")
    for d in (OUT_IMG, OUT_LBL, REVIEW):
        d.mkdir(parents=True, exist_ok=True)
    stats = {"auto": 0, "review": 0, "noapple": 0}
    review_sheets = []

    for f, tag in cands:
        img = imread_u(f)
        if img is None:
            continue
        r = model.predict(img, conf=0.15, imgsz=640, verbose=False)[0]
        boxes = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            boxes = [(float(x1), float(y1), float(x2), float(y2), float(c))
                     for (x1, y1, x2, y2), c in zip(xyxy, confs)]
        if not boxes:
            stats["noapple"] += 1
            continue
        maxc = max(b[4] for b in boxes)
        name = Path(f).stem
        (OUT_IMG / (name + Path(f).suffix)).write_bytes(
            np.fromfile(str(f), dtype=np.uint8).tobytes())
        save_label(img, boxes, OUT_LBL / (name + ".txt"))
        if maxc >= 0.85:
            stats["auto"] += 1
        else:
            stats["review"] += 1
            view = img.copy()
            for (x1, y1, x2, y2, c) in boxes:
                color = (0, 255, 0) if c >= 0.85 else (0, 165, 255)
                cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                cv2.putText(view, f"{c:.2f}", (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            review_sheets.append((name, view))

    print("stats:", stats, "review sheets:", len(review_sheets))
    SHEET = 16
    for si in range(math.ceil(len(review_sheets) / SHEET)):
        chunk = review_sheets[si * SHEET:(si + 1) * SHEET]
        cell = 300
        canvas = np.full((4 * cell, 4 * cell, 3), 30, np.uint8)
        for i, (name, img) in enumerate(chunk):
            h, w = img.shape[:2]
            scale = min((cell - 10) / h, (cell - 10) / w)
            im2 = cv2.resize(img, (int(w * scale), int(h * scale)))
            rr, cc = divmod(i, 4)
            x0 = cc * cell + (cell - im2.shape[1]) // 2
            y0 = rr * cell + (cell - im2.shape[0]) // 2
            canvas[y0:y0 + im2.shape[0], x0:x0 + im2.shape[1]] = im2
            cv2.putText(canvas, name[:22], (cc * cell + 4, rr * cell + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        out_p = REVIEW / f"review_{si:03d}.png"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", canvas)
        if ok:
            buf.tofile(str(out_p))
    print("review sheets ->", REVIEW)
    print("auto labels ->", OUT_LBL)

if __name__ == "__main__":
    main()
