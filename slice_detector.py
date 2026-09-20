# -*- coding: utf-8 -*-
"""苹果切片（鲜切苹果）视觉褐变检测模块。

独立于整果缺陷检测：输入切片照片，自动分割切片区域，
统计褐变面积占比，输出褐变指数 BI、褐变等级（1~5 级）和标注图。

检测引擎：
  - rule：内置 OpenCV 规则算法（无需模型文件，实时可用）
  - unet：U-Net 分割模型（models/slice_unet.pt，由 scripts/train_slice_unet.py 训练）
  - auto：有模型用模型，没有则回退规则算法（默认）

用法（Web 端）：由 app.py 的 /api/slice/detect 调用。
用法（命令行）：python slice_detector.py <图片路径> [--engine rule|unet|auto] [--out 标注图路径]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "models" / "slice_unet.pt"

# 褐变等级表：[(阈值占比, 等级, 状态名, 建议)]，占比 >= 阈值则取该级
GRADE_DEFS = [
    (0.05, 1, "特鲜", "无明显褐变，可作鲜切销售"),
    (0.15, 2, "轻微褐变", "局部褐变，当日可售"),
    (0.30, 3, "中度褐变", "褐变明显，建议降价或深加工"),
    (0.50, 4, "明显褐变", "不宜鲜食，建议加工处理"),
    (1.01, 5, "严重褐变", "已不可逆劣变，建议废弃"),
]

# 分割类别：0 背景 / 1 果肉 / 2 果核 / 3 褐变 / 4 果皮
SEG_BG, SEG_FLESH, SEG_CORE, SEG_BROWN, SEG_PEEL = 0, 1, 2, 3, 4

_unet_handle = {"model": None, "path": None, "err": None}


def _flesh_mask(hsv: np.ndarray) -> np.ndarray:
    """果肉：白/淡黄/淡绿（H 15~60，低饱和，高亮度）"""
    lower = np.array([15, 0, 90], dtype=np.uint8)
    upper = np.array([60, 120, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def _peel_mask(hsv: np.ndarray) -> np.ndarray:
    """果皮：红（两端）+ 绿/黄绿"""
    red1 = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([168, 60, 50]), np.array([180, 255, 255]))
    green = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([90, 255, 255]))
    return cv2.bitwise_or(cv2.bitwise_or(red1, red2), green)


def _brown_mask(hsv: np.ndarray) -> np.ndarray:
    """褐变：棕褐色（H 8~30，中高饱和，中低亮度）"""
    lower = np.array([8, 50, 40], dtype=np.uint8)
    upper = np.array([30, 255, 230], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def _clean_connected(mask: np.ndarray, min_frac: float = 0.0008) -> np.ndarray:
    """去噪点；剔除接触图像边缘的大块背景区域。"""
    h, w = mask.shape
    k = max(3, int(min(h, w) * 0.01) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    total = h * w
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < total * min_frac:
            continue
        # 大块背景：接触图像边缘且面积占比过大
        touches_edge = (x == 0 or y == 0 or x + bw >= w or y + bh >= h)
        if touches_edge and area > total * 0.30:
            continue
        keep[labels == i] = 255
    return keep


def _seed_mask(hsv: np.ndarray) -> np.ndarray:
    """Seeds / very dark impurities: low brightness + mid-high saturation dark dots, exclude from browning."""
    lower = np.array([0, 40, 30], dtype=np.uint8)
    upper = np.array([179, 255, 95], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def _slice_geometry(slice_mask: np.ndarray):
    """Slice geometry: return ((cx, cy), radius) from min enclosing circle (fallback to centroid)."""
    ys, xs = np.nonzero(slice_mask)
    if len(xs) == 0:
        return None
    contours, _ = cv2.findContours(slice_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        (cx, cy), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
        return (float(cx), float(cy)), float(max(r, 1.0))
    cx, cy = float(xs.mean()), float(ys.mean())
    r = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).mean() * 1.6)
    return (cx, cy), max(r, 1.0)


def _core_mask(shape, center, R, core_frac: float = 0.30) -> np.ndarray:
    """Core exclusion zone: central circle (core + seeds area)."""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    return (dist <= core_frac * R).astype(np.uint8) * 255


def _ring_stats(shape, center, R, flesh_mask, brown,
                core_frac: float = 0.30, outer_frac: float = 0.92) -> dict:
    """Per-ring browning stats: inner / mid / outer annuli between core and peel ring."""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    bands = {"inner": (core_frac, 0.55), "mid": (0.55, 0.75), "outer": (0.75, outer_frac)}
    out = {}
    for name, (lo, hi) in bands.items():
        band = (dist >= lo * R) & (dist < hi * R)
        fb = band & (flesh_mask > 0)
        bb = band & (brown > 0)
        fpx = int(np.count_nonzero(fb))
        bpx = int(np.count_nonzero(bb))
        out[name] = {"flesh_px": fpx, "brown_px": bpx,
                     "ratio": round(bpx / fpx * 100.0, 2) if fpx > 0 else 0.0}
    return out


def rule_segmentation(img_bgr: np.ndarray) -> np.ndarray:
    """规则算法 → 5 类分割掩膜（uint8：0 背景/1 果肉/2 果核/3 褐变/4 果皮）。

    果核=中心 30% 半径圆 + 深色种子；果皮=外圈颜色带；褐变只统计果肉区。
    该掩膜既用于直接分析，也作为 U-Net 训练的伪标签起点。
    """
    h, w = img_bgr.shape[:2]
    seg = np.zeros((h, w), dtype=np.uint8)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    flesh = _flesh_mask(hsv)
    peel = _peel_mask(hsv)
    brown = _brown_mask(hsv)
    seeds = _seed_mask(hsv)

    slice_mask = cv2.bitwise_or(cv2.bitwise_or(flesh, peel), brown)
    slice_mask = _clean_connected(slice_mask)
    geom = _slice_geometry(slice_mask)
    if geom is None:
        return seg
    center, R = geom

    core = _core_mask(img_bgr.shape, center, R, 0.30)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    outer_ring = (dist > 0.92 * R).astype(np.uint8) * 255

    # 褐变：果肉区内（排除果核圆、外圈果皮环、种子；不排除果皮色带，
    # 因为 H 8~10 的深红褐变与红果皮范围重叠，重叠区按褐变处理更合理）
    brown_final = cv2.bitwise_and(brown, brown, mask=slice_mask)
    brown_final = cv2.bitwise_and(brown_final, cv2.bitwise_not(core))
    brown_final = cv2.bitwise_and(brown_final, cv2.bitwise_not(outer_ring))
    brown_final = cv2.bitwise_and(brown_final, cv2.bitwise_not(seeds))

    seg[slice_mask > 0] = SEG_FLESH          # 默认果肉
    seg[peel > 0] = SEG_PEEL                 # 果皮
    seg[core > 0] = SEG_CORE                 # 果核圆
    seg[(seeds > 0) & (slice_mask > 0)] = SEG_CORE   # 种子归入果核（仅切片内，不吞背景）
    seg[brown_final > 0] = SEG_BROWN         # 褐变（最后覆盖）
    return seg


def segment_stats(seg: np.ndarray, img_bgr: np.ndarray) -> dict:
    """从 5 类分割掩膜计算褐变指标（与规则算法输出结构一致）。"""
    h, w = seg.shape[:2]
    flesh = seg == SEG_FLESH
    core = seg == SEG_CORE
    brown = seg == SEG_BROWN
    peel = seg == SEG_PEEL
    slice_mask = ((seg >= SEG_FLESH) & (seg <= SEG_PEEL)).astype(np.uint8) * 255

    flesh_px = int(np.count_nonzero(flesh))
    brown_px = int(np.count_nonzero(brown))
    ratio = brown_px / flesh_px if flesh_px > 0 else 0.0
    bi_pct = round(ratio * 100.0, 2)

    # Standard browning index BI (Palou 1999, L*a*b*) on flesh area
    bi_lab = None
    if flesh_px > 100:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0] / 255.0 * 100.0
        A = lab[:, :, 1] - 128.0
        B = lab[:, :, 2] - 128.0
        m = flesh > 0
        Lm = float(L[m].mean())
        Am = float(A[m].mean())
        Bm = float(B[m].mean())
        denom = 5.645 * Lm + Am - 3.012 * Bm
        if abs(denom) > 1e-6:
            x = (Am + 1.75 * Lm) / denom
            bi_lab = round(100.0 * (x - 0.31) / 0.172, 2)

    # 环形统计（以切片几何中心为圆心）
    geom = _slice_geometry(slice_mask)
    ring = {}
    if geom is not None:
        ring = _ring_stats((h, w), geom[0], geom[1], flesh.astype(np.uint8),
                           brown.astype(np.uint8))

    core_px = int(np.count_nonzero(core & (slice_mask > 0)))

    grade, status, advice = 1, GRADE_DEFS[0][2], GRADE_DEFS[0][3]
    for thr, g, st, ad in GRADE_DEFS:
        if ratio >= thr:
            grade, status, advice = g, st, ad

    return {
        "slice_px": int(np.count_nonzero(slice_mask)),
        "flesh_px": flesh_px,
        "core_px": core_px,
        "seed_px": 0,                      # 种子已并入果核类
        "browning_px": brown_px,
        "browning_ratio": round(ratio * 100.0, 2),   # browning % on FLESH area
        "bi_pct": bi_pct,
        "bi_lab": bi_lab,
        "grade": grade,
        "status": status,
        "advice": advice,
        "ring": ring,
    }


def unet_available() -> bool:
    """切片 U-Net 模型是否存在且可加载（懒加载，避免拖慢启动）。"""
    if not MODEL_PATH.exists():
        return False
    try:
        if _unet_handle["model"] is None or _unet_handle["path"] != str(MODEL_PATH):
            import sys as _sys
            import torch
            root = str(Path(__file__).resolve().parent)
            if root not in _sys.path:
                _sys.path.insert(0, root)
            from scripts.slice_unet_model import load_checkpoint
            model, _payload = load_checkpoint(MODEL_PATH, device="cpu")
            _unet_handle["model"] = model
            _unet_handle["path"] = str(MODEL_PATH)
            _unet_handle["err"] = None
        return True
    except Exception as e:  # noqa: BLE001
        _unet_handle["err"] = str(e)
        return False


def unet_predict(img_bgr: np.ndarray) -> np.ndarray:
    """U-Net 推理：返回与原图同尺寸的 5 类分割掩膜。"""
    import torch
    import torch.nn.functional as F
    if not unet_available():
        return None
    model = _unet_handle["model"]
    h, w = img_bgr.shape[:2]
    max_dim = 320
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale < 1.0:
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        small = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        small, nh, nw = img_bgr, h, w
    x = torch.from_numpy(small[:, :, ::-1].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        seg = torch.argmax(logits, dim=1).squeeze(0).numpy().astype(np.uint8)
    if (nh, nw) != (h, w):
        seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
    return seg


def analyze_slice(img_bgr: np.ndarray, engine: str = "auto") -> dict:
    """Analyze a fresh-cut apple slice photo.

    engine: 'rule'（规则算法）/ 'unet'（U-Net 分割模型）/ 'auto'（有模型用模型，否则规则）。
    输出包含 engine / engine_key，标注图叠加了引擎名称。
    """
    if img_bgr is None:
        raise ValueError("image read failed")
    img = img_bgr.copy()

    if engine in ("unet", "auto") and unet_available():
        seg = unet_predict(img)
        if seg is None:
            seg = rule_segmentation(img)
        engine_key, engine_name = "unet", "U-Net 分割模型"
    else:
        seg = rule_segmentation(img)
        engine_key, engine_name = "rule", "规则算法"

    res = segment_stats(seg, img)
    res["engine"] = engine_name
    res["engine_key"] = engine_key

    h, w = img.shape[:2]
    slice_mask = ((seg >= SEG_FLESH) & (seg <= SEG_PEEL)).astype(np.uint8) * 255
    geom = _slice_geometry(slice_mask)
    center, R = (geom[0], geom[1]) if geom is not None else (None, None)
    res["annotated"] = _draw_annotation(img, seg, res, center, R)
    return res


def _draw_annotation(img: np.ndarray, seg: np.ndarray, res: dict,
                     center=None, R=None) -> np.ndarray:
    """Draw annotation: dim background, red browning, slice contour,
    core/ring markers, two-line info bar."""
    out = img.copy()
    h, w = out.shape[:2]
    slice_mask = ((seg >= SEG_FLESH) & (seg <= SEG_PEEL)).astype(np.uint8) * 255
    brown = (seg == SEG_BROWN).astype(np.uint8) * 255

    # Dim background
    dark = cv2.addWeighted(out, 0.35, np.zeros_like(out), 0, 0)
    out = np.where(slice_mask[:, :, None] > 0, out, dark)

    # Browning red highlight
    overlay = out.copy()
    overlay[brown > 0] = (0, 0, 255)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    # Slice contour
    contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 200, 0), max(2, int(min(h, w) * 0.004)))

    # Core circle + ring boundaries
    if center is not None and R is not None:
        cx, cy = int(center[0]), int(center[1])
        lw1 = max(1, int(min(h, w) * 0.002))
        lw2 = max(1, int(min(h, w) * 0.001))
        cv2.circle(out, (cx, cy), max(2, int(0.30 * R)), (0, 212, 255), lw1)   # core (cyan)
        cv2.circle(out, (cx, cy), max(2, int(0.55 * R)), (255, 255, 255), lw2) # inner/mid
        cv2.circle(out, (cx, cy), max(2, int(0.75 * R)), (255, 255, 255), lw2) # mid/outer
        cv2.circle(out, (cx, cy), max(2, int(0.92 * R)), (0, 255, 136), lw1)   # peel ring (green)

    # Two-line info bar
    bar_h = max(64, int(h * 0.09))
    cv2.rectangle(out, (0, 0), (w, bar_h), (10, 16, 32), -1)
    cv2.line(out, (0, bar_h), (w, bar_h), (0, 212, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(w * 0.001, 0.95))
    ratio = res.get("browning_ratio", 0.0)
    grade = res.get("grade", 1)
    status = res.get("status", "")
    engine = res.get("engine", "")
    text1 = f"flesh browning {ratio:.1f}% | grade {grade} | {status}"
    cv2.putText(out, text1, (8, int(bar_h * 0.38)), font, scale, (255, 255, 255), 2, cv2.LINE_AA)
    ring = res.get("ring") or {}
    if ring:
        rt = "rings  in {}% mid {}% out {}%".format(
            ring.get("inner", {}).get("ratio", 0),
            ring.get("mid", {}).get("ratio", 0),
            ring.get("outer", {}).get("ratio", 0))
        cv2.putText(out, rt, (8, int(bar_h * 0.70)), font, scale * 0.85, (0, 212, 255), 1, cv2.LINE_AA)
    if engine:
        cv2.putText(out, f"engine: {engine}", (8, int(bar_h * 0.96)), font, scale * 0.75,
                    (0, 255, 136), 1, cv2.LINE_AA)
    return out


def main():
    if len(sys.argv) < 2:
        print("用法: python slice_detector.py <图片路径> [--engine rule|unet|auto] [--out 标注图保存路径]")
        return 1
    path = sys.argv[1]
    engine = "auto"
    if "--engine" in sys.argv:
        engine = sys.argv[sys.argv.index("--engine") + 1]
    img = cv2.imread(path)
    if img is None:
        print(f"无法读取图片: {path}")
        return 1
    res = analyze_slice(img, engine=engine)
    annotated = res.pop("annotated")
    for k, v in res.items():
        print(f"{k}: {v}")
    out = None
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    else:
        out = path.rsplit(".", 1)[0] + "_annotated.jpg"
    cv2.imwrite(out, annotated)
    print(f"标注图已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())