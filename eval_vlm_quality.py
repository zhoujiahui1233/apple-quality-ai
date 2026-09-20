# -*- coding: utf-8 -*-
"""
VLM 自动标注质量评估
====================
拿 VLM 自动生成的标签和你人工标注的标签比对，输出：
  - Recall   : 人工缺陷框里，VLM 框中的比例（越高越不漏检）
  - Precision: VLM 出的框里，确实对上人工框的比例（越高越少误报）
IoU >= 阈值(默认0.25) 算“对上”。

额外两个对“预标注草稿”更重要的指标：
  - 中心命中率 : 人工缺陷框的中心，被至少一个 VLM 框覆盖的比例。
                 高 = VLM 框基本“指到了地方”，人工只要微调即可。
  - 图片级命中 : 至少命中一个缺陷的图片占“有缺陷图片”的比例。

用法（命令行长路径建议用引号包裹）：
  python eval_vlm_quality.py --vlm-labels E:/apple/auto_label_output/labels --human-labels E:/apple/defect_annotate/lables

说明：两边的标签都按 YOLO 格式(类别 cx cy w h)，同名 txt 一一对应。
"""
import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_boxes(txt_path: Path):
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            _, cx, cy, w, h = (float(p) for p in parts)
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def contains(big, small_center):
    x1, y1, x2, y2 = big
    cx, cy = small_center
    return x1 <= cx <= x2 and y1 <= cy <= y2


def box_center(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-labels", required=True, help="VLM 输出 labels 目录")
    ap.add_argument("--human-labels", required=True, help="人工标注 labels 目录")
    ap.add_argument("--iou", type=float, default=0.25, help="IoU 阈值(默认0.25)")
    args = ap.parse_args()

    vlm_dir = Path(args.vlm_labels)
    human_dir = Path(args.human_labels)
    human_files = sorted(p for p in human_dir.glob("*.txt"))
    if not human_files:
        print(f"人工标注目录为空: {human_dir}")
        sys.exit(2)

    total_human = total_vlm = total_match = 0
    total_center_hit = 0
    img_with_human = 0
    img_center_hit = 0
    per_image = []
    for hf in human_files:
        human = load_boxes(hf)
        vf = vlm_dir / hf.name
        vlm = load_boxes(vf)
        if not human and not vlm:
            continue
        matched = 0
        used = [False] * len(vlm)
        for hb in human:
            for j, vb in enumerate(vlm):
                if not used[j] and iou(hb, vb) >= args.iou:
                    matched += 1
                    used[j] = True
                    break
        total_human += len(human)
        total_vlm += len(vlm)
        total_match += matched
        center_hit = 0
        for hb in human:
            hc = box_center(hb)
            if any(contains(vb, hc) for vb in vlm):
                center_hit += 1
        total_center_hit += center_hit
        if human:
            img_with_human += 1
            if center_hit > 0:
                img_center_hit += 1
        if human or vlm:
            per_image.append((hf.name, len(human), len(vlm), matched, center_hit))

    recall = total_match / total_human if total_human else 0.0
    precision = total_match / total_vlm if total_vlm else 0.0
    center_hit_rate = total_center_hit / total_human if total_human else 0.0
    img_hit_rate = img_center_hit / img_with_human if img_with_human else 0.0

    print(f"IoU 阈值: {args.iou}")
    print(f"参与图片数: {len(per_image)}")
    print(f"人工缺陷框总数: {total_human}  VLM 框总数: {total_vlm}  匹配数: {total_match}")
    print(f"Recall   (人工框被 VLM 框中的比例): {recall:.1%}")
    print(f"Precision(VLM 框中真正对上人工框的): {precision:.1%}")
    print(f"中心命中率(人工框中心被 VLM 覆盖)  : {center_hit_rate:.1%}")
    print(f"图片级命中(有缺陷图至少中一个)     : {img_hit_rate:.1%}  ({img_center_hit}/{img_with_human})")
    if not per_image:
        return
    print("\n=== 每张明细 (文件 | 人工框 | VLM框 | 匹配 | 中心命中) ===")
    for name, h, v, m, ch in per_image[:60]:
        flag = "  <-- 有缺陷却完全没标到" if h and ch == 0 else ""
        print(f"{name} | {h} | {v} | {m} | {ch}{flag}")
    if len(per_image) > 60:
        print(f"... 其余 {len(per_image) - 60} 张略")


if __name__ == "__main__":
    main()
