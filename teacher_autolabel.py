# -*- coding: utf-8 -*-
"""
教师模型批量自动标注（本地免费跑剩余未标图片）
=============================================
加载训练好的 teacher_defect_best.pt，批量给"不合格"图打缺陷框：
  - 输出 YOLO 标签 + summary.csv（每张图最高置信度）
  - conf>=accept_conf 视为"自动采纳"草稿
  - conf<accept_conf 或 完全没检出 的进 review 队列，等人工复核

用法:
  python teacher_autolabel.py                      # 用默认参数跑
  python teacher_autolabel.py --limit 200          # 先试 200 张
  python teacher_autolabel.py --resume             # 断点续跑

依赖已装: ultralytics
"""
import argparse
import csv
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "teacher_defect_best.pt"))
    ap.add_argument("--images",
                    default=str(ROOT / "dataset_collected" / "apple" / "不合格"))
    ap.add_argument("--out", default=str(ROOT / "teacher_autolabel_out"))
    ap.add_argument("--conf", type=float, default=0.35, help="检出最低置信度")
    ap.add_argument("--accept-conf", type=float, default=0.6,
                    help="高于此值算自动采纳")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-stems-file", default="",
                    help="跳过文件：每行一个图片stem（如人工已复核的种子名单）")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"找不到模型: {weights}")
        print("请先把 Kaggle 训练的 teacher_defect_best.pt 下载到 E:\\apple\\")
        sys.exit(2)
    images_dir = Path(args.images)
    if not images_dir.is_dir():
        print(f"图片目录不存在: {images_dir}")
        sys.exit(2)
    out = Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "viz").mkdir(parents=True, exist_ok=True)

    # 收集图片（去重扩展名，优先 .jpg）
    seen = {}
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            if p.stem not in seen or p.suffix.lower() == ".jpg":
                seen[p.stem] = p
    files = sorted(seen.values())
    if args.skip_stems_file:
        skip_file = Path(args.skip_stems_file)
        if skip_file.exists():
            skip = {Path(line.strip()).stem if "\\" in line or "/" in line else line.strip()
                    for line in skip_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()}
            before = len(files)
            files = [p for p in files if p.stem not in skip]
            print(f"按名单跳过 {before - len(files)} 张已人工复核的图")
        else:
            print(f"警告: 跳过名单不存在 {skip_file}")
    if args.resume:
        done = {p.stem for p in (out / "labels").glob("*.txt")}
        files = [p for p in files if p.stem not in done]
    if args.limit:
        files = files[:args.limit]
    print(f"模型: {weights.name}")
    print(f"待处理: {len(files)} 张 -> {out}")
    if not files:
        print("没有待处理图片（可能已全部跑完，或目录为空）")
        return

    model = YOLO(str(weights))
    records = []
    import time
    t0 = time.time()
    for i, fp in enumerate(files, 1):
        res = model.predict(str(fp), conf=args.conf, imgsz=640, verbose=False)[0]
        boxes = res.boxes
        rows = []
        max_conf = 0.0
        if boxes is not None and len(boxes):
            names = res.names
            for j in range(len(boxes)):
                cls = int(boxes.cls[j])
                conf = float(boxes.conf[j])
                x1, y1, x2, y2 = boxes.xyxy[j].tolist()
                max_conf = max(max_conf, conf)
                cx = (x1 + x2) / 2 / res.orig_shape[1]
                cy = (y1 + y2) / 2 / res.orig_shape[0]
                w = (x2 - x1) / res.orig_shape[1]
                h = (y2 - y1) / res.orig_shape[0]
                rows.append(f"{cls} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}")
        lab = out / "labels" / (fp.stem + ".txt")
        lab.write_text("\n".join(rows), encoding="utf-8")
        status = "accept" if max_conf >= args.accept_conf else "review"
        if not rows:
            status = "nodetect"
        records.append({"file": fp.name, "n_defect": len(rows),
                        "max_conf": round(max_conf, 3), "status": status})
        if i % 50 == 0:
            eta = (time.time() - t0) / i * (len(files) - i)
            print(f"[{i}/{len(files)}] 已处理，预计还需 {eta/60:.1f} 分钟")

    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "n_defect", "max_conf", "status"])
        wr.writeheader()
        wr.writerows(records)

    from collections import Counter
    cnt = Counter(r["status"] for r in records)
    (out / "review_list.txt").write_text(
        "\n".join(r["file"] for r in records if r["status"] != "accept"),
        encoding="utf-8")
    print("\n完成:", len(records))
    print("  自动采纳(高置信):", cnt.get("accept", 0))
    print("  需人工复核(低置信):", cnt.get("review", 0))
    print("  未检出缺陷(重点抽查):", cnt.get("nodetect", 0))
    print(f"明细: {out / 'summary.csv'}")
    print(f"复核清单: {out / 'review_list.txt'}")


if __name__ == "__main__":
    main()
