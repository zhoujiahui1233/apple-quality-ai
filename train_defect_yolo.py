# -*- coding: utf-8 -*-
"""缺陷检测 YOLO 训练（通用版：本地 GPU / 阿里云 / 任何 Linux 或 Windows 机器）。

与 Kaggle 版的区别：不依赖 /kaggle 路径，数据目录用参数指定。

用法:
  python train_defect_yolo.py --data /path/to/defect_dataset_clean_v2
  python train_defect_yolo.py --data E:\\apple\\defect_dataset_clean_v2 --batch 4

常用参数:
  --model yolo11s.pt | yolo11n.pt | yolo11m.pt
  --epochs 200 --imgsz 1280 --batch 8
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="数据集根目录（含 images/ 与 labels/）")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--device", default=None, help="0 / 0,1 / cpu；默认自动")
    ap.add_argument("--project", default=None)
    ap.add_argument("--name", default="defect_clean_v2")
    args = ap.parse_args()

    data = os.path.abspath(args.data)
    if not os.path.isdir(os.path.join(data, "images", "train")):
        sys.exit("数据目录不正确，需要包含 images/train: " + data)

    # 重写 data.yaml，修正为当前机器上的实际路径（跨平台迁移必需）
    yaml_path = os.path.join(data, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write('path: %s\ntrain: images/train\nval: images/val\n'
                'nc: 1\nnames: ["defect"]\n' % data.replace("\\", "/"))
    print("data.yaml ->", yaml_path)

    from ultralytics import YOLO

    device = args.device
    if device is None:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = YOLO(args.model)
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        project=args.project or os.path.join(data, "runs"),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        warmup_epochs=3.0,
        mosaic=1.0,
        close_mosaic=15,
        scale=0.6,
        degrees=10.0,
        translate=0.15,
        fliplr=0.5,
        flipud=0.2,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        mixup=0.1,
        copy_paste=0.1,
        workers=4,
        verbose=True,
    )

    best = os.path.join(args.project or os.path.join(data, "runs"),
                        args.name, "weights", "best.pt")
    print("\nbest.pt:", best, "存在:", os.path.exists(best))
    if os.path.exists(best):
        m = YOLO(best)
        r = m.val(data=yaml_path, imgsz=args.imgsz, device=device)
        print("mAP50: %.4f | mAP50-95: %.4f | P: %.4f | R: %.4f"
              % (r.box.map50, r.box.map, r.box.mp, r.box.mr))


if __name__ == "__main__":
    main()
