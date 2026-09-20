# -*- coding: utf-8 -*-
"""切片 U-Net 分割模型训练脚本（本地 CPU / Kaggle GPU 均可直接跑）。

输入数据（由 scripts/slice_seg_prepare.py 生成）：
  slice_seg_dataset/
    images/*.jpg   masks/*.png（0背景/1果肉/2果核/3褐变/4果皮）
    train.txt  val.txt  class_weights.json

用法（本地）：
  python scripts/train_slice_unet.py --data slice_seg_dataset --epochs 60 --out models/slice_unet.pt
  python scripts/train_slice_unet.py --backbone none --epochs 40   # 离线/无需预训练权重

用法（Kaggle）：
  把 slice_seg_dataset 文件夹和本脚本上传到 Kaggle Notebook，GPU 加速器下运行：
  !python train_slice_unet.py --data slice_seg_dataset --epochs 80 --backbone mobilenet_v2 --out models/slice_unet.pt
  训练完把 models/slice_unet.pt 拿回来放到 E:\\apple\\models\\ 即可。
"""
import os
# Windows 下 torch+numpy(MKL) 可能发生 OpenMP 库冲突，提前规避
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.slice_unet_model import (SLICE_CLASSES, build_unet, load_checkpoint,
                                      save_checkpoint)

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_pair(line, data_root, size, train=False, rng=None):
    """读取 (图片, 掩膜)，统一缩放到 size×size。返回归一化张量 + 掩膜张量。"""
    img_rel, mask_rel = line.split()
    img = cv2.imread(str(data_root / img_rel))
    mask = cv2.imread(str(data_root / mask_rel), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    if train and rng is not None:
        # 数据增强：水平/垂直翻转、90 度旋转、亮度对比度抖动
        if rng.random() < 0.5:
            img = img[:, ::-1]; mask = mask[:, ::-1]
        if rng.random() < 0.5:
            img = img[::-1, :]; mask = mask[::-1, :]
        k = rng.choice([0, 1, 2, 3])
        if k:
            img = np.rot90(img, k); mask = np.rot90(mask, k)
        img = img.astype(np.float32)
        alpha = 1.0 + rng.uniform(-0.15, 0.15)
        beta = rng.uniform(-20, 20)
        img = np.clip(img * alpha + beta, 0, 255)
    else:
        img = img.astype(np.float32)

    x = torch.from_numpy((img / 255.0 - IMG_MEAN) / IMG_STD).permute(2, 0, 1)
    y = torch.from_numpy(mask.astype(np.int64))
    return x, y


def load_pairs(txt_path, data_root, size, train, rng, shuffle=True):
    lines = [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if shuffle:
        rng.shuffle(lines)
    batch = []
    for ln in lines:
        item = read_pair(ln, data_root, size, train=train, rng=rng)
        if item is not None:
            batch.append(item)
    return batch


def batch_collate(batch, size):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    return xs, ys


def iou_parts(pred, target, n_classes):
    """逐类 IoU 与 Dice 的分子/分母累加（用于批次聚合）。返回 (n_classes, 4)。"""
    hist = torch.zeros((n_classes, 4), dtype=torch.long)
    if pred.numel() == 0:
        return hist
    for c in range(n_classes):
        pc = (pred == c)
        tc = (target == c)
        inter = int((pc & tc).sum())
        union = int((pc | tc).sum())
        hist[c, 0] += inter
        hist[c, 1] += union
        hist[c, 2] += inter * 2
        hist[c, 3] += int(pc.sum()) + int(tc.sum())
    return hist


def compute_metrics(hist, n_classes):
    ious, dices = [], []
    for c in range(n_classes):
        inter, union = int(hist[c, 0]), int(hist[c, 1])
        ious.append(inter / union if union > 0 else float("nan"))
        d2, d3 = int(hist[c, 2]), int(hist[c, 3])
        dices.append(d2 / d3 if d3 > 0 else float("nan"))
    miou = float(np.nanmean([v for v in ious if not np.isnan(v)]))
    return ious, dices, miou


def browning_ratio_from_mask(mask):
    """从掩膜计算褐变占比（与推理逻辑一致）。"""
    flesh = (mask == 1)
    brown = (mask == 3)
    fpx = int(flesh.sum())
    return float(brown.sum()) / fpx if fpx > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="切片 U-Net 分割训练")
    ap.add_argument("--data", default=str(ROOT / "slice_seg_dataset"))
    ap.add_argument("--out", default=str(ROOT / "models" / "slice_unet.pt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", default="mobilenet_v2", choices=["mobilenet_v2", "none"])
    ap.add_argument("--pretrained", type=int, default=1, help="mobilenet_v2 是否用预训练权重")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    data_root = Path(args.data)
    train_txt = data_root / "train.txt"
    val_txt = data_root / "val.txt"
    if not (train_txt.exists() and val_txt.exists()):
        print(f"未找到 {train_txt} / {val_txt}，请先运行 scripts/slice_seg_prepare.py")
        return 1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"设备: {device} | 数据: {data_root} | 骨架: {args.backbone} | 尺寸: {args.size}")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data = load_pairs(train_txt, data_root, args.size, train=True, rng=rng)
    val_data = load_pairs(val_txt, data_root, args.size, train=False, rng=rng, shuffle=False)
    if not train_data or not val_data:
        print("训练/验证数据为空，请检查数据集目录")
        return 1

    weights = [1.0] * len(SLICE_CLASSES)
    cw_path = data_root / "class_weights.json"
    if cw_path.exists():
        try:
            weights = json.loads(cw_path.read_text(encoding="utf-8"))["weights"]
        except Exception:
            pass
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))

    model = build_unet(n_classes=len(SLICE_CLASSES), backbone=args.backbone,
                       pretrained=bool(args.pretrained))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_classes = len(SLICE_CLASSES)
    best_miou = -1.0
    history = {"train_loss": [], "val_miou": [], "val_brown_mae": []}
    t0 = time.time()

    def evaluate():
        model.eval()
        hist = torch.zeros((n_classes, 4), dtype=torch.long)
        errs = []
        n_val = len(val_data)
        with torch.no_grad():
            for i in range(0, n_val, args.batch):
                chunk = val_data[i:i + args.batch]
                xs, ys = batch_collate(chunk, args.size)
                xs, ys = xs.to(device), ys.to(device)
                logits = model(xs)
                pred = torch.argmax(logits, dim=1)
                hist += iou_parts(pred.cpu(), ys.cpu(), n_classes)
                for p, y in zip(pred.cpu().numpy(), ys.cpu().numpy()):
                    errs.append(abs(browning_ratio_from_mask(p) - browning_ratio_from_mask(y)))
        ious, dices, miou = compute_metrics(hist, n_classes)
        mae = float(np.mean(errs)) if errs else 0.0
        return miou, ious, dices, mae

    print(f"训练样本 {len(train_data)} / 验证样本 {len(val_data)} | 类别权重 {[round(w,2) for w in weights]}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_data)
        total_loss = 0.0
        n_batch = 0
        for i in range(0, len(train_data), args.batch):
            chunk = train_data[i:i + args.batch]
            xs, ys = batch_collate(chunk, args.size)
            xs, ys = xs.to(device), ys.to(device)
            optimizer.zero_grad()
            logits = model(xs)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batch += 1
        scheduler.step()

        miou, ious, dices, mae = evaluate()
        avg_loss = total_loss / max(n_batch, 1)
        history["train_loss"].append(avg_loss)
        history["val_miou"].append(miou)
        history["val_brown_mae"].append(mae)
        if epoch % 5 == 0 or epoch == args.epochs or miou > best_miou:
            print(f"epoch {epoch:03d}/{args.epochs} | loss {avg_loss:.4f} | val mIoU {miou:.4f} | 褐变占比MAE {mae:.4f} | {int(time.time()-t0)}s")
        if miou > best_miou:
            best_miou = miou
            save_checkpoint(model, out_path, extra={"best_miou": float(miou), "epoch": epoch})

    # 最终指标
    miou, ious, dices, mae = evaluate()
    print("\n===== 最终验证结果 =====")
    for c, name in enumerate(SLICE_CLASSES):
        print(f"  {name}: IoU {ious[c]:.4f}  Dice {dices[c]:.4f}")
    print(f"  mIoU: {miou:.4f} | 褐变占比 MAE: {mae:.4f}")
    print(f"模型已保存: {out_path}")

    metrics = {"best_miou": float(best_miou), "final_miou": float(miou),
               "per_class_iou": {n: (None if np.isnan(v) else float(v)) for n, v in zip(SLICE_CLASSES, ious)},
               "per_class_dice": {n: (None if np.isnan(v) else float(v)) for n, v in zip(SLICE_CLASSES, dices)},
               "browning_mae": float(mae), "epochs": args.epochs, "size": args.size,
               "backbone": args.backbone, "history": history}
    (data_root / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    # 训练曲线（matplotlib 可用时）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ep = range(1, len(history["train_loss"]) + 1)
        ax[0].plot(ep, history["train_loss"]); ax[0].set_title("train loss")
        ax[1].plot(ep, history["val_miou"]); ax[1].set_title("val mIoU")
        ax[2].plot(ep, history["val_brown_mae"]); ax[2].set_title("browning MAE")
        for a in ax: a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(data_root / "train_curves.png"), dpi=110)
        print(f"训练曲线: {data_root / 'train_curves.png'}")
    except Exception:
        pass

    print("\n下一步：把模型放到 E:\\apple\\models\\slice_unet.pt 后重启服务，"
          "切片检测会自动优先使用 U-Net（引擎显示 U-Net 分割模型）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())