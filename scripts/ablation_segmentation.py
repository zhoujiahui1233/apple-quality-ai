# -*- coding: utf-8 -*-
"""
消融实验 B：原始图像 vs 分割后图像 对深度学习分类准确率的影响

用法:
  python scripts/ablation_segmentation.py                  # 苹果，5 个 epoch
  python scripts/ablation_segmentation.py --epochs 8       # 加大训练轮数
  python scripts/ablation_segmentation.py --category apple --seed 42

原理:
  同一份数据、同一个网络(MobileNetV3 Small)、同样的训练参数，
  唯一变量是输入图像是否先去背景(segment_apple 分割)：
    raw: 原图直接分类
    seg: 背景置黑后再分类
  比较两者验证集准确率，验证"分割预处理"对 DL 是否有帮助。

输出:
  output/ablation_segmentation.json

注意:
  本机 CPU 训练较慢，数据少时结果仅供流程验证；毕设数据扩充后建议用 GPU 重跑。
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import config

DATASET_DIR = config.ROOT_DIR / "dataset_collected"
OUTPUT_DIR = config.OUTPUT_DIR
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def label_classes(folder_name):
    low = folder_name.lower()
    if "不合格" in folder_name or "reject" in low:
        return 1
    return 0


def iter_samples(category, limit=None):
    cat_dir = DATASET_DIR / category
    if not cat_dir.is_dir():
        return
    for label_dir in sorted(cat_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        cls = label_classes(label_dir.name)
        files = sorted(label_dir.glob("*"))
        if limit:
            files = files[:limit]
        for f in files:
            if f.suffix.lower() in IMG_EXTS:
                yield f, cls


def to_segmented(img_bgr):
    """分割去背景：返回背景置黑的预处理图；失败返回 None"""
    from apple_defect_detector import preprocess, segment_apple
    img_p, _ = preprocess(img_bgr)
    mask, _ = segment_apple(img_p)
    if mask is None:
        return None
    out = img_p.copy()
    out[mask == 0] = 0
    return out


def load_split(category, test_ratio, seed, limit):
    samples = list(iter_samples(category, limit))
    if len(samples) < 8:
        raise RuntimeError(
            f"数据不足（{len(samples)} 张），请在 dataset_collected/{category}/{{合格,不合格}}/ 扩充图片")
    from sklearn.model_selection import train_test_split
    paths = [str(p) for p, _ in samples]
    y = np.array([c for _, c in samples])
    tr_idx, va_idx = train_test_split(
        np.arange(len(y)), test_size=test_ratio, random_state=seed, stratify=y)
    return ([paths[i] for i in tr_idx], y[tr_idx],
            [paths[i] for i in va_idx], y[va_idx])


def train_pipeline(name, train_paths, train_y, val_paths, val_y, epochs, seed):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models

    torch.manual_seed(seed)
    np.random.seed(seed)

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.15, 0.12, 0.08, 0.04),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tfv = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    preproc = (to_segmented if name == "seg" else lambda img: img)

    class FoodDS(Dataset):
        def __init__(self, paths, labels, transform):
            self.items = [(p, int(l)) for p, l in zip(paths, labels)]
            self.transform = transform
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            p, label = self.items[i]
            img = imread_unicode(p)
            if img is None:
                return self.__getitem__((i + 1) % len(self.items))
            img = preproc(img)
            if img is None:
                return self.__getitem__((i + 1) % len(self.items))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.transform(img), label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = FoodDS(train_paths, train_y, tf)
    val_ds = FoodDS(val_paths, val_y, tfv)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = models.mobilenet_v3_small(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 128),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    t0 = time.time()
    last_train = last_val = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        correct = total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            correct += int((out.argmax(1) == yb).sum().item())
            total += len(yb)
        last_train = correct / max(1, total) * 100

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                correct += int((out.argmax(1) == yb).sum().item())
                total += len(yb)
        last_val = correct / max(1, total) * 100
        print(f"  [{name}] epoch {ep}/{epochs}  train {last_train:.1f}%  val {last_val:.1f}%")
    return {
        "train_acc": round(last_train, 2),
        "val_acc": round(last_val, 2),
        "device": str(device),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="消融实验 B：分割有无对 DL 的影响")
    parser.add_argument("--category", default="apple")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    train_paths, train_y, val_paths, val_y = load_split(
        args.category, args.test_ratio, args.seed, args.limit)
    print(f"[{args.category}] 训练 {len(train_paths)} 张 / 验证 {len(val_paths)} 张，"
          f"epochs={args.epochs}，两种输入对比\n")

    results = {}
    for name, label in (("raw", "原始图像"), ("seg", "分割后图像")):
        print(f"训练 [{name}] ({label})...")
        results[name] = train_pipeline(name, train_paths, train_y,
                                       val_paths, val_y, args.epochs, args.seed)

    payload = {
        "source": "ablation_segmentation",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category": args.category,
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "epochs": args.epochs,
        "seed": args.seed,
        "results": results,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_json = OUTPUT_DIR / "ablation_segmentation.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\n分割消融结果（验证集 {len(val_paths)} 张）:")
    print(f"  原始图像      train {results['raw']['train_acc']:.2f}%  val {results['raw']['val_acc']:.2f}%")
    print(f"  分割后图像    train {results['seg']['train_acc']:.2f}%  val {results['seg']['val_acc']:.2f}%")
    diff = results["seg"]["val_acc"] - results["raw"]["val_acc"]
    print(f"  分割 vs 原始: {diff:+.2f} 个百分点")
    print(f"已保存: {out_json}")


if __name__ == "__main__":
    main()