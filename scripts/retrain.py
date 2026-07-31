# -*- coding: utf-8 -*-
"""
一键重训：导出数据集 → 训练 SVM/RF/DL → 备份旧模型 → 登记模型版本

用法:
  python scripts/retrain.py --category apple --models svm rf
  python scripts/retrain.py --category apple --models svm rf dl --epochs 15
  python scripts/retrain.py --category apple --models svm rf --dry-run   # 只导出+特征统计，不训练

数据要求: dataset_collected/{category}/ 下含 合格(或优等) 与 不合格 两个标签目录。
训练后模型写入生产路径（apple_svm_model.dat / apple_rf_model.pkl / apple_dl_model.pth），
旧模型自动备份到 model_backups/，版本登记到 models_registry.json。
"""
import argparse
import json
import pickle
import shutil
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
EXPORT_DIR = config.OUTPUT_DIR / "dataset_export"
BACKUP_DIR = config.ROOT_DIR / "model_backups"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
GOOD_LABELS = ("合格", "优等")
BAD_LABELS = ("不合格",)

MODEL_FILES = {
    "svm": "apple_svm_model.dat",
    "rf": "apple_rf_model.pkl",
    "dl": "apple_dl_model.pth",
    "scaler": "apple_scaler.pkl",
}


def _log(log, msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    if log is not None:
        log(line)


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def label_to_class(name):
    if any(b in name for b in BAD_LABELS):
        return 1
    return 0


def export_dataset(category, seed=42):
    from scripts.export_dataset import export
    cat_export = EXPORT_DIR / category
    if cat_export.exists():
        shutil.rmtree(str(cat_export))
    total = export(category, 0.8, seed)
    if not total:
        raise RuntimeError("没有可用的数据集，请先在 dataset_collected/{category}/{合格,不合格}/ 放入图片")
    return total


def load_samples(split, category):
    """返回 [(image_path, label)]"""
    samples = []
    base = EXPORT_DIR / category / split
    if not base.is_dir():
        return samples
    for label_dir in sorted(base.iterdir()):
        if not label_dir.is_dir():
            continue
        cls = label_to_class(label_dir.name)
        for f in sorted(label_dir.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                samples.append((f, cls))
    return samples


def build_features(samples, log):
    from apple_defect_detector import extract_features_v2
    X, y, skipped = [], [], 0
    for i, (path, cls) in enumerate(samples):
        img = imread_unicode(path)
        if img is None:
            skipped += 1
            continue
        feats = extract_features_v2(img)
        if feats is None:
            skipped += 1
            continue
        X.append(feats)
        y.append(cls)
        if (i + 1) % 20 == 0:
            _log(log, f"    特征提取 {i + 1}/{len(samples)}")
    _log(log, f"    完成: {len(X)} 个有效样本, 跳过 {skipped}")
    return np.array(X, dtype=np.float32), np.array(y)


def backup_models():
    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / stamp
    dst.mkdir(parents=True, exist_ok=True)
    for key, fname in MODEL_FILES.items():
        src = config.ROOT_DIR / fname
        if src.exists():
            shutil.copy2(str(src), str(dst / fname))
    return dst


def train_ml(category, log):
    _log(log, "[SVM/RF] 读取训练/验证集...")
    train = load_samples("train", category)
    val = load_samples("val", category)
    if len(train) < 4 or len(val) == 0:
        raise RuntimeError("训练样本不足（train/val 切分后样本过少）")

    _log(log, f"[SVM/RF] 提取训练特征（{len(train)} 张）...")
    X_train, y_train = build_features(train, log)
    _log(log, f"[SVM/RF] 提取验证特征（{len(val)} 张）...")
    X_val, y_val = build_features(val, log)
    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError("特征提取失败（无法分割出食品区域）")

    # 平衡训练样本
    n0 = int(np.sum(y_train == 0)); n1 = int(np.sum(y_train == 1))
    if n0 == 0 or n1 == 0:
        raise RuntimeError("训练集需要同时包含 合格 与 不合格 两类样本")
    min_c = min(n0, n1)
    rng = np.random.RandomState(42)
    i0 = rng.choice(np.where(y_train == 0)[0], min_c, replace=False)
    i1 = rng.choice(np.where(y_train == 1)[0], min_c, replace=False)
    sel = np.concatenate([i0, i1])
    Xb, yb = X_train[sel], y_train[sel]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xb_s = scaler.fit_transform(Xb)
    Xv_s = scaler.transform(X_val)

    results = {}

    # ---------- SVM ----------
    _log(log, "训练 SVM (OpenCV RBF)...")
    t0 = time.time()
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setGamma(0.01)
    svm.setC(50.0)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100000, 1e-7))
    svm.train(Xb_s, cv2.ml.ROW_SAMPLE, yb)
    _, pred = svm.predict(Xv_s)
    pred = pred.flatten().astype(np.int32)
    svm_acc = float(np.mean(pred == y_val)) * 100
    results["svm"] = svm_acc
    _log(log, f"  SVM 验证准确率: {svm_acc:.2f}% （训练 {time.time()-t0:.1f}s）")

    # ---------- RF ----------
    _log(log, "训练 RandomForest (500 棵树)...")
    from sklearn.ensemble import RandomForestClassifier
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=500, max_depth=30,
                                min_samples_leaf=2, random_state=42,
                                n_jobs=-1, class_weight="balanced")
    rf.fit(Xb, yb)
    rf_acc = float(np.mean(rf.predict(X_val) == y_val)) * 100
    results["rf"] = rf_acc
    _log(log, f"  RF 验证准确率: {rf_acc:.2f}% （训练 {time.time()-t0:.1f}s）")

    # 保存
    backup = backup_models()
    _log(log, f"旧模型已备份到 {backup.name}")
    with open(config.ROOT_DIR / MODEL_FILES["scaler"], "wb") as f:
        pickle.dump(scaler, f)
    svm.save(str(config.ROOT_DIR / MODEL_FILES["svm"]))
    with open(config.ROOT_DIR / MODEL_FILES["rf"], "wb") as f:
        pickle.dump(rf, f)
    _log(log, f"新模型已保存: {MODEL_FILES['svm']}, {MODEL_FILES['rf']}, {MODEL_FILES['scaler']}")
    return results, len(X_train), len(X_val)


def train_dl(category, epochs, log):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models

    _log(log, "[DL] 准备数据 (MobileNetV3 Small, 224x224)...")
    train = load_samples("train", category)
    val = load_samples("val", category)
    if len(train) < 4 or len(val) == 0:
        raise RuntimeError("DL 训练样本不足")

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

    class FoodDS(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, i):
            path, label = self.samples[i]
            img = imread_unicode(path)
            if img is None:
                return self.__getitem__((i + 1) % len(self.samples))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.transform(img), label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(FoodDS(train, tf), batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(FoodDS(val, tfv), batch_size=16, shuffle=False, num_workers=0)

    model = models.mobilenet_v3_small(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 128),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))
    prev_path = config.ROOT_DIR / MODEL_FILES['dl']
    lr = 1e-3
    if prev_path.exists():
        try:
            model.load_state_dict(torch.load(str(prev_path), map_location=device, weights_only=True))
            lr = 5e-4
            _log(log, f"[DL] \u5df2\u52a0\u8f7d\u5f53\u524d\u6a21\u578b\u7ee7\u7eed\u5fae\u8c03: {MODEL_FILES['dl']}")
        except Exception as e:
            _log(log, f"[DL] \u5f53\u524d\u6a21\u578b\u52a0\u8f7d\u5931\u8d25\uff0c\u6539\u4e3a\u4ece\u96f6\u8bad\u7ec3: {e}")
    else:
        _log(log, '[DL] \u672a\u627e\u5230\u5f53\u524d\u6a21\u578b\uff0c\u4ece\u96f6\u8bad\u7ec3')
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    _log(log, f"[DL] 训练 {epochs} 轮 (设备: {device})...")
    for epoch in range(epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            _log(log, f"  epoch {epoch+1}/{epochs} loss={run_loss/max(total,1):.4f} train_acc={correct/max(total,1)*100:.1f}%")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += xb.size(0)
    acc = correct / max(total, 1) * 100
    _log(log, f"[DL] 验证准确率: {acc:.2f}%")

    backup = backup_models()
    torch.save(model.state_dict(), str(config.ROOT_DIR / MODEL_FILES["dl"]))
    _log(log, f"[DL] 新模型已保存: {MODEL_FILES['dl']}（旧模型备份 {backup.name}）")
    return acc, len(train), len(val)


def run_retrain(category, models, epochs=15, log=None):
    _log(log, f"=== 一键重训: {category} / {models} ===")
    if not (DATASET_DIR / category).is_dir():
        raise RuntimeError(f"品类目录不存在: {category}")

    from services.model_registry import ModelRegistry
    registry = ModelRegistry()
    summary = {}

    if any(m in models for m in ("svm", "rf", "dl")):
        n_export = export_dataset(category)
        _log(log, f"分类数据集已导出（train/val 切分，共 {n_export} 张）")

        if "svm" in models or "rf" in models:
            accs, n_tr, n_va = train_ml(category, log)
            for m in ("svm", "rf"):
                if m in models:
                    entry = registry.register(category, m, accs[m], n_tr)
                    summary[m] = {"accuracy": round(accs[m], 2), "version": entry["version"], "train": n_tr, "val": n_va}
                    _log(log, f"已登记 {category}_{m} {entry['version']} 准确率 {accs[m]:.2f}%")

        if "dl" in models:
            acc, n_tr, n_va = train_dl(category, epochs, log)
            entry = registry.register(category, "dl", acc, n_tr)
            summary["dl"] = {"accuracy": round(acc, 2), "version": entry["version"], "train": n_tr, "val": n_va}
            _log(log, f"已登记 {category}_dl {entry['version']} 准确率 {acc:.2f}%")

    if "yolo" in models:
        from scripts.yolo_export import merge
        from scripts.yolo_retrain import train as train_yolo
        n_yolo = merge(category, log=log)
        _log(log, f"YOLO 训练集已合并（导出 {n_yolo['exported']} 张）")
        acc = train_yolo(category, epochs=epochs, log=log)
        entry = registry.register(category, "yolo", acc, n_yolo["exported"])
        summary["yolo"] = {"accuracy": acc, "version": entry["version"], "train": n_yolo["exported"]}
        _log(log, f"已登记 {category}_yolo {entry['version']} mAP50-95 {acc}%")

    _log(log, f"=== 重训完成: {summary} ===")
    return {"category": category, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="一键重训：导出 → 训练 → 备份 → 登记版本")
    parser.add_argument("--category", default="apple", choices=["apple", "bread", "beverage"])
    parser.add_argument("--models", nargs="+", default=["svm", "rf"], choices=["svm", "rf", "dl"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        n = export_dataset(args.category)
        train = load_samples("train", args.category)
        val = load_samples("val", args.category)
        print(f"dry-run: 导出 {n} 张, 训练 {len(train)} 张, 验证 {len(val)} 张")
        return

    run_retrain(args.category, args.models, args.epochs)


if __name__ == "__main__":
    main()