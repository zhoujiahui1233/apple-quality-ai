"""
苹果深度学习分类模型训练 (PyTorch + MobileNetV3)
=================================================
运行前先安装: pip install torch torchvision pillow

用法: python train_dl_model.py
训练完成后自动保存为 apple_dl_model.pth
"""

import os, time, sys
import cv2
import numpy as np
from PIL import Image

# ---------- PyTorch ----------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ---------- 配置 ----------
DATA_DIR = "E:\\apple\\kaggle_data\\extracted"
OUTPUT_MODEL = "E:\\apple\\apple_dl_model.pth"
IMG_SIZE = 224
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".torch_cache")  # MobileNet 输入尺寸
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

# ---------- 数据集 ----------
class AppleDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform
        for label, cat in [(0, "freshapples"), (1, "rottenapples")]:
            folder = os.path.join(root_dir, split, cat)
            for fname in os.listdir(folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.paths.append(os.path.join(folder, fname))
                    self.labels.append(label)
        print(f"  {split}: {len(self.paths)} 图片 ({sum(1 for l in self.labels if l==0)} 新鲜, {sum(1 for l in self.labels if l==1)} 腐烂)")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------- 数据增强 ----------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("\n加载数据...")
train_dataset = AppleDataset(DATA_DIR, "train", train_transform)
test_dataset = AppleDataset(DATA_DIR, "test", val_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------- 模型 ----------
print("\n构建 MobileNetV3...")
model = models.mobilenet_v3_small(weights=None)
# 替换分类头（原来 1000 类 -> 2 类）
in_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2),
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------- 训练 ----------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item()
    return loss_sum / len(loader), correct / total

def eval_model(model, loader):
    model.eval()
    total, correct = 0, 0
    fresh_correct, fresh_total = 0, 0
    rot_correct, rot_total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            fresh_correct += ((pred == 0) & (y == 0)).sum().item()
            fresh_total += (y == 0).sum().item()
            rot_correct += ((pred == 1) & (y == 1)).sum().item()
            rot_total += (y == 1).sum().item()
    return correct/total, fresh_correct/fresh_total, rot_correct/rot_total

print(f"\n开始训练 ({EPOCHS} epochs)...")
best_acc = 0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    test_acc, fresh_acc, rot_acc = eval_model(model, test_loader)
    scheduler.step()
    print(f"  Epoch {epoch:2d}/{EPOCHS} | 损失: {train_loss:.4f} | 训练: {train_acc*100:.1f}% | 测试: {test_acc*100:.1f}% (新鲜:{fresh_acc*100:.1f}% 腐烂:{rot_acc*100:.1f}%) | {(time.time()-t0):.0f}s")
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), OUTPUT_MODEL)
        print(f"    -> 保存最佳模型 (acc={test_acc*100:.1f}%)")

print(f"\n训练完成！最佳准确率: {best_acc*100:.1f}%")
print(f"模型已保存: {OUTPUT_MODEL}")
