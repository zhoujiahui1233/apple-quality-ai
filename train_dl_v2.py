"""DL Model v2 - No download needed"""
import sys, os, cv2, numpy as np, time, pickle
sys.stdout.reconfigure(encoding="utf-8")
os.environ["YOLO_CONFIG_DIR"] = "E:\\apple\\.ultralytics"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

DATA_DIR = "E:\\apple\\kaggle_data\\extracted"
OLD_MODEL = "E:\\apple\\apple_dl_model.pth"
OUTPUT_MODEL = "E:\\apple\\apple_dl_model.pth"
BATCH_SIZE = 16
EPOCHS = 50
IMG_SIZE = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AppleDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.samples = []
        self.transform = transform
        for label, cls_name in [(0, "freshapples"), (1, "rottenapples")]:
            path = os.path.join(root_dir, split, cls_name)
            for fname in os.listdir(path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(path, fname), label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None: return self.__getitem__((idx + 1) % len(self.samples))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(img)
        return img, label

print("=" * 55)
print("  DL Model v2")
print("=" * 55)
print(f"  Model: MobileNetV3 Large (2.5x bigger than Small)")
print(f"  Input: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Device: {DEVICE}")

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.15, 0.1, 0.05),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("\n[1/4] Loading data...")
train_ds = AppleDataset(DATA_DIR, "train", train_transform)
val_ds = AppleDataset(DATA_DIR, "test", val_transform)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

print("\n[2/4] Creating model (MobileNetV3 Large)...")
model = models.mobilenet_v3_large(weights=None)
in_feat = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_feat, 512), nn.Hardswish(), nn.Dropout(0.3),
    nn.Linear(512, 128), nn.Hardswish(), nn.Dropout(0.2),
    nn.Linear(128, 2),
)

# Try to load old weights as initialization
if os.path.exists(OLD_MODEL):
    try:
        old_state = torch.load(OLD_MODEL, map_location=DEVICE, weights_only=True)
        model.load_state_dict(old_state, strict=False)
        print("  Loaded old model weights as initialization")
    except: print("  Could not load old weights, training from scratch")

model = model.to(DEVICE)
total = sum(p.numel() for p in model.parameters())
print(f"  Params: {total:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

print("\n[3/4] Training...")
best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    correct = total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward(); optimizer.step()
        _, preds = torch.max(model(inputs), 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
    train_acc = correct / total * 100
    scheduler.step()
    
    model.eval()
    vc = vt = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            _, preds = torch.max(model(inputs), 1)
            vc += torch.sum(preds == labels).item(); vt += labels.size(0)
    val_acc = vc / vt * 100
    
    mark = " *" if val_acc > best_acc else "  "
    print(f"  Ep {epoch:2d}/{EPOCHS} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%{mark}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), OUTPUT_MODEL)

print(f"\n[4/4] Done! Best val accuracy: {best_acc:.2f}%")
print(f"  Saved: {OUTPUT_MODEL}")
