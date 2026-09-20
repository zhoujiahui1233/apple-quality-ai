# ============================================================
# DL 外观分类 v2 训练（Kaggle GPU 一键版）
# 数据: dl_dataset_v2.zip (你的 合格/不合格，均衡各~5400)
# 结构: MobileNetV3-Large @384（与系统 DL 加载器一致）
# 产物: apple_dl_model.pth  -> 放到 E:\apple\apple_dl_model.pth
# ============================================================
import os, glob, zipfile, shutil, re, time

# ---------- 智能定位数据 ----------
def locate_input():
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    if zips:
        return max(zips, key=os.path.getsize), True
    if glob.glob("/kaggle/input/**/images/train", recursive=True):
        for p in glob.glob("/kaggle/input/**/images/train", recursive=True):
            return p[: p.rindex("/images")], False
    raise SystemExit("input 里没有 dl_dataset_v2，请 +Add Input 添加。内容: "
                     + str(os.listdir("/kaggle/input")))

src, is_zip = locate_input()
DST = "/kaggle/working/dl_dataset_v2"
if is_zip:
    os.makedirs(DST, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(DST)
    for d in os.listdir(DST):
        sub = os.path.join(DST, d)
        if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "images")):
            DST = sub
            break
else:
    if os.path.abspath(src).startswith("/kaggle/working"):
        DST = src
    else:
        shutil.rmtree(DST, ignore_errors=True)
        shutil.copytree(src, DST)
print("数据目录:", DST)

# ---------- 训练 ----------
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image

class AppleDS(Dataset):
    def __init__(self, root, split, tf):
        self.samples = []; self.tf = tf
        for label, cls in [(0, "good"), (1, "bad")]:
            p = os.path.join(root, "images", split, cls)
            for f in os.listdir(p):
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                fp = os.path.join(p, f)
                try:
                    with Image.open(fp) as im:
                        im.verify()      # 坏图直接跳过，防止训练中断
                except Exception:
                    continue
                self.samples.append((fp, label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]
        from PIL import Image
        im = Image.open(p).convert("RGB")
        return self.tf(im), y

IMG = 384
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf_tr = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.15, 0.1, 0.05),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
tf_ev = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])

train_ds = AppleDS(DST, "train", tf_tr)
val_ds = AppleDS(DST, "val", tf_ev)
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
train_loader = DataLoader(train_ds, 32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, 32, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model = models.mobilenet_v3_large(weights=None)
in_feat = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_feat, 512), nn.Hardswish(), nn.Dropout(0.3),
    nn.Linear(512, 128), nn.Hardswish(), nn.Dropout(0.2),
    nn.Linear(128, 2),
)
model = model.to(device)

EPOCHS = 60
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    correct = total = 0
    for bi, (x, y) in enumerate(train_loader, 1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward(); optimizer.step()
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
        if bi % 200 == 0:
            print(f"  epoch {epoch} batch {bi}/{len(train_loader)}", flush=True)
    scheduler.step()
    model.eval()
    vc = vt = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            vc += (model(x).argmax(1) == y).sum().item()
            vt += y.size(0)
    acc = vc / vt * 100
    mark = " *" if acc > best_acc else ""
    print(f"Ep {epoch:2d}/{EPOCHS} | Train {(correct/total*100):.2f}% | Val {acc:.2f}% | {time.time()-t0:.0f}s{mark}",
          flush=True)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "/kaggle/working/apple_dl_model.pth")

print(f"\n最佳验证准确率: {best_acc:.2f}%")
print("apple_dl_model.pth 已保存，下载后放到 E:\\apple\\apple_dl_model.pth")
