# ============================================================
# DL 外观分类 重训（Kaggle GPU 版，无泄漏切分）
# ------------------------------------------------------------
# 数据: dl_dataset_clean.zip （由 scripts/build_dl_dataset_clean.py 生成，
#       按拍摄簇切分，train/val 无近重复）
# 结构: MobileNetV3-Large @384，与系统 apple_defect_detector.load_dl_model_raw 一致
# 产物: apple_dl_model.pth  -> 下载后覆盖 E:\apple\apple_dl_model.pth
#
# 相比旧版的关键改动：
#   1. 用 ImageNet 预训练 backbone（旧版 weights=None 从零训，小数据很难收敛）
#   2. 标签平滑 + 更强增强 + 随机擦除，抑制过拟合
#   3. 带 warmup 的余弦退火 + 早停，按「平衡准确率」保存最优（类别更均衡）
#   4. 输出混淆矩阵与两类各自准确率
# ============================================================
import os, glob, zipfile, shutil, time, json


def locate_input():
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    pref = [z for z in zips if "dl_dataset" in os.path.basename(z).lower()]
    if pref:
        return max(pref, key=os.path.getsize), True
    if zips:
        return max(zips, key=os.path.getsize), True
    hits = glob.glob("/kaggle/input/**/images/train", recursive=True)
    if hits:
        return hits[0][: hits[0].rindex("/images")], False
    raise SystemExit("没找到 dl_dataset_clean，请 +Add Input 上传。当前: "
                     + str(os.listdir("/kaggle/input")))


src, is_zip = locate_input()
DST = "/kaggle/working/dl_dataset_clean"
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
    DST = src if os.path.abspath(src).startswith("/kaggle/working") else DST
    if DST != src:
        shutil.rmtree(DST, ignore_errors=True)
        shutil.copytree(src, DST)
print("数据目录:", DST)

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from PIL import Image

IMG = 384
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EPOCHS = 60
WARMUP = 3
BATCH = 32
PATIENCE = 12


class AppleDS(Dataset):
    def __init__(self, root, split, tf):
        self.tf = tf
        self.samples = []
        for label, cls in [(0, "good"), (1, "bad")]:
            p = os.path.join(root, "images", split, cls)
            for f in os.listdir(p):
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                fp = os.path.join(p, f)
                try:
                    with Image.open(fp) as im:
                        im.verify()
                except Exception:
                    continue
                self.samples.append((fp, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.tf(Image.open(p).convert("RGB")), y


tf_tr = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.25, 0.2, 0.15, 0.05),
    transforms.RandomAffine(0, translate=(0.12, 0.12), scale=(0.85, 1.15), shear=8),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
])
tf_ev = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = AppleDS(DST, "train", tf_tr)
val_ds = AppleDS(DST, "val", tf_ev)
print("Train:", len(train_ds), " Val:", len(val_ds))
train_loader = DataLoader(train_ds, BATCH, shuffle=True, num_workers=4,
                          pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, BATCH, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- 预训练 backbone + 与系统一致的分类头 ----
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
in_feat = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_feat, 512), nn.Hardswish(), nn.Dropout(0.3),
    nn.Linear(512, 128), nn.Hardswish(), nn.Dropout(0.2),
    nn.Linear(128, 2),
)
model = model.to(device)

# 分类器新初始化，给更大学习率；backbone 用较小学习率微调
head_params = list(model.classifier.parameters())
head_ids = {id(p) for p in head_params}
bb_params = [p for p in model.parameters() if id(p) not in head_ids]
optimizer = optim.AdamW([
    {"params": bb_params, "lr": 3e-4},
    {"params": head_params, "lr": 1e-3},
], weight_decay=1e-4)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


def lr_lambda(ep):
    if ep < WARMUP:
        return (ep + 1) / WARMUP
    prog = (ep - WARMUP) / max(1, EPOCHS - WARMUP)
    return 0.5 * (1 + torch.cos(torch.tensor(prog * 3.1415926))).item()


scheduler = LambdaLR(optimizer, lr_lambda)


def evaluate():
    model.eval()
    ok = tot = 0
    c0 = [0, 0]
    c1 = [0, 0]
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            ok += (pred == y).sum().item()
            tot += y.size(0)
            for t, p in zip(y.tolist(), pred.tolist()):
                if t == 0:
                    c0[0 if p == 0 else 1] += 1
                else:
                    c1[0 if p == 0 else 1] += 1
    acc = 100.0 * ok / max(1, tot)
    a0 = 100.0 * c0[0] / max(1, sum(c0))
    a1 = 100.0 * c1[1] / max(1, sum(c1))
    return acc, (a0 + a1) / 2.0, a0, a1


best = 0.0
best_ep = 0
no_imp = 0
history = []
for ep in range(EPOCHS):
    t0 = time.time()
    model.train()
    tot_loss = 0.0
    nb = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        nb += 1
    scheduler.step()
    tr_loss = tot_loss / max(1, nb)
    acc, bacc, a0, a1 = evaluate()
    history.append({"epoch": ep + 1, "train_loss": tr_loss,
                    "val_acc": acc, "val_bal_acc": bacc,
                    "acc_good": a0, "acc_bad": a1})
    mark = ""
    if bacc > best:
        best, best_ep = bacc, ep + 1
        no_imp = 0
        torch.save(model.state_dict(), "/kaggle/working/apple_dl_model.pth")
        mark = " *saved"
    else:
        no_imp += 1
    print("Ep %2d/%d | loss %.4f | val acc %.2f%% | bal_acc %.2f%% "
          "(good %.1f / bad %.1f) | %.0fs%s"
          % (ep + 1, EPOCHS, tr_loss, acc, bacc, a0, a1, time.time() - t0, mark),
          flush=True)
    if no_imp >= PATIENCE:
        print("早停：连续 %d 轮无提升" % PATIENCE)
        break

print("\n最佳 bal_acc %.2f%% (epoch %d)" % (best, best_ep))
with open("/kaggle/working/dl_clean_history.json", "w") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)
print("apple_dl_model.pth 已保存 -> 下载后放到 E:\\apple\\apple_dl_model.pth")
