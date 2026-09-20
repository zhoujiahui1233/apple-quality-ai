# ============================================================
# YOLO 分类 重训（Kaggle GPU 版，无泄漏切分）
# ------------------------------------------------------------
# 这是参与「SVM/RF/DL/YOLO 四模型投票」的那个 YOLO 分类模型
# （系统里 config.MODEL_PATHS["yolo_cls"]）。
#
# 数据: dl_dataset_clean.zip （与 dl_train_clean_kaggle.py 共用同一份数据，
#       无需再传新数据。YOLO 分类的目录格式就是 train/<类名>/，正好匹配）
# 产物: yolo_cls_best.pt -> 下载后覆盖
#       E:\apple\yolo_output\yolo_classify-2\weights\best.pt
#
# 相比旧版的关键改动：
#   1. yolo11n-cls.yaml -> yolo11n-cls.pt：旧版是从零训，现在用 ImageNet 预训练
#   2. imgsz 224 -> 320：缺陷/果面细节在 224 下太糊
#   3. epochs 20 -> 100 + 早停：旧版只跑 20 轮，明显欠训练
#   4. device cpu -> 0：旧版居然跑在 CPU 上
#   5. 切分按同源簇，指标不再虚高
# ============================================================
import os, glob, zipfile, shutil


def locate():
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    pref = [z for z in zips if "dl_dataset" in os.path.basename(z).lower()]
    if pref:
        return max(pref, key=os.path.getsize), True
    if zips:
        return max(zips, key=os.path.getsize), True
    hits = glob.glob("/kaggle/input/**/images/train", recursive=True)
    if hits:
        return hits[0][: hits[0].rindex("/images")], False
    raise SystemExit("没找到 dl_dataset_clean，请 +Add Input。当前: "
                     + str(os.listdir("/kaggle/input")))


src, is_zip = locate()
ROOTDIR = "/kaggle/working/dl_dataset_clean"
if is_zip:
    os.makedirs(ROOTDIR, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(ROOTDIR)
    for d in os.listdir(ROOTDIR):
        sub = os.path.join(ROOTDIR, d)
        if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "images")):
            ROOTDIR = sub
            break
else:
    ROOTDIR = src

# YOLO 分类要求 data 目录下直接是 train/<类名>/ 与 val/<类名>/
DATA = os.path.join(ROOTDIR, "images")
print("分类数据根目录:", DATA)
for sp in ("train", "val"):
    for cls in ("good", "bad"):
        p = os.path.join(DATA, sp, cls)
        print("  %-5s %-4s %d" % (sp, cls, len(os.listdir(p)) if os.path.isdir(p) else -1))

# ---- 确保 ultralytics 可用（Kaggle 镜像不一定预装）----
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess, sys
    print("未检测到 ultralytics，正在安装...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
    from ultralytics import YOLO


def find_weight(name):
    """优先用 +Add Input 上传的权重文件，避免 Kaggle 未开联网导致下载失败。"""
    hits = glob.glob("/kaggle/input/**/" + name, recursive=True)
    if hits:
        print("使用上传的权重:", hits[0])
        return hits[0]
    print("未找到上传的权重，尝试联网下载:", name)
    return name

EPOCHS = 100
IMGSZ = 320
BATCH = 64
PATIENCE = 20

# 预训练权重（旧版用的是 .yaml = 从零训）
model = YOLO(find_weight("yolo11n-cls.pt"))

model.train(
    data=DATA,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=PATIENCE,
    device=0,
    project="/kaggle/working/runs/classify",
    name="yolo_cls_clean",
    exist_ok=True,
    pretrained=True,
    optimizer="auto",
    cos_lr=True,
    warmup_epochs=3.0,
    dropout=0.1,
    # 分类增广
    fliplr=0.5,
    flipud=0.2,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
    scale=0.4,
    erasing=0.25,
    auto_augment="randaugment",
    workers=4,
    verbose=True,
)

best = "/kaggle/working/runs/classify/yolo_cls_clean/weights/best.pt"
print("best.pt 存在:", os.path.exists(best))
if os.path.exists(best):
    m = YOLO(best)
    r = m.val(data=DATA, imgsz=IMGSZ, device=0)
    print("\n=== 干净 val 上的指标（按同源簇切分，无泄漏）===")
    print("top1 acc: %.4f" % (r.top1 if hasattr(r, "top1") else -1))
    if hasattr(r, "top5"):
        print("top5 acc: %.4f" % r.top5)
    shutil.copy2(best, "/kaggle/working/yolo_cls_best.pt")
    print("已导出 /kaggle/working/yolo_cls_best.pt")
    print("下载后覆盖: E:\\apple\\yolo_output\\yolo_classify-2\\weights\\best.pt")
