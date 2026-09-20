# ============================================================
# YOLO 缺陷检测 重训（Kaggle GPU 版，无泄漏切分）
# ------------------------------------------------------------
# 数据: defect_dataset_clean.zip（由 scripts/rebuild_defect_split.py 生成，
#       按同源帧聚类切分，train/val 无近重复）
# 产物: defect_detect_best_v3.pt -> 下载后放到 E:\apple\defect_detect_best_v2.pt
#       （或改 config / apple_defect_detector 里的路径）
#
# 相比旧版的关键改动：
#   1. imgsz 640 -> 1280：小目标（虫眼/小斑）在 640 下像素太少，这是旧模型
#      小缺陷召回只有 0.06~0.25 的主因之一
#   2. yolo11n -> yolo11s：容量翻倍，小目标更稳
#   3. epochs 100 -> 200 + patience 40，并增强小目标相关增广
#   4. 切分按同源簇，指标不再虚高
# ============================================================
import os, glob, zipfile, json


def locate():
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    # 按文件名认数据包，避免误取到脚本包等其它 zip
    pref = [z for z in zips if "defect" in os.path.basename(z).lower()]
    if pref:
        return max(pref, key=os.path.getsize), True
    if zips:
        return max(zips, key=os.path.getsize), True
    hits = glob.glob("/kaggle/input/**/images/train", recursive=True)
    if hits:
        return hits[0][: hits[0].rindex("/images")], False
    raise SystemExit("没找到 defect_dataset_clean，请 +Add Input。当前: "
                     + str(os.listdir("/kaggle/input")))


src, is_zip = locate()
DATA = "/kaggle/working/defect_dataset_clean"
if is_zip:
    os.makedirs(DATA, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(DATA)
    for d in os.listdir(DATA):
        sub = os.path.join(DATA, d)
        if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "images")):
            DATA = sub
            break
else:
    DATA = src
print("数据目录:", DATA)

for sp in ("train", "val"):
    ni = len(os.listdir(os.path.join(DATA, "images", sp)))
    nl = len(os.listdir(os.path.join(DATA, "labels", sp)))
    print("  %s: images %d, labels %d" % (sp, ni, nl))

# 重写 data.yaml 指向 Kaggle 路径
YAML = "/kaggle/working/defect_clean.yaml"
with open(YAML, "w", encoding="utf-8") as f:
    f.write("path: %s\ntrain: images/train\nval: images/val\nnc: 1\nnames: [\"defect\"]\n"
            % DATA)
print(open(YAML).read())

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

EPOCHS = 200
IMGSZ = 1280
BATCH = 8            # T4 16GB 下 yolo11s@1280 用 8；显存不够就降到 4
PATIENCE = 40

model = YOLO(find_weight("yolo11s.pt"))   # 预训练权重；想更快可换 yolo11n.pt

model.train(
    data=YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=PATIENCE,
    device=0,
    project="/kaggle/working/runs/detect",
    name="defect_clean_v3",
    exist_ok=True,
    pretrained=True,
    optimizer="auto",
    cos_lr=True,
    warmup_epochs=3.0,
    # 小目标相关增广
    mosaic=1.0,
    close_mosaic=15,
    scale=0.6,          # 随机缩放，制造更小的目标
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

best = "/kaggle/working/runs/detect/defect_clean_v3/weights/best.pt"
print("best.pt 存在:", os.path.exists(best))
if os.path.exists(best):
    m = YOLO(best)
    r = m.val(data=YAML, imgsz=IMGSZ, device=0)
    print("\n=== 干净 val 上的指标（按同源簇切分，无泄漏）===")
    print("mAP50: %.4f | mAP50-95: %.4f" % (r.box.map50, r.box.map))
    print("Precision: %.4f | Recall: %.4f" % (r.box.mp, r.box.mr))
    import shutil
    shutil.copy2(best, "/kaggle/working/defect_detect_best_v3.pt")
    print("已导出 /kaggle/working/defect_detect_best_v3.pt")
