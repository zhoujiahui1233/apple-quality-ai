# ============================================================
# 苹果 YOLO 检测模型重训（Kaggle GPU 一键版）
# 使用: 在 Kaggle Notebook 新建代码, 粘贴本文件全部内容
# 前置: 上传 yolo_detect_dataset_v2.zip 到 Kaggle (/kaggle/input/)
# ============================================================
import os, zipfile, shutil

# ---- 1. 定位并解压数据集 ----
print("查找数据集 zip...")
zip_path = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f.endswith(".zip") and "yolo" in f.lower():
            zip_path = os.path.join(root, f)
            break
    if zip_path:
        break

if zip_path is None:
    print("未找到 zip！请确认已上传 yolo_detect_dataset_v2.zip")
    print("当前 /kaggle/input 内容：", os.listdir("/kaggle/input"))
    raise SystemExit

print("找到:", zip_path)
DST = "/kaggle/working/yolo_detect_dataset"
os.makedirs(DST, exist_ok=True)
print("解压中（约2-3分钟）...")
with zipfile.ZipFile(zip_path) as z:
    z.extractall(DST)
print("解压完成")

# ---- 2. 修正 data.yaml 路径 ----
yaml_path = os.path.join(DST, "data.yaml")
with open(yaml_path, encoding="utf-8") as f:
    content = f.read()
content = content.replace("path: E:\\apple\\yolo_detect_dataset", "path: " + DST.replace("\\", "/"))
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(content)
print("data.yaml 已修正")
print(open(yaml_path, encoding="utf-8").read())

# ---- 3. 统计数据 ----
n_train_img = len(os.listdir(os.path.join(DST, "images", "train")))
n_train_lbl = len(os.listdir(os.path.join(DST, "labels", "train")))
n_val_img = len(os.listdir(os.path.join(DST, "images", "val")))
print(f"train: {n_train_img} imgs / {n_train_lbl} lbls")
print(f"val:   {n_val_img} imgs")

# ---- 4. 训练 ----
from ultralytics import YOLO
model = YOLO("yolo11n.pt")   # 预训练权重, Kaggle 会自动下载

print("开始训练...")
results = model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    patience=20,          # 20轮无提升自动早停
    name="apple_detect_v3",
    exist_ok=True,
    plots=True,           # 生成训练曲线
)

# ---- 5. 保存 best.pt 到工作区根目录方便下载 ----
best = "/kaggle/working/apple_detect_v3/weights/best.pt"
if os.path.exists(best):
    shutil.copy2(best, "/kaggle/working/best.pt")
    print("best.pt 已保存 -> /kaggle/working/best.pt")
    print("大小: %.1f MB" % (os.path.getsize(best)/1e6))
    print("下载: 右侧 Output 点 best.pt 下载, 存到 E:\\apple\\yolo_detect_best.pt 覆盖")
else:
    print("未找到 best.pt！")
    print(os.listdir("/kaggle/working/apple_detect_v3/weights/") if os.path.exists("/kaggle/working/apple_detect_v3/weights/") else "no weights dir")
