# ============================================================
# 苹果缺陷检测 YOLO 训练（Kaggle GPU 一键版）
# 使用: 上传 defect_dataset_yolo.zip 到 Kaggle, 粘贴本代码运行
# ============================================================
import os, zipfile, glob

# ---- 1. 找 zip 并解压 ----
zips = [z for z in glob.glob("/kaggle/input/**/*.zip", recursive=True)]
print("找到 zip:", zips)
if not zips:
    print("未找到 zip, input 内容:", os.listdir("/kaggle/input"))
    raise SystemExit
zip_path = max(zips, key=lambda p: os.path.getsize(p))
DST = "/kaggle/working/defect_dataset"
os.makedirs(DST, exist_ok=True)
print("解压", zip_path, "...")
with zipfile.ZipFile(zip_path) as z:
    z.extractall(DST)
# 处理可能的嵌套目录
for d in os.listdir(DST):
    sub = os.path.join(DST, d)
    if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "images")):
        DST = sub
        break
print("数据目录:", DST)

# ---- 2. 修 data.yaml ----
yaml_path = os.path.join(DST, "data.yaml")
if not os.path.exists(yaml_path):
    # 生成一个
    open(yaml_path, "w").write(
        f"path: {DST}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [\"defect\"]\n")
else:
    content = open(yaml_path, encoding="utf-8").read()
    content = content.replace("E:\\\\apple\\\\defect_dataset_yolo", DST).replace("E:\\apple\\defect_dataset_yolo", DST)
    open(yaml_path, "w", encoding="utf-8").write(content)
print(open(yaml_path, encoding="utf-8").read())

# ---- 3. 训练 ----
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(
    data=yaml_path,
    epochs=200,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    patience=30,
    name="defect_detect_v1",
    exist_ok=True,
    plots=True,
)

# ---- 4. 保存 best.pt ----
best = "/kaggle/working/runs/detect/defect_detect_v1/weights/best.pt"
if os.path.exists(best):
    import shutil
    shutil.copy2(best, "/kaggle/working/defect_best.pt")
    print("defect_best.pt 已保存 (%.1f MB)" % (os.path.getsize(best)/1e6))
    print("下载后存到 E:\\apple\\defect_detect_best.pt")
else:
    print("未找到 best.pt, 列出所有:")
    for root, dirs, files in os.walk("/kaggle/working"):
        for f in files:
            if f.endswith(".pt"):
                print(os.path.join(root, f))
