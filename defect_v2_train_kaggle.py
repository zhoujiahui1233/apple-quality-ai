# ============================================================
# 缺陷检测模型 v2 训练（Kaggle GPU 一键版）
# 数据: 400 人工种子 + 2026 高置信自动框 = 2426 张
# 产物: defect_detect_best_v2.pt  -> 放到 E:\apple 替换系统缺陷模型
# 步骤: 上传 defect_dataset_v2.zip 为 Dataset 并 Add
#       -> 粘贴本文件全部代码 -> 右上角 Save & Run All
#       -> 命令行拉取:
#          python -m kaggle kernels output 用户名/notebook名字 -p E:\apple\kaggle_download
# ============================================================
import os, glob, zipfile, shutil

# ---------- 智能定位数据 ----------
def locate_input():
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    if zips:
        return max(zips, key=os.path.getsize), True
    yamls = glob.glob("/kaggle/input/**/data.yaml", recursive=True)
    if yamls:
        return os.path.dirname(yamls[0]), False
    raise SystemExit("input 无 zip/data.yaml，请先 +Add Input 添加 defect_dataset_v2。内容: "
                     + str(os.listdir("/kaggle/input")))

src, is_zip = locate_input()
DST = "/kaggle/working/defect_dataset_v2"
if is_zip:
    os.makedirs(DST, exist_ok=True)
    print("解压", src, "...")
    with zipfile.ZipFile(src) as z:
        z.extractall(DST)
    for d in os.listdir(DST):
        sub = os.path.join(DST, d)
        if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "data.yaml")):
            DST = sub
            break
else:
    if os.path.abspath(src).startswith("/kaggle/working"):
        DST = src
    else:
        shutil.rmtree(DST, ignore_errors=True)
        shutil.copytree(src, DST)
print("数据目录:", DST)

# ---------- 修正路径 ----------
import re
yaml_path = os.path.join(DST, "data.yaml")
content = open(yaml_path, encoding="utf-8").read()
content = re.sub(r"path:\s*\S+", "path: " + DST, content)
open(yaml_path, "w", encoding="utf-8").write(content)
print(open(yaml_path, encoding="utf-8").read())

# ---------- 训练 ----------
from ultralytics import YOLO
# 定稿版本：yolo11s（不再换型号；以后只增数据，不改模型配置）
model = YOLO("yolo11s.pt")
model.train(
    data=yaml_path,
    epochs=200,
    imgsz=640,
    batch=32,
    device=0,
    workers=4,
    patience=40,
    name="defect_v2",
    exist_ok=True,
    plots=True,
)

best = "/kaggle/working/runs/detect/defect_v2/weights/best.pt"
assert os.path.exists(best), "未找到 best.pt"
shutil.copy2(best, "/kaggle/working/defect_detect_best_v2.pt")
print("defect_detect_best_v2.pt 已保存 (%.1f MB)" % (os.path.getsize(best) / 1e6))
