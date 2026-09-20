# ============================================================
# 教师模型训练（Kaggle GPU 一键版）
# 用途: 用 400 张人工复核种子训练"自动标注教师模型"
# 步骤: 上传 teacher_dataset.zip 到 Kaggle -> 新建 Notebook 用 GPU
#       -> 粘贴本文件全部代码运行 -> 下载 teacher_defect_best.pt
#       存到本地 E:\apple\teacher_defect_best.pt
# ============================================================
import os, glob, zipfile, shutil

# ================= 智能定位数据（zip 或 已解压目录都行） =================
def locate_input():
    """返回 (数据根目录, 是否为已解压文件夹)。"""
    # 情况1: input 里有 zip
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    print("候选 zip:", zips)
    if zips:
        return max(zips, key=lambda p: os.path.getsize(p)), True
    # 情况2: input 里直接有 data.yaml（Kaggle 可能已自动解压）
    yamls = glob.glob("/kaggle/input/**/data.yaml", recursive=True)
    print("候选 data.yaml:", yamls)
    if yamls:
        return os.path.dirname(yamls[0]), False
    raise SystemExit(
        "input 里既没有 zip 也没有 data.yaml。\n"
        "请在笔记本右上角点 +Add Input，搜索你上传的 teacher_dataset，点 + 加进来。\n"
        "当前 /kaggle/input 内容: " + str(os.listdir("/kaggle/input")))

src, is_zip = locate_input()
DST = "/kaggle/working/teacher_dataset"
if is_zip:
    os.makedirs(DST, exist_ok=True)
    print("解压", src, "...")
    with zipfile.ZipFile(src) as z:
        z.extractall(DST)
    # 处理嵌套目录
    for d in os.listdir(DST):
        sub = os.path.join(DST, d)
        if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "data.yaml")):
            DST = sub
            break
else:
    # input 目录只读，复制到 working 才能改 data.yaml
    if os.path.abspath(src).startswith("/kaggle/working"):
        DST = src
    else:
        shutil.rmtree(DST, ignore_errors=True)
        print("复制数据到 working ...")
        shutil.copytree(src, DST)
print("数据目录:", DST)

# ---- 2. 修正 data.yaml 里的绝对路径 ----
yaml_path = os.path.join(DST, "data.yaml")
content = open(yaml_path, encoding="utf-8").read()
import re
content = re.sub(r"path:\s*\S+", "path: " + DST, content)
open(yaml_path, "w", encoding="utf-8").write(content)
print(open(yaml_path, encoding="utf-8").read())

# ---- 3. 训练 ----
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(
    data=yaml_path,
    epochs=200,
    imgsz=512,          # 原图 256 小图，512 足够
    batch=32,
    device=0,
    workers=4,
    patience=40,
    name="teacher_defect_v1",
    exist_ok=True,
    plots=True,
)

# ---- 4. 保存 ----
best = "/kaggle/working/runs/detect/teacher_defect_v1/weights/best.pt"
assert os.path.exists(best), "未找到 best.pt"
shutil.copy2(best, "/kaggle/working/teacher_defect_best.pt")
print("teacher_defect_best.pt 已保存 (%.1f MB)" % (os.path.getsize(best) / 1e6))
print("下载它，存到 E:\\apple\\teacher_defect_best.pt")
