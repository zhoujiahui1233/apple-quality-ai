"""YOLO Detection Training"""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ["YOLO_CONFIG_DIR"] = "E:\\apple\\.ultralytics"
from ultralytics import YOLO
import torch

DETECT_DIR = "E:\\apple\\yolo_detect_dataset"
OUTPUT_DIR = "E:\\apple\\yolo_detect_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 55)
print("  YOLO 苹果检测模型训练")
print("=" * 55)
print(f"  设备: {DEVICE}")
print(f"  数据集: {DETECT_DIR}")

# Create model from yaml
print("\n[1/3] 创建模型...")
model = YOLO("yolo11n.yaml")

print(f"  模型: YOLO11n (Nano Detection)")
print(f"  类别: 1 (apple)")
print(f"  图片尺寸: 640x640")

# Train
print("\n[2/3] 开始训练...")
print("  (从零训练，无预训练权重，需要较多epochs)")
print(f"  训练集: 5670 张 | 验证集: 1778 张")
print()

t0 = time.time()
results = model.train(
    data=os.path.join(DETECT_DIR, "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=8,
    device=DEVICE,
    project=OUTPUT_DIR,
    name="apple_detect",
    exist_ok=True,
    patience=15,
    lr0=0.01,
    augment=True,
    workers=2,
)
train_time = time.time() - t0

print(f"\n[3/3] 训练完成!")
print(f"  耗时: {train_time/60:.1f} 分钟")
print(f"  模型保存: {OUTPUT_DIR}/apple_detect/weights/best.pt")
print(f"  最后模型: {OUTPUT_DIR}/apple_detect/weights/last.pt")
