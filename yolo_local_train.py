import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ["YOLO_CONFIG_DIR"] = r"E:\apple\.ultralytics"
from ultralytics import YOLO

DETECT_DIR = r"E:\apple\yolo_detect_dataset"
OUTPUT_DIR = r"E:\apple\yolo_detect_output"
DEVICE = "cpu"
PT_PATH = r"E:\yolo11n.pt"

print("YOLO Apple Detection (CPU)")
print("CPU cores: %d" % os.cpu_count())
print("Loading model...")
model = YOLO(PT_PATH)
print("Training start (~2-3 hours)...")
t0 = time.time()
model.train(
    data=os.path.join(DETECT_DIR, "data.yaml"),
    epochs=30, imgsz=320, batch=8,
    device=DEVICE,
    project=OUTPUT_DIR, name="apple_detect",
    exist_ok=True, patience=10,
    lr0=0.005, augment=True,
    workers=8, cos_lr=True,
    warmup_epochs=3, cache=True,
)
train_time = time.time() - t0
best = os.path.join(OUTPUT_DIR, "apple_detect", "weights", "best.pt")
print("Done! %.1f min" % (train_time / 60))
if os.path.exists(best):
    import shutil
    shutil.copy2(best, r"E:\apple\yolo_detect_best.pt")
    print("Saved: yolo_detect_best.pt")
    size_mb = os.path.getsize(best) / 1e6
    print("Size: %.1f MB" % size_mb)