"""Step 1: Generate YOLO detection dataset + training script"""
import sys, os, cv2, numpy as np, shutil, random
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
os.environ["YOLO_CONFIG_DIR"] = "E:\\apple\\.ultralytics"
sys.path.insert(0, "E:/apple")
for mod in list(sys.modules.keys()):
    if "apple" in mod.lower() or "defect" in mod.lower(): del sys.modules[mod]

DATA_SRC = "E:\\apple\\kaggle_data\\extracted"
DETECT_DIR = "E:\\apple\\yolo_detect_dataset"
IMG_W = 640

print("=" * 55)
print("  YOLO 检测数据集 - 自动标注生成")
print("=" * 55)

# Clean and recreate
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    p = os.path.join(DETECT_DIR, sub)
    if os.path.exists(p): shutil.rmtree(p)
    os.makedirs(p)

def gen_yolo_label(mask, img_w, img_h):
    """Convert OpenCV mask to YOLO format bounding box"""
    co, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not co: return None
    largest = max(co, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.003 * img_h * img_w: return None
    x, y, w, h = cv2.boundingRect(largest)
    # YOLO format: class x_center y_center width height (normalized 0-1)
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"

# Process all images
total = 0
success = 0
for split in ["train", "test"]:
    split_name = "train" if split == "train" else "val"
    for cls_name in ["freshapples", "rottenapples"]:
        src_dir = os.path.join(DATA_SRC, split, cls_name)
        for fname in os.listdir(src_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")): continue
            total += 1
            
            img_path = os.path.join(src_dir, fname)
            img = cv2.imread(img_path)
            if img is None: continue
            
            h, w = img.shape[:2]
            
            # Resize to 640 while keeping aspect ratio
            scale = IMG_W / w
            new_size = (IMG_W, int(round(h * scale)))
            img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            rh, rw = img_resized.shape[:2]
            
            # Use current segmentation to get mask
            mask, _ = segment_apple(cv2.GaussianBlur(img_resized, (5, 5), 0))
            if mask is None: continue
            
            label = gen_yolo_label(mask, rw, rh)
            if label is None: continue
            
            # Save image and label
            dst_img = os.path.join(DETECT_DIR, f"images/{split_name}", fname)
            dst_txt = os.path.join(DETECT_DIR, f"labels/{split_name}", Path(fname).stem + ".txt")
            
            cv2.imwrite(dst_img, img_resized)
            with open(dst_txt, "w") as f: f.write(label)
            success += 1
    
    if (total) % 1000 == 0:
        print(f"  Processed {total}... ({success} successful)")

print(f"\nTotal: {total} | Successful: {success} | Failed: {total-success}")

# Count per split
for split in ["train", "val"]:
    imgs = len(os.listdir(os.path.join(DETECT_DIR, f"images/{split}")))
    lbls = len(os.listdir(os.path.join(DETECT_DIR, f"labels/{split}")))
    print(f"  {split}: {imgs} images, {lbls} labels")

# Create data.yaml
yaml_content = f"""path: {DETECT_DIR}
train: images/train
val: images/val

nc: 1
names: ["apple"]
"""
with open(os.path.join(DETECT_DIR, "data.yaml"), "w") as f:
    f.write(yaml_content)
print(f"\ndata.yaml created at {DETECT_DIR}/data.yaml")
print("Setup complete!")
