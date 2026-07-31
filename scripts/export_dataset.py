# -*- coding: utf-8 -*-
"""
数据集导出脚本：把收集的 dataset_collected 按 80/20 切分 train/val

用法:
  python scripts/export_dataset.py                  # 全部品类
  python scripts/export_dataset.py --category apple # 单品类
  python scripts/export_dataset.py --ratio 0.8      # 自定义训练集比例
  python scripts/export_dataset.py --seed 42        # 固定随机种子保证可复现

输出:
  output/dataset_export/{category}/{split}/{label}/*.jpg
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

DATASET_DIR = config.ROOT_DIR / "dataset_collected"
OUTPUT_DIR = config.OUTPUT_DIR / "dataset_export"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def export(category: str, ratio: float, seed: int):
    cat_dir = DATASET_DIR / category
    if not cat_dir.is_dir():
        print(f"跳过 {category}: 目录不存在")
        return 0
    rng = random.Random(seed)
    total = 0
    for label_dir in sorted(cat_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        files = sorted(f for f in label_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if not files:
            continue
        rng.shuffle(files)
        n_train = max(1, int(len(files) * ratio))
        splits = {"train": files[:n_train], "val": files[n_train:]}
        for split, items in splits.items():
            for f in items:
                dst = OUTPUT_DIR / category / split / label_dir.name / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(f), str(dst))
        total += len(files)
        print(f"  {category}/{label_dir.name}: {len(files)} 张 -> train {len(splits['train'])} / val {len(splits['val'])}")
    return total


def main():
    parser = argparse.ArgumentParser(description="数据集导出（切分 train/val）")
    parser.add_argument("--category", default=None)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if OUTPUT_DIR.exists():
        shutil.rmtree(str(OUTPUT_DIR))
    OUTPUT_DIR.mkdir(parents=True)

    categories = [args.category] if args.category else ["apple", "bread", "beverage"]
    grand = 0
    for cat in categories:
        grand += export(cat, args.ratio, args.seed)
    print(f"\n导出完成: 共 {grand} 张 -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
