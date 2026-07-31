# -*- coding: utf-8 -*-
"""
数据备份脚本

用法:
  python scripts/backup.py            # 备份代码+数据(不含大模型)
  python scripts/backup.py --full     # 全量备份(含模型权重)
  python scripts/backup.py --list     # 列出已有备份
  python scripts/backup.py --clean 5  # 删除保留 5 个最新备份以外的旧备份
"""
import argparse
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = ROOT / "backups"

CODE_ITEMS = [
    "app.py", "config.py", "models.py",
    "apple_defect_detector.py",
    "detector", "services", "templates", "static", "tests",
]
MODEL_ITEMS = [
    "apple_scaler.pkl", "apple_svm_model.dat", "apple_svm_model.pkl",
    "apple_rf_model.pkl", "apple_dl_model.pth",
    "yolo_detect_best.pt", "yolo_output",
]
DATA_ITEMS = ["instance", "dataset_collected", "models_registry.json"]

SKIP_PARTS = {"__pycache__", ".pytest_cache", ".idea", ".git"}


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def make_zip(full: bool) -> Path:
    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"backup_{stamp}{'_full' if full else ''}.zip"
    target = BACKUP_DIR / name
    items = CODE_ITEMS + DATA_ITEMS + (MODEL_ITEMS if full else [])
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            src = ROOT / item
            if not src.exists():
                print(f"  跳过(不存在): {item}")
                continue
            if src.is_dir():
                for f in sorted(src.rglob("*")):
                    if f.is_file() and not _should_skip(f):
                        zf.write(f, f.relative_to(ROOT))
                        count += 1
            else:
                zf.write(src, src.relative_to(ROOT))
                count += 1
    size_mb = target.stat().st_size / 1024 / 1024
    print(f"已生成备份: {target.name}  ({count} 个文件, {size_mb:.1f} MB)")
    return target


def list_backups():
    if not BACKUP_DIR.exists():
        print("还没有备份文件")
        return
    files = sorted(BACKUP_DIR.glob("backup_*.zip"), reverse=True)
    if not files:
        print("还没有备任文件")
        return
    print("已有备份:")
    for f in files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}  ({size_mb:.1f} MB)")


def clean(keep: int):
    if not BACKUP_DIR.exists():
        return
    files = sorted(BACKUP_DIR.glob("backup_*.zip"), reverse=True)
    for f in files[keep:]:
        f.unlink()
        print(f"已删除: {f.name}")


def main():
    parser = argparse.ArgumentParser(description="智果 AI 品控系统备份工具")
    parser.add_argument("--full", action="store_true", help="包含模型权重的全量备份")
    parser.add_argument("--list", action="store_true", help="列出已有备份")
    parser.add_argument("--clean", type=int, metavar="N", help="只保留最新 N 个备份")
    args = parser.parse_args()

    if args.list:
        list_backups()
        return
    if args.clean:
        clean(args.clean)
        return
    make_zip(full=args.full)


if __name__ == "__main__":
    main()
