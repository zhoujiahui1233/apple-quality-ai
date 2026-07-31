import os
from datetime import datetime
import config


class DatasetService:
    BASE = config.ROOT_DIR / "dataset_collected"

    def save(self, file_storage, category: str, label: str) -> dict:
        cat_dir = self.BASE / category / label
        cat_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        ext = os.path.splitext(file_storage.filename)[1].lower() or ".jpg"
        fname = f"{label}_{ts}{ext}"
        file_storage.save(str(cat_dir / fname))
        return {"success": True, "category": category,
                "label": label, "filename": fname}

    def list(self) -> dict:
        if not self.BASE.exists():
            return {"categories": {}, "total": 0}
        result = {}
        total = 0
        for cat in sorted(os.listdir(self.BASE)):
            cat_dir = self.BASE / cat
            if not cat_dir.is_dir():
                continue
            labels = {}
            for label in sorted(os.listdir(cat_dir)):
                label_dir = cat_dir / label
                if label_dir.is_dir():
                    labels[label] = sorted(os.listdir(label_dir))
                    total += len(labels[label])
            result[cat] = labels
        return {"categories": result, "total": total}

    def delete(self, category: str, label: str, filename: str) -> bool:
        target = self.BASE / category / label / filename
        if target.exists() and target.is_file():
            target.unlink()
            return True
        return False
