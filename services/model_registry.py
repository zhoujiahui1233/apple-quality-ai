import json
from datetime import datetime
import config


class ModelRegistry:
    FILE = config.ROOT_DIR / "models_registry.json"

    def __init__(self):
        self._data = self._load()

    def _load(self):
        if self.FILE.exists():
            with open(self.FILE, encoding="utf-8") as f:
                return json.load(f)
        return {"models": {}}

    def save(self):
        with open(self.FILE, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def register(self, category: str, model_type: str,
                 accuracy: float, data_size: int) -> dict:
        key = f"{category}_{model_type}"
        version = self._next_version(key)
        entry = {
            "version": version,
            "accuracy": accuracy,
            "data_size": data_size,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "active": True,
        }
        if key not in self._data["models"]:
            self._data["models"][key] = []
        # Deactivate older versions
        for e in self._data["models"][key]:
            e["active"] = False
        self._data["models"][key].append(entry)
        self.save()
        return entry

    def _next_version(self, key) -> str:
        entries = self._data["models"].get(key, [])
        return f"v{len(entries) + 1}"

    def history(self) -> dict:
        return self._data["models"]

    def active(self, category: str, model_type: str):
        key = f"{category}_{model_type}"
        for e in reversed(self._data["models"].get(key, [])):
            if e.get("active"):
                return e
        return None

    def rollback(self, category: str, model_type: str, version: str):
        key = f"{category}_{model_type}"
        found = False
        for e in self._data["models"].get(key, []):
            e["active"] = (e["version"] == version)
            if e["version"] == version:
                found = True
        self.save()
        return found
