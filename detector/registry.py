from detector.base import CategoryAdapter
from detector.apple import AppleAdapter
from detector.bread import BreadAdapter
from detector.beverage import BeverageAdapter


class CategoryRegistry:
    def __init__(self):
        self._adapters = {}
        for cls in [AppleAdapter, BreadAdapter, BeverageAdapter]:
            adapter = cls()
            self._adapters[adapter.category_id] = adapter

    def get(self, category_id: str):
        return self._adapters.get(category_id)

    def list(self):
        return [
            {"id": a.category_id, "name": a.display_name}
            for a in self._adapters.values()
        ]
