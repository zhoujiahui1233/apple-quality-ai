from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class DefectRegion:
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    kind: str          # "scar" / "rot" / "mold" / "burn" / ...
    area: int
    severity: str      # "low" / "medium" / "high"


@dataclass
class DetectionResult:
    defects: List[DefectRegion] = field(default_factory=list)
    defect_ratio: float = 0.0
    apple_mask: np.ndarray = None
    defect_mask: np.ndarray = None
    extra: dict = field(default_factory=dict)

    @property
    def defect_count(self) -> int:
        return len(self.defects)

    @property
    def severity(self) -> str:
        if self.defect_ratio >= 0.15:
            return "high"
        if self.defect_ratio >= 0.05:
            return "medium"
        return "none"


@dataclass
class GradeResult:
    grade: str        # "premium" / "standard" / "reject"
    label: str        # "优等" / "合格" / "不合格"
    color: Tuple[int, int, int]
    confidence: float = 1.0


class CategoryAdapter(ABC):
    category_id: str = ""
    display_name: str = ""

    @abstractmethod
    def detect_surface(self, img_bgr: np.ndarray, mask: np.ndarray,
                       contour: np.ndarray) -> DetectionResult:
        """表面缺陷检测"""

    @abstractmethod
    def classify_grade(self, result: DetectionResult) -> GradeResult:
        """品质分级"""

    @abstractmethod
    def visualize(self, img_bgr: np.ndarray, result: DetectionResult,
                  grade: GradeResult) -> np.ndarray:
        """结果可视化"""
