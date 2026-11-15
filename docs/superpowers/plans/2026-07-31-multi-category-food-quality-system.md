# 多品类食品品质检测系统 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有苹果缺陷检测系统升级为覆盖苹果/面包/饮料的多品类食品品质检测平台，含数据闭环、显著性分析、部署稳定性，作为食品科学本科毕设。

**Architecture:** 品类适配器插件化架构。detector/ 包定义统一接口并承载各品类算法；services/ 包承载业务逻辑（检测编排、数据集、统计）；Flask 路由只做转发；前端提供品类切换。

**Tech Stack:** Python 3, Flask, OpenCV, NumPy, scikit-learn, PyTorch, Ultralytics YOLO, scipy, pytest

**前置:** 现有系统已可运行（app.py 在 5020 端口），设计文档见 docs/superpowers/specs/2026-07-31-food-quality-system-design.md

---

## 文件结构总览

- Create: `detector/__init__.py` - 检测引擎包
- Create: `detector/base.py` - 品类适配器抽象基类 + 数据结构
- Create: `detector/registry.py` - 品类注册表
- Create: `detector/pipeline.py` - 统一检测管线
- Create: `detector/apple.py` - 苹果插件（迁移现有逻辑）
- Create: `detector/bread.py` - 面包插件
- Create: `detector/beverage.py` - 饮料插件
- Create: `detector/visualizers.py` - 可视化工具
- Create: `services/__init__.py` - 业务逻辑包
- Create: `services/detection_service.py` - 检测编排
- Create: `services/dataset_service.py` - 数据集管理
- Create: `services/stats_service.py` - 统计与显著性分析
- Create: `services/model_registry.py` - 模型版本管理
- Create: `tests/` - pytest 测试目录
- Modify: `app.py` - 路由瘦身，接入服务层
- Modify: `config.py` - 扩展品类配置
- Modify: `templates/base.html`, `templates/index.html`, `templates/dataset.html`, `templates/stats.html` - 前端品类切换
- Create: `templates/significance.html` - 显著性分析页
- Create: `start.bat` - 一键启动
- Create: `scripts/backup.py` - 数据库备份

---

## 阶段 0: 基础设施（测试框架 + 包结构）

### Task 0.1: 创建包结构与 pytest

**Files:**
- Create: `detector/__init__.py`
- Create: `services/__init__.py`
- Create: `tests/__init__.py`
- Modify: `requirements.txt` 增加 pytest, scipy

- [ ] **Step 1: 创建包目录与空 __init__.py**

```powershell
New-Item -ItemType Directory -Path "E:\apple\detector" -Force
New-Item -ItemType Directory -Path "E:\apple\services" -Force
New-Item -ItemType Directory -Path "E:\apple\tests" -Force
New-Item -ItemType File -Path "E:\apple\detector\__init__.py" -Force
New-Item -ItemType File -Path "E:\apple\services\__init__.py" -Force
New-Item -ItemType File -Path "E:\apple\tests\__init__.py" -Force
```

- [ ] **Step 2: 更新 requirements.txt**

```
flask>=3.0
flask-sqlalchemy>=3.0
opencv-python>=4.0
numpy>=1.20
scikit-learn>=1.0
torch>=2.0
ultralytics>=8.0
scipy>=1.10
pytest>=7.0
```

- [ ] **Step 3: 验证导入**

Run: `python -c "import detector, services; print('OK')"`
Expected: OK

## 阶段 1: 品类适配器接口 + 统一管线

### Task 1.1: 定义数据结构与抽象基类

**Files:**
- Create: `detector/base.py`

- [ ] **Step 1: 写测试 tests/test_base.py**

```python
from detector.base import CategoryAdapter, DetectionResult, GradeResult

def test_detection_result_defaults():
    r = DetectionResult(defects=[], defect_ratio=0.0)
    assert r.defect_count == 0
    assert r.severity == "none"

def test_grade_result():
    g = GradeResult(grade="good", label="优等", color=(0, 200, 0))
    assert g.grade == "good"
    assert g.label == "优等"
```

- [ ] **Step 2: 实现 detector/base.py**

```python
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
```

- [ ] **Step 3: 运行测试**

Run: `python -m pytest tests/test_base.py -v`
Expected: PASS (2 passed)
### Task 1.2: 品类注册表

**Files:**
- Create: `detector/registry.py`

- [ ] **Step 1: 写测试 tests/test_registry.py**

```python
from detector.registry import CategoryRegistry


def test_register_and_get():
    reg = CategoryRegistry()
    assert reg.get("apple") is not None
    assert reg.get("bread") is not None
    assert reg.get("beverage") is not None
    assert reg.get("unknown") is None
    assert len(reg.list()) == 3
```

- [ ] **Step 2: 实现 detector/registry.py**

```python
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
```

> 注: 依赖 AppleAdapter/BreadAdapter/BeverageAdapter，须在 Task 2/3/4 完成前保持占位可导入（先建空类，后续填充）。

- [ ] **Step 3: 创建占位插件类（使注册表可导入）**

创建 `detector/apple.py`, `detector/bread.py`, `detector/beverage.py`，各含：

```python
from detector.base import CategoryAdapter, DetectionResult, GradeResult
import numpy as np


class AppleAdapter(CategoryAdapter):
    category_id = "apple"
    display_name = "苹果"

    def detect_surface(self, img_bgr, mask, contour):
        raise NotImplementedError("Phase 2")

    def classify_grade(self, result):
        raise NotImplementedError("Phase 2")

    def visualize(self, img_bgr, result, grade):
        raise NotImplementedError("Phase 2")
```

- [ ] **Step 4: 运行测试**

Run: `python -m pytest tests/ -v`
Expected: PASS (3 passed)

## 阶段 2: 苹果插件（迁移现有逻辑）

### Task 2.1: 迁移苹果检测核心

**Files:**
- Modify: `detector/apple.py`

- [ ] **Step 1: 从 apple_defect_detector.py 迁移核心逻辑**

将现有 `segment_apple`, `detect_defects`, `_is_shadow`, `classify_quality`, `visualize_result` 逻辑迁移到 AppleAdapter 中，签名适配 base.py 接口。保留对 `config.APPLE_HSV_RANGES` 的使用。

```python
from detector.base import CategoryAdapter, DetectionResult, DefectRegion, GradeResult
import cv2
import numpy as np
import config


class AppleAdapter(CategoryAdapter):
    category_id = "apple"
    display_name = "苹果"

    def detect_surface(self, img_bgr, mask, contour):
        # 迁移: detect_defects + _is_shadow + 轮廓收集
        defect_mask, defect_contours = self._detect_defects(img_bgr, mask)
        defects = []
        for c in defect_contours:
            if self._is_shadow(c, img_bgr, mask):
                continue
            x, y, w, h = cv2.boundingRect(c)
            kind = self._classify_defect_kind(img_bgr, c)
            defects.append(DefectRegion(
                contour=c, bbox=(x, y, w, h), kind=kind,
                area=int(cv2.contourArea(c)), severity="medium"
            ))
        apple_pixels = int(np.sum(mask > 0))
        defect_pixels = int(np.sum(defect_mask > 0))
        ratio = defect_pixels / apple_pixels if apple_pixels else 0.0
        return DetectionResult(defects=defects, defect_ratio=ratio,
                               apple_mask=mask, defect_mask=defect_mask)

    def classify_grade(self, result):
        ratio = result.defect_ratio
        serious = any(d.kind in ("rot", "mold") for d in result.defects)
        if ratio < config.PREMIUM_THRESHOLD and not serious:
            return GradeResult("premium", "优等", (0, 200, 0))
        if ratio < config.STANDARD_THRESHOLD and not serious:
            return GradeResult("standard", "合格", (0, 255, 255))
        return GradeResult("reject", "不合格", (0, 0, 255))
```

- [ ] **Step 2: 用现有测试图片验证苹果检测仍工作**

Run: `python -c "from detector.registry import CategoryRegistry; r=CategoryRegistry(); a=r.get('apple'); print(a.category_id, a.display_name)"`
Expected: `apple 苹果`

- [ ] **Step 3: 回归验证上传接口**

启动服务后上传 `images/test_bad_apple.jpg`，确认返回 grade 字段非空。

## 阶段 3: 面包插件

### Task 3.1: 面包表面检测

**Files:**
- Modify: `detector/bread.py`

- [ ] **Step 1: 写测试 tests/test_bread.py（用合成图像）**

```python
import numpy as np
from detector.bread import BreadAdapter


def _make_bread_image(with_mold=False, with_burn=False):
    img = np.full((200, 200, 3), (210, 180, 140), dtype=np.uint8)  # 面包色
    if with_mold:
        img[80:120, 80:120] = (90, 160, 90)  # 绿色霉斑
    if with_burn:
        img[30:60, 30:60] = (40, 20, 10)     # 深褐焦糊
    return img


def test_bread_clean():
    a = BreadAdapter()
    img = _make_bread_image()
    res = a.detect_surface(img, np.ones((200, 200), np.uint8) * 255, None)
    assert res.defect_count == 0
    assert a.classify_grade(res).grade == "premium"


def test_bread_mold_detected():
    a = BreadAdapter()
    img = _make_bread_image(with_mold=True)
    mask = np.ones((200, 200), np.uint8) * 255
    res = a.detect_surface(img, mask, None)
    assert res.defect_count >= 1
    assert any(d.kind == "mold" for d in res.defects)
    assert a.classify_grade(res).grade == "reject"
```

- [ ] **Step 2: 实现 bread.py 霉斑/焦糊/形状检测**

```python
from detector.base import CategoryAdapter, DetectionResult, DefectRegion, GradeResult
import cv2
import numpy as np
import config


class BreadAdapter(CategoryAdapter):
    category_id = "bread"
    display_name = "面包"

    MOLD_RANGES = [
        (np.array([35, 40, 60]), np.array([90, 255, 200])),   # 绿霉
        (np.array([0, 0, 180]), np.array([180, 30, 255])),    # 白霉（高亮偏白）
        (np.array([0, 0, 100]), np.array([180, 40, 180])),    # 灰霉
    ]
    BURN_RANGES = [
        (np.array([0, 60, 0]), np.array([30, 255, 90])),      # 焦糊深褐
    ]

    def detect_surface(self, img_bgr, mask, contour):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        defects = []

        # 霉斑
        mold_mask = np.zeros(mask.shape, np.uint8)
        for lower, upper in self.MOLD_RANGES:
            mold_mask = cv2.bitwise_or(mold_mask,
                cv2.inRange(hsv, lower, upper))
        mold_mask = cv2.bitwise_and(mold_mask, mask)
        mold_mask = self._filter_small(mold_mask)
        for c in self._contours_of(mold_mask):
            x, y, w, h = cv2.boundingRect(c)
            defects.append(DefectRegion(c, (x, y, w, h), "mold",
                                        int(cv2.contourArea(c)), "high"))

        # 焦糊
        burn_mask = np.zeros(mask.shape, np.uint8)
        for lower, upper in self.BURN_RANGES:
            burn_mask = cv2.bitwise_or(burn_mask,
                cv2.inRange(hsv, lower, upper))
        burn_mask = cv2.bitwise_and(burn_mask, mask)
        burn_mask = self._filter_small(burn_mask)
        for c in self._contours_of(burn_mask):
            x, y, w, h = cv2.boundingRect(c)
            defects.append(DefectRegion(c, (x, y, w, h), "burn",
                                        int(cv2.contourArea(c)), "medium"))

        total_pixels = int(np.sum(mask > 0)) or 1
        defect_pixels = sum(d.area for d in defects)
        return DetectionResult(defects=defects,
                               defect_ratio=defect_pixels / total_pixels,
                               apple_mask=mask, defect_mask=mold_mask + burn_mask)

    def classify_grade(self, result):
        has_mold = any(d.kind == "mold" for d in result.defects)
        if has_mold or result.defect_ratio >= config.STANDARD_THRESHOLD:
            return GradeResult("reject", "不合格", (0, 0, 255))
        if result.defect_ratio >= config.PREMIUM_THRESHOLD:
            return GradeResult("standard", "合格", (0, 255, 255))
        return GradeResult("premium", "优等", (0, 200, 0))

    def visualize(self, img_bgr, result, grade):
        out = img_bgr.copy()
        for d in result.defects:
            x, y, w, h = d.bbox
            color = (0, 0, 255) if d.kind == "mold" else (0, 165, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, d.kind, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return out

    def _filter_small(self, mask, min_area=80):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out

    def _contours_of(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in cnts if cv2.contourArea(c) >= 80]
```

- [ ] **Step 3: 运行面包测试**

Run: `python -m pytest tests/test_bread.py -v`
Expected: PASS (2 passed)

## 阶段 4: 饮料插件

### Task 4.1: 瓶装饮料表面检测

**Files:**
- Modify: `detector/beverage.py`

- [ ] **Step 1: 写测试 tests/test_beverage.py（合成瓶身图像）**

```python
import numpy as np
from detector.beverage import BeverageAdapter


def _make_bottle(liquid_level=0.5, cap_ok=True):
    img = np.full((200, 100, 3), (220, 220, 220), dtype=np.uint8)  # 瓶身透明
    y = int(200 * liquid_level)
    img[y:, :] = (120, 100, 40)          # 液面以下为饮料色
    if not cap_ok:
        img[5:20, 40:60] = (0, 0, 0)     # 瓶盖缺失区域
    return img


def test_beverage_level_ok():
    a = BeverageAdapter()
    img = _make_bottle(liquid_level=0.5)
    res = a.detect_surface(img, np.ones((200, 100), np.uint8) * 255, None)
    assert a.classify_grade(res).grade == "premium"
```

- [ ] **Step 2: 实现 beverage.py 液位/瓶盖/污渍检测**

```python
from detector.base import CategoryAdapter, DetectionResult, DefectRegion, GradeResult
import cv2
import numpy as np
import config


class BeverageAdapter(CategoryAdapter):
    category_id = "beverage"
    display_name = "饮料"

    def detect_surface(self, img_bgr, mask, contour):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        defects = []

        # 液位检测: 每列找颜色梯度突变最大的行
        h, w = img_bgr.shape[:2]
        column_levels = []
        for x in range(int(w * 0.3), int(w * 0.7)):
            col = gray[:, x]
            diff = np.abs(np.diff(col.astype(np.int16)))
            if diff.max() > 20:
                column_levels.append(np.argmax(diff))
        if column_levels:
            level_ratio = np.mean(column_levels) / h
            if level_ratio < 0.35 or level_ratio > 0.75:
                defects.append(DefectRegion(
                    None, (0, int(level_ratio * h), w, 4), "level",
                    int(w * 4), "medium"))

        # 污渍/划痕: 瓶身区域高梯度异常块
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.bitwise_and(edges, mask)
        for c in self._contours_of(edges):
            if cv2.contourArea(c) < 30:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            defects.append(DefectRegion(c, (x, y, ww, hh), "stain",
                                        int(cv2.contourArea(c)), "low"))

        total_pixels = int(np.sum(mask > 0)) or 1
        defect_pixels = sum(d.area for d in defects)
        return DetectionResult(defects=defects,
                               defect_ratio=defect_pixels / total_pixels,
                               apple_mask=mask)

    def classify_grade(self, result):
        has_level = any(d.kind == "level" for d in result.defects)
        if has_level or result.defect_ratio >= config.STANDARD_THRESHOLD:
            return GradeResult("reject", "不合格", (0, 0, 255))
        if result.defect_ratio >= config.PREMIUM_THRESHOLD:
            return GradeResult("standard", "合格", (0, 255, 255))
        return GradeResult("premium", "优等", (0, 200, 0))

    def visualize(self, img_bgr, result, grade):
        out = img_bgr.copy()
        for d in result.defects:
            if d.bbox:
                x, y, w, h = d.bbox
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(out, d.kind, (x, max(0, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return out

    def _contours_of(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        return list(cnts)
```

- [ ] **Step 3: 运行饮料测试**

Run: `python -m pytest tests/test_beverage.py -v`
Expected: PASS (1 passed)
## 阶段 5: 统一检测管线 + 服务层

### Task 5.1: 统一检测管线 pipeline.py

**Files:**
- Create: `detector/pipeline.py`

- [ ] **Step 1: 写测试 tests/test_pipeline.py**

```python
import numpy as np
from detector.pipeline import run_detection


def test_run_detection_apple():
    img = np.full((200, 200, 3), (40, 40, 200), dtype=np.uint8)  # 红色苹果
    result = run_detection(img, "apple")
    assert result["category"] == "apple"
    assert "grade" in result
    assert "grade_label" in result
    assert "defect_ratio" in result
    assert "visualized" in result
```

- [ ] **Step 2: 实现 pipeline.py**

```python
import cv2
import numpy as np
from detector.registry import CategoryRegistry
from apple_defect_detector import segment_apple


_registry = CategoryRegistry()


def run_detection(img_bgr: np.ndarray, category: str) -> dict:
    """统一检测管线: 预处理→分割→品类检测→分级→可视化"""
    adapter = _registry.get(category)
    if adapter is None:
        raise ValueError(f"Unknown category: {category}")

    # 预处理 + 定位（苹果类用现有分割，面包/饮料用整图掩码）
    if category == "apple":
        mask, contour = segment_apple(img_bgr)
        if mask is None:
            return {"category": category, "error": "未检测到食品区域",
                    "grade": None, "grade_label": "未检出", "defect_ratio": 0.0}
    else:
        mask = np.ones(img_bgr.shape[:2], np.uint8) * 255
        contour = None

    result = adapter.detect_surface(img_bgr, mask, contour)
    grade = adapter.classify_grade(result)
    visualized = adapter.visualize(img_bgr, result, grade)

    return {
        "category": category,
        "grade": grade.grade,
        "grade_label": grade.label,
        "grade_color": grade.color,
        "defect_ratio": round(result.defect_ratio, 4),
        "defect_count": result.defect_count,
        "defects": [d.kind for d in result.defects],
        "visualized": visualized,
    }
```

- [ ] **Step 3: 运行测试**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS (1 passed)

### Task 5.2: 检测编排服务

**Files:**
- Create: `services/detection_service.py`

- [ ] **Step 1: 实现 detection_service.py**

```python
import uuid
from pathlib import Path
import cv2
import config
from detector.pipeline import run_detection
from models import db, DetectionRecord


class DetectionService:
    def process_upload(self, file_storage, category: str) -> dict:
        ext = Path(file_storage.filename).suffix.lower() or ".jpg"
        unique_name = str(uuid.uuid4()) + ext
        upload_path = config.UPLOAD_DIR / unique_name
        file_storage.save(str(upload_path))

        img = cv2.imread(str(upload_path))
        if img is None:
            return {"error": "无法解析图片文件"}

        # 自动缩放
        h, w = img.shape[:2]
        if max(w, h) > 1600:
            scale = 1600.0 / max(w, h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(upload_path), img)

        pipe = run_detection(img, category)
        if "error" in pipe:
            return pipe

        # 保存可视化结果
        result_filename = f"result_{unique_name}"
        cv2.imwrite(str(config.UPLOAD_DIR / result_filename),
                    pipe["visualized"])

        # 入库
        record = DetectionRecord(
            filename=unique_name,
            result_filename=result_filename,
            grade=pipe["grade_label"],
            grade_color=str(pipe["grade_color"]),
            defect_ratio=pipe["defect_ratio"],
            defect_count=pipe["defect_count"],
        )
        db.session.add(record)
        db.session.commit()

        return {
            "original_image": f"/static/uploads/{unique_name}",
            "result_image": f"/static/uploads/{result_filename}",
            "grade": pipe["grade_label"],
            "grade_en": pipe["grade"],
            "defect_ratio": f"{pipe['defect_ratio'] * 100:.2f}%",
            "defect_count": pipe["defect_count"],
            "defects": pipe["defects"],
            "category": pipe["category"],
        }
```

- [ ] **Step 2: app.py 的 /upload 路由改用 DetectionService**

```python
from services.detection_service import DetectionService

detection_service = DetectionService()

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "未找到图片文件"}), 400
    category = request.form.get("category", "apple")
    result = detection_service.process_upload(request.files["file"], category)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)
```

- [ ] **Step 3: 回归测试上传接口**

上传 `images/test_bad_apple.jpg`，确认返回正常 JSON 且数据库有记录。

## 阶段 6: 数据闭环 + 模型版本管理

### Task 6.1: 数据集服务

**Files:**
- Create: `services/dataset_service.py`

- [ ] **Step 1: 实现 dataset_service.py**

```python
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
```

- [ ] **Step 2: 数据集路由改用 DatasetService**

`/api/collect`, `/api/dataset/list` 改为调用 DatasetService，并支持 label 参数。

### Task 6.2: 模型版本管理

**Files:**
- Create: `services/model_registry.py`

- [ ] **Step 1: 实现 model_registry.py**

```python
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
                 accuracy: float, data_size: int):
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
        self._data["models"][key].append(entry)
        self.save()
        return entry

    def _next_version(self, key) -> str:
        entries = self._data["models"].get(key, [])
        n = len(entries) + 1
        return f"v{n}"

    def history(self) -> dict:
        return self._data["models"]

    def rollback(self, category: str, model_type: str, version: str):
        key = f"{category}_{model_type}"
        for e in self._data["models"].get(key, []):
            e["active"] = (e["version"] == version)
        self.save()
```

- [ ] **Step 2: 训练脚本接入 ModelRegistry**

训练完成后调用 `registry.register(category, model_type, acc, data_size)` 记录版本。

## 阶段 7: 显著性分析

### Task 7.1: 统计服务

**Files:**
- Create: `services/stats_service.py`

- [ ] **Step 1: 写测试 tests/test_stats.py**

```python
import numpy as np
from services.stats_service import mcnemar_test, kappa_score, confusion_matrix_stats


def test_kappa_perfect():
    y1 = np.array([1, 0, 1, 0, 1])
    y2 = np.array([1, 0, 1, 0, 1])
    k, _ = kappa_score(y1, y2)
    assert k == 1.0


def test_mcnemar_significant():
    # 构造显著差异的配对数据
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    y1 = np.array([1, 1, 1, 1, 0, 0, 0, 0])   # 模型1: 4错
    y2 = np.array([1, 1, 1, 1, 1, 1, 0, 0])   # 模型2: 2错
    stat, p = mcnemar_test(y1, y2, y_true)
    assert p < 0.05
```

- [ ] **Step 2: 实现 stats_service.py**

```python
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def mcnemar_test(y_pred1, y_pred2, y_true):
    """McNemar 检验: 两模型预测差异显著性"""
    e1 = (y_pred1 != y_true)
    e2 = (y_pred2 != y_true)
    b = np.sum(e1 & ~e2)   # 模型1错、模型2对
    c = np.sum(~e1 & e2)   # 模型1对、模型2错
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p


def kappa_score(y1, y2):
    """Cohen Kappa 一致性"""
    return cohen_kappa_score(y1, y2), None


def confusion_matrix_stats(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    acc = tp.sum() / max(cm.sum(), 1)
    return {"accuracy": acc, "precision": precision.tolist(),
            "recall": recall.tolist(), "f1": f1.tolist(), "cm": cm.tolist()}
```

- [ ] **Step 3: 运行测试**

Run: `python -m pytest tests/test_stats.py -v`
Expected: PASS (2 passed)

### Task 7.2: 显著性分析页

**Files:**
- Create: `templates/significance.html`
- Modify: `app.py` 增加 /significance 路由

- [ ] **Step 1: 实现 /significance 路由**

```python
@app.route("/significance")
def significance():
    return render_template("significance.html", active_tab="significance")
```

- [ ] **Step 2: significance.html 展示四模型×三品类准确率矩阵 + 显著性标记**

页面用 Chart.js 渲染柱状图，数据来自 /api/stats/models API（返回各品类各模型 acc/precision/recall/f1 + McNemar 显著性矩阵）。先以占位数据实现，真实数据接入训练结果。

## 阶段 8: 前端品类切换

### Task 8.1: 品类切换器

**Files:**
- Modify: `templates/base.html`, `templates/index.html`, `templates/dataset.html`

- [ ] **Step 1: base.html 顶部加品类切换器**

```html
<div class="category-switch">
  <button class="cat-btn active" data-cat="apple" onclick="switchCategory('apple')">🍎 苹果</button>
  <button class="cat-btn" data-cat="bread" onclick="switchCategory('bread')">🍞 面包</button>
  <button class="cat-btn" data-cat="beverage" onclick="switchCategory('beverage')">🥤 饮料</button>
</div>
```

- [ ] **Step 2: 用 localStorage 记住选择**

```javascript
function switchCategory(cat) {
  localStorage.setItem("category", cat);
  document.querySelectorAll(".cat-btn").forEach(function(b) {
    b.classList.toggle("active", b.dataset.cat === cat);
  });
  // 检测页随品类切换显示对应提示
  if (window.onCategoryChange) window.onCategoryChange(cat);
}
var savedCat = localStorage.getItem("category") || "apple";
switchCategory(savedCat);
```

- [ ] **Step 3: 上传时携带 category 参数**

`up()` 函数中 `fd.append("category", localStorage.getItem("category") || "apple")`

- [ ] **Step 4: 数据集页按品类→标签两级展示**

### Task 8.2: 移动端适配

- [ ] **Step 1: 手机底部导航增加"显著性"入口**
- [ ] **Step 2: 品类切换器在移动端横向滚动可点**

## 阶段 9: 部署稳定性

### Task 9.1: 一键启动脚本

**Files:**
- Create: `start.bat`

```bat
@echo off
chcp 65001 >nul
title 智果 AI 品控系统
cd /d %~dp0
echo [1/3] 检查端口...
netstat -ano | findstr ":5020" >nul 2>&1
if not errorlevel 1 (
  echo 服务已在运行，浏览器打开 http://127.0.0.1:5020
  start http://127.0.0.1:5020
  exit /b
)
echo [2/3] 启动服务器...
start "" python app.py
echo [3/3] 等待启动...
timeout /t 5 /nobreak >nul
start http://127.0.0.1:5020
echo 启动完成
```

### Task 9.2: 数据库备份脚本

**Files:**
- Create: `scripts/backup.py`

```python
import shutil
from datetime import datetime
from pathlib import Path
import config

BACKUP_DIR = config.ROOT_DIR / "backups"
DB_PATH = config.ROOT_DIR / "instance" / "database.db"

def backup():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"database_{ts}.db"
    if DB_PATH.exists():
        shutil.copy2(DB_PATH, dst)
    # 清理30天前的备份
    import os, time
    cutoff = time.time() - 30 * 86400
    for f in BACKUP_DIR.glob("database_*.db"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
    print(f"备份完成: {dst.name}")

if __name__ == "__main__":
    backup()
```

### Task 9.3: 日志轮转 + 自动重启说明

- [ ] **Step 1: app.py 配置 RotatingFileHandler**

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    str(config.ROOT_DIR / "logs" / "app.log"),
    maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
logger.addHandler(handler)
```

- [ ] **Step 2: 文档记录 Windows 任务计划程序自动重启配置**

## 阶段 10: 文档 + 毕设材料

### Task 10.1: 项目文档

**Files:**
- Create: `README.md` - 项目介绍/快速开始
- Create: `docs/architecture.md` - 架构说明
- Create: `docs/userguide.md` - 使用手册

### Task 10.2: 实验数据采集指引

- [ ] 记录样本采集要求（每品类 >= 30 份，含合格/不合格）
- [ ] 记录人工感官评价表模板（供 Kappa 验证）
- [ ] 记录理化指标测定方法（苹果硬度/TSS、面包水分、饮料 pH）

---

## 验收清单

- [ ] 三类食品均可检测分级，前端品类切换正常
- [ ] 四模型×三品类准确率矩阵可导出
- [ ] McNemar/Kappa 检验可用
- [ ] 数据闭环可运转（拍→存→训→升级）
- [ ] 模型版本可回滚
- [ ] start.bat 一键启动
- [ ] 数据库自动备份
- [ ] README/架构/使用手册/论文大纲齐全
- [ ] 全部 pytest 通过
