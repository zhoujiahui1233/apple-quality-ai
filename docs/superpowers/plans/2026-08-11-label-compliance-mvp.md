# AI 预包装食品标签合规审查助手（MVP）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个上传标签照片即可输出合规审查报告的 AI 工具 MVP（暑假 3 周内完成）。

**Architecture:** PaddleOCR 提取文本 → 规则解析器结构化 → 确定性规则引擎为主 + 本地大模型（Qwen2.5-7B）RAG 语义复核为辅 → 统一格式审查报告（HTML）。

**Tech Stack:** Python 3.10、PaddleOCR 2.7.3、FastAPI、Streamlit、Chroma + bge-m3、Ollama + Qwen2.5-7B-Instruct、pytest。

**标准依据（2026-09-01 实施的新版，落地前以官方发布文本核对阈值与致敏物质清单）：** GB 7718-2025《预包装食品标签通则》、GB 28050-2025《预包装食品营养标签通则》。

---

## 文件结构

```
label-compliance/
├── requirements.txt
├── .gitignore
├── README.md
├── parser/
│   ├── __init__.py
│   ├── models.py            # LabelData / NutritionItem 数据模型
│   └── label_parser.py      # 文本 → 结构化字段
├── ocr/
│   ├── __init__.py
│   └── ocr_service.py       # PaddleOCR 封装
├── rules/
│   ├── __init__.py
│   ├── base.py              # RuleStatus / RiskLevel / RuleResult / Rule 基类
│   ├── engine.py            # RuleEngine
│   ├── r1_net_content.py
│   ├── r2_date_labeling.py
│   ├── r3_ingredient_list.py
│   ├── r4_allergen.py
│   ├── r5_nutrition_table.py
│   ├── r6_nutrition_claim.py
│   ├── r7_sc_number.py
│   ├── r8_storage_condition.py
│   └── r9_sensitive_word.py
├── llm/
│   ├── __init__.py
│   ├── prompts.py           # 系统提示词
│   ├── rag.py               # Chroma + bge-m3 检索
│   └── semantic_review.py   # Ollama 调用 + JSON 解析
├── report/
│   ├── __init__.py
│   ├── report_generator.py  # Jinja2 → HTML
│   └── template.html
├── app/
│   ├── __init__.py
│   ├── api.py               # FastAPI /api/review
│   └── ui.py                # Streamlit 前端
├── scripts/
│   ├── build_index.py       # 构建国标向量库
│   └── evaluate.py          # 测试集评估
├── data/
│   ├── standards/           # 国标 txt 全文（自行从官方渠道下载）
│   ├── test_labels.json     # 测试集标注
│   └── test_images/         # 20~30 张实拍标签
└── tests/
    ├── __init__.py
    ├── test_models.py
    ├── test_ocr.py
    ├── test_parser.py
    ├── test_engine.py
    ├── test_rule_net_content.py
    ├── test_rule_date.py
    ├── test_rule_ingredient_list.py
    ├── test_rule_allergen.py
    ├── test_rule_nutrition_table.py
    ├── test_rule_nutrition_claim.py
    ├── test_rule_sc_number.py
    ├── test_rule_storage.py
    ├── test_rule_sensitive.py
    ├── test_semantic_review.py
    ├── test_rag.py
    ├── test_report.py
    ├── test_api.py
    └── test_pipeline.py
```

---

### Task 0: 环境与仓库初始化

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`（先放标题占位，Task 15 补全）

- [ ] **Step 1: 创建目录并初始化独立 Git 仓库**

```powershell
New-Item -ItemType Directory -Force -Path "E:\apple\label-compliance"
cd E:\apple\label-compliance
git init
```

Expected: `Initialized empty Git repository in E:/apple/label-compliance/.git`

- [ ] **Step 2: 写 requirements.txt**

```text
# 核心
fastapi==0.115.6
uvicorn[standard]==0.32.1
streamlit==1.40.1
jinja2==3.1.4
pydantic==2.9.2
httpx==0.27.2
python-multipart==0.0.17

# OCR（Windows CPU 版；学校 GPU 环境请按 Paddle 官方文档换装 paddlepaddle-gpu）
paddleocr==2.7.3
paddlepaddle==2.6.2

# RAG
chromadb==0.5.15
sentence-transformers==3.2.1

# 测试
pytest==8.3.3
pillow==10.4.0
```

- [ ] **Step 3: 写 .gitignore**

```text
.venv/
__pycache__/
*.pyc
.pytest_cache/
data/vector_db/
data/test_images/
```

- [ ] **Step 4: 创建虚拟环境并安装依赖**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Expected: 所有包安装成功，无报错。若 `paddlepaddle`/`paddleocr` 安装冲突，先单独装 `pip install paddlepaddle==2.6.2 paddleocr==2.7.3` 再装其余。

- [ ] **Step 5: 在 GPU 机器上拉取大模型**

```powershell
ollama pull qwen2.5:7b-instruct
```

Expected: 输出 `success`。

- [ ] **Step 6: 首次提交**

```powershell
git add .
git commit -m "chore: project scaffold"
```

---

### Task 1: 数据模型（LabelData / RuleResult）

**Files:**
- Create: `parser/__init__.py`（空文件）
- Create: `rules/__init__.py`（空文件）
- Create: `tests/__init__.py`（空文件）
- Create: `parser/models.py`
- Create: `rules/base.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: 写失败测试 `tests/test_models.py`**

```python
from parser.models import LabelData, NutritionItem
from rules.base import RiskLevel, RuleResult, RuleStatus


def test_label_data_defaults():
    label = LabelData()
    assert label.net_content == ""
    assert label.nutrition == []


def test_nutrition_item_fields():
    item = NutritionItem(name="能量", value=1800.0, unit="kJ")
    assert item.name == "能量"
    assert item.unit == "kJ"


def test_rule_result_fields():
    result = RuleResult("R001", "净含量", RuleStatus.FAIL, RiskLevel.RED, "GB 7718", "缺净含量")
    assert result.status.value == "fail"
    assert result.level == RiskLevel.RED
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'parser'`）

- [ ] **Step 3: 实现 `parser/models.py`**

```python
from dataclasses import dataclass, field


@dataclass
class NutritionItem:
    name: str
    value: float
    unit: str


@dataclass
class LabelData:
    raw_text: str = ""
    product_name: str = ""
    net_content: str = ""
    production_date: str = ""
    shelf_life: str = ""
    storage_condition: str = ""
    ingredients: list[str] = field(default_factory=list)
    allergen_prompt: str = ""
    nutrition: list[NutritionItem] = field(default_factory=list)
    nutrition_claims: list[str] = field(default_factory=list)
    sc_number: str = ""
    manufacturer: str = ""
```

- [ ] **Step 4: 实现 `rules/base.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from parser.models import LabelData


class RuleStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SUSPECT = "suspect"


class RiskLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class RuleResult:
    rule_id: str
    rule_name: str
    status: RuleStatus
    level: RiskLevel
    basis: str
    message: str


class Rule(ABC):
    rule_id: str = ""
    rule_name: str = ""

    def __init__(self) -> None:
        if not self.rule_id:
            raise ValueError("Rule subclass must define rule_id")

    @abstractmethod
    def check(self, label: LabelData) -> RuleResult:
        ...
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_models.py -v`
Expected: 3 passed

- [ ] **Step 6: 提交**

```bash
git add parser rules tests
git commit -m "feat: label and rule result data models"
```

---

### Task 2: OCR 服务

**Files:**
- Create: `ocr/__init__.py`（空文件）
- Create: `ocr/ocr_service.py`
- Test: `tests/test_ocr.py`

- [ ] **Step 1: 写失败测试 `tests/test_ocr.py`**

```python
from ocr.ocr_service import extract_text


def test_extract_text_joins_lines(monkeypatch):
    class FakeOCR:
        def ocr(self, path, cls=True):
            return [[(None, ("净含量：500g", 0.98)), (None, ("保质期：12个月", 0.97))]]

    monkeypatch.setattr("ocr.ocr_service._get_ocr", lambda: FakeOCR())
    text = extract_text("dummy.png")
    assert "净含量：500g" in text
    assert "保质期：12个月" in text
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_ocr.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'ocr'`）

- [ ] **Step 3: 实现 `ocr/ocr_service.py`**

```python
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)


def extract_text(image_path: str) -> str:
    ocr = _get_ocr()
    result = ocr.ocr(image_path, cls=True)
    lines = []
    for page in result:
        if not page:
            continue
        for line in page:
            lines.append(line[1][0])
    return "\n".join(lines)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_ocr.py -v`
Expected: 1 passed

- [ ] **Step 5: 真机冒烟验证（可选但有价值）**

Run: `python -c "from ocr.ocr_service import extract_text; print(extract_text('data/test_images/T001.jpg'))"`
Expected: 打印出图片中的标签文字。首次运行会自动下载 PaddleOCR 模型（需要网络）。

- [ ] **Step 6: 提交**

```bash
git add ocr tests
git commit -m "feat: paddleocr text extraction service"
```

---

### Task 3: 结构化解析器

**Files:**
- Create: `parser/label_parser.py`
- Test: `tests/test_parser.py`

- [ ] **Step 1: 写失败测试 `tests/test_parser.py`**

```python
from parser.label_parser import parse

SAMPLE = """品名：奶油曲奇饼干
净含量：300g
配料：小麦粉、白砂糖、食用植物油、鸡蛋、乳粉
致敏物质：含有小麦、鸡蛋、乳及乳制品
生产日期：2026-08-01
保质期：12个月
贮存条件：请置于阴凉干燥处
生产商：示例食品有限公司
SC12345678901234
营养成分表
能量 1980 kJ
蛋白质 6.5 g
脂肪 22.0 g
碳水化合物 60.0 g
钠 380 mg
低糖
"""


def test_parse_full_label():
    label = parse(SAMPLE)
    assert label.product_name == "奶油曲奇饼干"
    assert label.net_content == "300g"
    assert label.production_date == "2026-08-01"
    assert label.shelf_life == "12个月"
    assert label.storage_condition == "请置于阴凉干燥处"
    assert label.ingredients[0] == "小麦粉"
    assert "鸡蛋" in label.ingredients
    assert "乳及乳制品" in label.allergen_prompt
    assert label.sc_number == "SC12345678901234"
    assert label.nutrition[0].name == "能量"
    assert label.nutrition[4].unit == "mg"
    assert "低糖" in label.nutrition_claims
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_parser.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `parser/label_parser.py`**

```python
import re

from parser.models import LabelData, NutritionItem

_NET_CONTENT = re.compile(r"净含量[:：]?\s*(\d+(?:\.\d+)?\s*(?:克|千克|g|kg|毫升|升|mL|ml|L))", re.IGNORECASE)
_PRODUCT_NAME = re.compile(r"(?:品名|产品名称)[:：]?\s*([^\n]+)")
_PRODUCTION_DATE = re.compile(r"生产日期[:：]?\s*(\d{4}[-年.]\d{1,2}[-月.]\d{1,2}日?)")
_SHELF_LIFE = re.compile(r"保质期[:：]?\s*(\d+(?:\.\d+)?\s*(?:天|日|个月|月|年))")
_SHELF_LIFE_UNTIL = re.compile(r"保质期至[:：]?\s*(\d{4}[-年.]\d{1,2}[-月.]\d{1,2}日?)")
_STORAGE = re.compile(r"贮存条件[:：]?\s*([^\n]+)")
_INGREDIENTS = re.compile(r"配料[:：]?\s*([^\n]+)")
_ALLERGEN = re.compile(r"(?:致敏物质|过敏原|过敏源)[:：]?\s*([^\n]+)")
_SC = re.compile(r"(SC\d{14})")
_MANUFACTURER = re.compile(r"(?:生产商|制造商|生产企业|生产厂家)[:：]?\s*([^\n]+)")
_CLAIM = re.compile(r"(低脂|低糖|低钠|无糖|无脂|高钙|高蛋白|富含|0糖)")
_NUTRITION = re.compile(r"营养成分表")

_NUTRITION_ITEMS = ("能量", "蛋白质", "脂肪", "碳水化合物", "钠")


def _parse_nutrition(text: str) -> list[NutritionItem]:
    if not _NUTRITION.search(text):
        return []
    items = []
    for name in _NUTRITION_ITEMS:
        m = re.search(rf"{name}\s*(\d+(?:\.\d+)?)\s*([a-zA-Zµ%]+)", text)
        if m:
            items.append(NutritionItem(name=name, value=float(m.group(1)), unit=m.group(2)))
    return items


def parse(text: str) -> LabelData:
    label = LabelData(raw_text=text)
    m = _NET_CONTENT.search(text)
    if m:
        label.net_content = m.group(1).replace(" ", "")
    m = _PRODUCT_NAME.search(text)
    if m:
        label.product_name = m.group(1).strip()
    m = _PRODUCTION_DATE.search(text)
    if m:
        label.production_date = m.group(1)
    m = _SHELF_LIFE_UNTIL.search(text) or _SHELF_LIFE.search(text)
    if m:
        label.shelf_life = m.group(1)
    m = _STORAGE.search(text)
    if m:
        label.storage_condition = m.group(1).strip()
    m = _INGREDIENTS.search(text)
    if m:
        label.ingredients = [s.strip() for s in re.split(r"[、，,；;]", m.group(1)) if s.strip()]
    m = _ALLERGEN.search(text)
    if m:
        label.allergen_prompt = m.group(1).strip()
    m = _SC.search(text)
    if m:
        label.sc_number = m.group(1)
    m = _MANUFACTURER.search(text)
    if m:
        label.manufacturer = m.group(1).strip()
    label.nutrition = _parse_nutrition(text)
    label.nutrition_claims = _CLAIM.findall(text)
    return label
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_parser.py -v`
Expected: 1 passed

- [ ] **Step 5: 提交**

```bash
git add parser tests
git commit -m "feat: structured label parser"
```

---

### Task 4: 规则引擎框架

**Files:**
- Create: `rules/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: 写失败测试 `tests/test_engine.py`**

```python
from parser.models import LabelData
from rules.base import RiskLevel, Rule, RuleResult, RuleStatus
from rules.engine import RuleEngine


class FakeRule(Rule):
    rule_id = "R999"
    rule_name = "假规则"

    def check(self, label: LabelData) -> RuleResult:
        return RuleResult(self.rule_id, self.rule_name, RuleStatus.PASS, RiskLevel.GREEN, "测试", "ok")


def test_engine_runs_all_rules():
    engine = RuleEngine()
    engine.register(FakeRule())
    engine.register(FakeRule())
    results = engine.run(LabelData())
    assert len(results) == 2
    assert results[0].rule_id == "R999"
    assert results[0].status == RuleStatus.PASS
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_engine.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `rules/engine.py`**

```python
from parser.models import LabelData
from rules.base import Rule, RuleResult


class RuleEngine:
    def __init__(self) -> None:
        self._rules: list[Rule] = []

    def register(self, rule: Rule) -> None:
        self._rules.append(rule)

    def run(self, label: LabelData) -> list[RuleResult]:
        return [rule.check(label) for rule in self._rules]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_engine.py -v`
Expected: 1 passed

- [ ] **Step 5: 提交**

```bash
git add rules tests
git commit -m "feat: rule engine framework"
```

---

### Task 5: 规则 R001 净含量 + R002 日期标示

**Files:**
- Create: `rules/r1_net_content.py`
- Create: `rules/r2_date_labeling.py`
- Test: `tests/test_rule_net_content.py`
- Test: `tests/test_rule_date.py`

- [ ] **Step 1: 写失败测试 `tests/test_rule_net_content.py`**

```python
from parser.models import LabelData
from rules.base import RuleStatus
from rules.r1_net_content import NetContentRule


def _check(raw: str):
    return NetContentRule().check(LabelData(net_content=raw))


def test_valid_formats_pass():
    for raw in ("500g", "1L", "300克", "1.5kg", "250毫升"):
        assert _check(raw).status == RuleStatus.PASS


def test_missing_net_content_fails():
    assert _check("").status == RuleStatus.FAIL


def test_bad_format_fails():
    assert _check("500").status == RuleStatus.FAIL
    assert _check("净含量：500g").status == RuleStatus.FAIL
```

- [ ] **Step 2: 写失败测试 `tests/test_rule_date.py`**

```python
from parser.models import LabelData
from rules.base import RuleStatus
from rules.r2_date_labeling import DateLabelingRule


def test_valid_dates_pass():
    label = LabelData(production_date="2026-08-01", shelf_life="12个月")
    assert DateLabelingRule().check(label).status == RuleStatus.PASS
    label = LabelData(production_date="2026年8月1日", shelf_life="2027-08-01")
    assert DateLabelingRule().check(label).status == RuleStatus.PASS


def test_missing_dates_fail():
    assert DateLabelingRule().check(LabelData()).status == RuleStatus.FAIL
```

- [ ] **Step 3: 运行两个测试确认失败**

Run: `python -m pytest tests/test_rule_net_content.py tests/test_rule_date.py -v`
Expected: 全部 FAIL（`ModuleNotFoundError`）

- [ ] **Step 4: 实现 `rules/r1_net_content.py`**

```python
import re

from parser.models import LabelData
from rules.base import RiskLevel, Rule, RuleResult, RuleStatus

_UNIT = r"(?:克|千克|g|kg|毫升|升|mL|ml|L)"
_PATTERN = re.compile(rf"^\d+(?:\.\d+)?\s*{_UNIT}$", re.IGNORECASE)


class NetContentRule(Rule):
    rule_id = "R001"
    rule_name = "净含量标示格式"
    basis = "GB 7718-2025 4.1.5 净含量标示"

    def check(self, label: LabelData) -> RuleResult:
        raw = label.net_content.strip()
        if not raw:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.FAIL, RiskLevel.RED, self.basis, "未识别到净含量标示，请确认标签是否标注净含量")
        if not _PATTERN.match(raw):
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.FAIL, RiskLevel.RED, self.basis, f"净含量格式不规范：'{raw}'，应为数字+法定计量单位（克/千克/毫升/升），如 500g、1L")
        return RuleResult(self.rule_id, self.rule_name, RuleStatus.PASS, RiskLevel.GREEN, self.basis, "净含量格式合规")
```

- [ ] **Step 5: 实现 `rules/r2_date_labeling.py`**

```python
import re

from parser.models import LabelData
from rules.base import RiskLevel, Rule, RuleResult, RuleStatus

_DATE = re.compile(r"^\d{4}[-年.]\d{1,2}[-月.]\d{1,2}日?$")
_SHELF = re.compile(r"^\d+(?:\.\d+)?\s*(?:天|日|个月|月|年)$|^\d{4}[-年.]\d{1,2}[-月.]\d{1,2}日?$")


class DateLabelingRule(Rule):
    rule_id = "R002"
    rule_name = "生产日期与保质期标示"
    basis = "GB 7718-2025 4.1.7 日期标示"

    def check(self, label: LabelData) -> RuleResult:
        problems = []
        if not label.production_date:
            problems.append("未识别到生产日期")
        elif not _DATE.match(label.production_date):
            problems.append(f"生产日期格式不规范：{label.production_date}")
        if not label.shelf_life:
            problems.append("未识别到保质期")
        elif not _SHELF.match(label.shelf_life):
            problems.append(f"保质期标示不规范：{label.shelf_life}，应为'保质期 X 天/个月/年'或'保质期至 日期'")
        if problems:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.FAIL, RiskLevel.RED, self.basis, "；".join(problems))
        return RuleResult(self.rule_id, self.rule_name, RuleStatus.PASS, RiskLevel.GREEN, self.basis, "生产日期与保质期标示完整")
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_rule_net_content.py tests/test_rule_date.py -v`
Expected: 全部 PASS

- [ ] **Step 7: 提交**

```bash
git add rules tests
git commit -m "feat: net content and date labeling rules"
```

---

### Task 6: 规则 R003 配料表完整性 + R004 致敏物质

**Files:**
- Create: `rules/r3_ingredient_list.py`
- Create: `rules/r4_allergen.py`
- Test: `tests/test_rule_ingredient_list.py`
- Test: `tests/test_rule_allergen.py`

- [ ] **Step 1: 写失败测试 `tests/test_rule_ingredient_list.py`**

```python
from parser.models import LabelData
from rules.base import RuleStatus
from rules.r3_ingredient_list import IngredientListRule


def test_missing_ingredients_fail():
    result = IngredientListRule().check(LabelData())
    assert result.status == RuleStatus.FAIL


def test_bare_additive_is_suspect():
    label = LabelData(ingredients=["小麦粉", "山梨酸钾"])
    result = IngredientListRule().check(label)
    assert result.status == RuleStatus.SUSPECT


def test_proper_additive_passes():
    label = LabelData(ingredients=["小麦粉", "防腐剂（山梨酸钾）"])
    result = IngredientListRule().check(label)
    assert result.status == RuleStatus.PASS
```

- [ ] **Step 2: 写失败测试 `tests/test_rule_allergen.py`**

```python
from parser.models import LabelData
from rules.base import RuleStatus
from rules.r4_allergen import AllergenRule


def test_allergen_without_prompt_fails():
    label = LabelData(ingredients=["小麦粉", "鸡蛋"])
    result = AllergenRule().check(label)
    assert result.status == RuleStatus.FAIL


def test_allergen_with_prompt_passes():
    label = LabelData(ingredients=["小麦粉", "鸡蛋"], allergen_prompt="含有小麦、鸡蛋")
    assert AllergenRule().check(label).status == RuleStatus.PASS


def test_no_allergen_passes():
    label = LabelData(ingredients=["白砂糖", "食用盐"])
    assert AllergenRule().check(label).status == RuleStatus.PASS
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_rule_ingredient_list.py tests/test_rule_allergen.py -v`
Expected: 全部 FAIL（`ModuleNotFoundError`）

- [ ] **Step 4: 实现 `rules/r3_ingredient_list.py`**

```python
import re

from parser.models import LabelData
from rules.base import RiskLevel, Rule, RuleResult, RuleStatus

_ADDITIVE_WITH_CATEGORY = re.compile(r"[\u4e00-\u9fa5]{2,4}[（(][\u4e00-\u9fa5A-Za-z0-9]+[）)]")
_COMMON_ADDITIVES = re.compile(r"(山梨酸钾|苯甲酸钠|安赛蜜|阿斯巴甜|柠檬酸|黄原胶|瓜尔胶)")


class IngredientListRule(Rule):
    rule_id = "R003"
    rule_name = "配料表标示完整性"
    basis = "GB 7718-2025 4.1.3 配料表"

    def check(self, label: LabelData) -> RuleResult:
        if not label.ingredients:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.FAIL, RiskLevel.RED, self.basis, "未识别到配料表，预包装食品应标示配料表")
        bare = [i for i in label.ingredients if _COMMON_ADDITIVES.search(i) and not _ADDITIVE_WITH_CATEGORY.search(i)]
        if bare:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.SUSPECT, RiskLevel.YELLOW, self.basis, f"食品添加剂未按'类别（名称）'标示：{bare}，如'防腐剂（山梨酸钾）'")
        return RuleResult(self.rule_id, self.rule_name, RuleStatus.PASS, RiskLevel.GREEN, self.basis, "配料表标示完整")
```

- [ ] **Step 5: 实现 `rules/r4_allergen.py`**

```python
from parser.models import LabelData
from rules.base import RiskLevel, Rule, RuleResult, RuleStatus

# 清单以 GB 7718-2025 正式文本为准（落地前核对，新版可能扩充至芝麻、芹菜等）
ALLERGEN_KEYWORDS = [
    "小麦", "大麦", "黑麦", "燕麦", "麸质", "虾", "蟹", "甲壳", "鸡蛋", "蛋清", "蛋黄",
    "蛋粉", "花生", "大豆", "黄豆", "豆粕", "乳", "牛奶", "奶粉", "奶油", "黄油", "乳清",
    "杏仁", "核桃", "腰果", "榛子", "开心果", "夏威夷果", "碧根果", "松子", "芝麻", "芹菜",
]


class AllergenRule(Rule):
    rule_id = "R004"
    rule_name = "致敏物质标示"
    basis = "GB 7718-2025 4.1.10 致敏物质标示"

    def check(self, label: LabelData) -> RuleResult:
        if not label.ingredients:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.SUSPECT, RiskLevel.YELLOW, self.basis, "未识别到配料表，无法核对致敏物质")
        hits = [kw for kw in ALLERGEN_KEYWORDS if any(kw in ing for ing in label.ingredients)]
        if hits and not label.allergen_prompt:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.FAIL, RiskLevel.RED, self.basis, f"配料含致敏物质（{'、'.join(sorted(set(hits)))}）但未标示致敏物质提示")
        if hits and label.allergen_prompt:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.PASS, RiskLevel.GREEN, self.basis, "致敏物质已提示")
        return RuleResult(self.rule_id, self.rule_name, RuleStatus.PASS, RiskLevel.GREEN, self.basis, "未发现常见致敏物质")
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_rule_ingredient_list.py tests/test_rule_allergen.py -v`
Expected: 全部 PASS

- [ ] **Step 7: 提交**

```bash
git add rules tests
git commit -m "feat: ingredient list and allergen rules"
```

---

### Task 7: 规则 R005 营养成分表 + R006 营养声称

**Files:**
- Create: `rules/r5_nutrition_table.py`
- Create: `rules/r6_nutrition_claim.py`
- Test: `tests/test_rule_nutrition_table.py`
- Test: `tests/test_rule_nutrition_claim.py`

- [ ] **Step 1: 写失败测试 `tests/test_rule_nutrition_table.py`**

```python
from parser.models import LabelData, NutritionItem
from rules.base import RuleStatus
from rules.r5_nutrition_table import NutritionTableRule


def _full_nutrition():
    return [
        NutritionItem("能量", 1980, "kJ"), NutritionItem("蛋白质", 6.5, "g"),
        NutritionItem("脂肪", 22.0, "g"), NutritionItem("碳水化合物", 60.0, "g"),
        NutritionItem("钠", 380, "mg"),
    ]


def test_complete_table_passes():
    result = NutritionTableRule().check(LabelData(nutrition=_full_nutrition()))
    assert result.status == RuleStatus.PASS


def test_missing_item_fails():
    label = LabelData(nutrition=_full_nutrition()[:-1])
    assert NutritionTableRule().check(label).status == RuleStatus.FAIL


def test_wrong_unit_fails():
    nutrition = _full_nutrition()
    nutrition[0] = NutritionItem("能量", 1980, "kcal")
    result = NutritionTableRule().check(LabelData(nutrition=nutrition))
    assert result.status == RuleStatus.FAIL
```

- [ ] **Step 2: 写失败测试 `tests/test_rule_nutrition_claim.py`**

```python
from parser.models import LabelData, NutritionItem
from rules.base import RuleStatus
from rules.r6_nutrition_claim import NutritionClaimRule


def test_low_sugar_within_threshold_passes():
    label = LabelData(nutrition_claims=["低糖"], nutrition=[NutritionItem("碳水化合物", 3.0, "g")])
    assert NutritionClaimRule().check(label).status == RuleStatus.PASS


def test_low_sugar_exceeding_threshold_fails():
    label = LabelData(nutrition_claims=["低糖"], nutrition=[NutritionItem("碳水化合物", 8.0, "g")])
    assert NutritionClaimRule().check(label).status == RuleStatus.FAIL


def test_claim_without_data_is_suspect():
    label = LabelData(nutrition_claims=["高钙"], nutrition=[])
    result = NutritionClaimRule().check(label)
    assert result.status == RuleStatus.SUSPECT
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_rule_nutrition_table.py tests/test_rule_nutrition_claim.py -v`
Expected: 全部 FAIL（`ModuleNotFoundError`）

- [ ] **Step 4: 实现 `rules/r5_nutrition_table.py`**

```python
from parser.models import LabelData
from rules.base import RiskLevel, Rule, RuleResult, RuleStatus

REQUIRED = {"能量": "kJ", "蛋白质": "g", "脂肪": "g", "碳水化合物": "g", "钠": "mg"}


class NutritionTableRule(Rule):
    rule_id = "R005"
    rule_name = "营养成分表五项强制标示"
    basis = "GB 28050-2025 强制标示内容"

    def check(self, label: LabelData) -> RuleResult:
        by_name = {item.name: item for item in label.nutrition}
        missing = [name for name in REQUIRED if name not in by_name]
        if missing:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.FAIL, RiskLevel.RED, self.basis, f"营养成分表缺少强制标示项：{missing}（应含能量、蛋白质、脂肪、碳水化合物、钠）")
        wrong_unit = [f"{name}({by_name[name].unit})" for name in REQUIRED if by_name[name].unit != REQUIRED[name]]
        if wrong_unit:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.FAIL, RiskLevel.RED, self.basis, f"营养成分表单位不规范：{wrong_unit}，应为能量 kJ、蛋白质/脂肪/碳水化合物 g、钠 mg")
        return RuleResult(self.rule_id, self.rule_name, RuleStatus.PASS, RiskLevel.GREEN, self.basis, "营养成分表五项标示齐全且单位规范")
```

- [ ] **Step 5: 实现 `rules/r6_nutrition_claim.py`**

```python
from parser.models import LabelData
from rules.base import RiskLevel, Rule, RuleResult, RuleStatus

# 阈值与 NRV 基数以 GB 28050-2025 附录 C 正式文本为准（落地前核对）；MVP 按固体阈值初判
CLAIM_THRESHOLDS = {
    "低脂": ("脂肪", "g", 3.0, "≤3 g/100g（固体）"),
    "低糖": ("碳水化合物", "g", 5.0, "≤5 g/100g 或 100mL"),
    "无糖": ("碳水化合物", "g", 0.5, "≤0.5 g/100g 或 100mL"),
    "低钠": ("钠", "mg", 120.0, "≤120 mg/100g 或 100mL"),
    "高钙": ("钙", "mg", 240.0, "≥240 mg/100g 或 100mL（≥30% NRV）"),
}


class NutritionClaimRule(Rule):
    rule_id = "R006"
    rule_name = "营养声称条件符合性"
    basis = "GB 28050-2025 表 C.1"

    def check(self, label: LabelData) -> RuleResult:
        active = [c for c in label.nutrition_claims if c in CLAIM_THRESHOLDS]
        if not active:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.PASS, RiskLevel.GREEN, self.basis, "未发现需核对的营养声称")
        problems = []
        for claim in active:
            item_name, unit, threshold, desc = CLAIM_THRESHOLDS[claim]
            item = next((n for n in label.nutrition if n.name == item_name), None)
            if item is None:
                problems.append(f"声称'{claim}'但营养成分表中缺少'{item_name}'，无法核实")
                continue
            ok = item.value >= threshold if claim == "高钙" else item.value <= threshold
            if not ok:
                problems.append(f"声称'{claim}'不成立：{item_name} 为 {item.value}{item.unit}，要求 {desc}")
        if problems:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.FAIL, RiskLevel.RED, self.basis, "；".join(problems))
        return RuleResult(self.rule_id, self.rule_name, RuleStatus.PASS, RiskLevel.GREEN, self.basis, "营养声称与营养成分表数据一致")
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_rule_nutrition_table.py tests/test_rule_nutrition_claim.py -v`
Expected: 全部 PASS

- [ ] **Step 7: 提交**

```bash
git add rules tests
git commit -m "feat: nutrition table and claim rules"
```

---

### Task 8: 规则 R007 SC 编号 + R008 贮存条件 + R009 敏感词

**Files:**
- Create: `rules/r7_sc_number.py`
- Create: `rules/r8_storage_condition.py`
- Create: `rules/r9_sensitive_word.py`
- Test: `tests/test_rule_sc_number.py`
- Test: `tests/test_rule_storage.py`
- Test: `tests/test_rule_sensitive.py`

- [ ] **Step 1: 写失败测试（三个测试文件合并在此展示）**

`tests/test_rule_sc_number.py`:

```python
from parser.models import LabelData
from rules.base import RuleStatus
from rules.r7_sc_number import ScNumberRule


def test_valid_sc_passes():
    assert ScNumberRule().check(LabelData(sc_number="SC12345678901234")).status == RuleStatus.PASS


def test_missing_sc_fails():
    assert ScNumberRule().check(LabelData()).status == RuleStatus.FAIL


def test_bad_sc_fails():
    assert ScNumberRule().check(LabelData(sc_number="SC123")).status == RuleStatus.FAIL
```

`tests/test_rule_storage.py`:

```python
from parser.models import LabelData
from rules.base import RuleStatus
from rules.r8_storage_condition import StorageConditionRule


def test_no_storage_is_suspect():
    assert StorageConditionRule().check(LabelData()).status == RuleStatus.SUSPECT


def test_conflicting_storage_is_suspect():
    label = LabelData(storage_condition="常温保存，开封后请冷藏")
    assert StorageConditionRule().check(label).status == RuleStatus.SUSPECT


def test_normal_storage_passes():
    label = LabelData(storage_condition="请置于阴凉干燥处")
    assert StorageConditionRule().check(label).status == RuleStatus.PASS
```

`tests/test_rule_sensitive.py`:

```python
from parser.models import LabelData
from rules.base import RuleStatus
from rules.r9_sensitive_word import SensitiveWordRule


def test_sensitive_word_is_suspect():
    label = LabelData(raw_text="本品为顶级配方，纯天然无副作用")
    assert SensitiveWordRule().check(label).status == RuleStatus.SUSPECT


def test_clean_text_passes():
    label = LabelData(raw_text="品名：曲奇饼干，净含量：300g")
    assert SensitiveWordRule().check(label).status == RuleStatus.PASS
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_rule_sc_number.py tests/test_rule_storage.py tests/test_rule_sensitive.py -v`
Expected: 全部 FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `rules/r7_sc_number.py`**

```python
import re

from parser.models import LabelData
from rules.base import RiskLevel, Rule, RuleResult, RuleStatus

_PATTERN = re.compile(r"^SC\d{14}$")


class ScNumberRule(Rule):
    rule_id = "R007"
    rule_name = "生产许可证编号标示"
    basis = "GB 7718-2025 4.1.9 生产者信息"

    def check(self, label: LabelData) -> RuleResult:
        if not label.sc_number:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.FAIL, RiskLevel.RED, self.basis, "未识别到生产许可证编号（SC + 14 位数字）")
        if not _PATTERN.match(label.sc_number):
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.FAIL, RiskLevel.RED, self.basis, f"生产许可证编号格式不规范：{label.sc_number}，应为 SC 后接 14 位数字")
        return RuleResult(self.rule_id, self.rule_name, RuleStatus.PASS, RiskLevel.GREEN, self.basis, "生产许可证编号格式合规")
```

- [ ] **Step 4: 实现 `rules/r8_storage_condition.py`**

```python
from parser.models import LabelData
from rules.base import RiskLevel, Rule, RuleResult, RuleStatus

_COLD_WORDS = ("冷藏", "冷冻", "0-4", "0~4", "零下", "-18")
_ROOM_WORDS = ("常温", "阴凉")


class StorageConditionRule(Rule):
    rule_id = "R008"
    rule_name = "贮存条件表述初筛"
    basis = "GB 7718-2025 4.1.8 贮存条件"

    def check(self, label: LabelData) -> RuleResult:
        if not label.storage_condition:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.SUSPECT, RiskLevel.YELLOW, self.basis, "未识别到贮存条件，预包装食品应标示贮存条件")
        cold = any(w in label.storage_condition for w in _COLD_WORDS)
        room = any(w in label.storage_condition for w in _ROOM_WORDS)
        if cold and room:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.SUSPECT, RiskLevel.YELLOW, self.basis, f"贮存条件表述疑似矛盾：{label.storage_condition}")
        return RuleResult(self.rule_id, self.rule_name, RuleStatus.PASS, RiskLevel.GREEN, self.basis, "贮存条件表述无冲突")
```

- [ ] **Step 5: 实现 `rules/r9_sensitive_word.py`**

```python
import re

from parser.models import LabelData
from rules.base import RiskLevel, Rule, RuleResult, RuleStatus

_SENSITIVE = re.compile(r"(最|第一|顶级|最佳|最好|国家级|纯天然|无副作用|治疗|疗效|治愈|抗癌|抗肿瘤|减肥|降糖|降血压|预防疾病)")


class SensitiveWordRule(Rule):
    rule_id = "R009"
    rule_name = "广告法敏感词初筛"
    basis = "广告法第九条 / 食品安全法第七十三条"

    def check(self, label: LabelData) -> RuleResult:
        hits = _SENSITIVE.findall(label.raw_text)
        if hits:
            return RuleResult(self.rule_id, self.rule_name, RuleStatus.SUSPECT, RiskLevel.YELLOW, self.basis, f"检测到广告法敏感词：{sorted(set(hits))}，请结合语境复核")
        return RuleResult(self.rule_id, self.rule_name, RuleStatus.PASS, RiskLevel.GREEN, self.basis, "未检测到广告法敏感词")
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_rule_sc_number.py tests/test_rule_storage.py tests/test_rule_sensitive.py -v`
Expected: 全部 PASS

- [ ] **Step 7: 提交**

```bash
git add rules tests
git commit -m "feat: sc number, storage, sensitive word rules"
```

---

### Task 9: LLM 语义审查模块

**Files:**
- Create: `llm/__init__.py`（空文件）
- Create: `llm/prompts.py`
- Create: `llm/semantic_review.py`
- Test: `tests/test_semantic_review.py`

- [ ] **Step 1: 写失败测试 `tests/test_semantic_review.py`**

```python
import json

from parser.models import LabelData
from rules.base import RuleStatus
from llm.semantic_review import semantic_review


def test_semantic_review_parses_llm_output(monkeypatch):
    fake_response = json.dumps([
        {"rule_id": "R010", "rule_name": "语义复核", "status": "fail", "level": "red",
         "basis": "GB 28050-2025", "message": "配料含白砂糖但碳水化合物标示为 0，疑似矛盾"},
    ])

    class FakeResp:
        def raise_for_status(self):
            pass

        @property
        def json(self):
            return {"response": fake_response}

    def fake_post(url, json, timeout):
        return FakeResp()

    monkeypatch.setattr("llm.semantic_review.httpx.post", fake_post)
    monkeypatch.setattr("llm.semantic_review.retrieve", lambda q, top_k: ["相关条款文本"])

    results = semantic_review(LabelData(ingredients=["白砂糖"], nutrition=[]))
    assert len(results) == 1
    assert results[0].rule_id == "R010"
    assert results[0].status == RuleStatus.FAIL
    assert "矛盾" in results[0].message
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_semantic_review.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `llm/prompts.py`**

```python
SYSTEM_PROMPT = """你是食品标签合规审查助手，精通 GB 7718-2025《预包装食品标签通则》与 GB 28050-2025《预包装食品营养标签通则》。
你将收到标签的结构化字段与相关国标条款。请只做语义层面的补充审查：
1. 配料表与营养成分表是否自相矛盾（如配料含白砂糖但碳水化合物标示为 0）
2. 贮存条件与保质期是否合理匹配（如"常温保存"却标注保质期 3 年）
3. 广告法敏感词是否构成违规宣传（结合语境判断）
4. 其他你发现的确凿合规问题
只输出一个 JSON 数组，每项为：
{"rule_id":"R010","rule_name":"语义复核","status":"pass|fail|suspect","level":"green|yellow|red","basis":"引用的条款原文","message":"具体问题说明（无问题则写"未发现语义问题"）"}
不要输出 JSON 以外的任何内容。"""

USER_PROMPT_TEMPLATE = """标签结构化字段：
{label_json}

相关国标条款（检索结果）：
{clauses}
"""
```

- [ ] **Step 4: 实现 `llm/semantic_review.py`**

```python
import json
import re

import httpx

from llm.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from llm.rag import retrieve
from parser.models import LabelData
from rules.base import RiskLevel, RuleResult, RuleStatus

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"


def _extract_json(text: str) -> str:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"模型输出中未找到 JSON：{text[:200]}")
    return match.group(0)


def semantic_review(label: LabelData, top_k: int = 3) -> list[RuleResult]:
    query = " ".join(label.ingredients) + " " + label.shelf_life + " " + label.storage_condition
    clauses = retrieve(query, top_k=top_k)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        label_json=json.dumps(label.__dict__, ensure_ascii=False, default=str),
        clauses="\n".join(clauses),
    )
    resp = httpx.post(
        OLLAMA_URL,
        json={"model": MODEL, "system": SYSTEM_PROMPT, "prompt": user_prompt, "stream": False, "temperature": 0.2},
        timeout=180,
    )
    resp.raise_for_status()
    items = json.loads(_extract_json(resp.json()["response"]))
    results = []
    for item in items:
        results.append(RuleResult(
            rule_id=item.get("rule_id", "R010"),
            rule_name=item.get("rule_name", "语义复核"),
            status=RuleStatus(item.get("status", "suspect")),
            level=RiskLevel(item.get("level", "yellow")),
            basis=item.get("basis", ""),
            message=item.get("message", ""),
        ))
    return results
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_semantic_review.py -v`
Expected: 1 passed

- [ ] **Step 6: 提交**

```bash
git add llm tests
git commit -m "feat: llm semantic review with strict json output"
```

---

### Task 10: RAG 条款检索

**Files:**
- Create: `llm/rag.py`
- Create: `scripts/build_index.py`
- Test: `tests/test_rag.py`

- [ ] **Step 1: 写失败测试 `tests/test_rag.py`**

```python
import numpy as np

from llm.rag import build_index, retrieve


def test_build_and_retrieve(tmp_path, monkeypatch):
    std_dir = tmp_path / "standards"
    std_dir.mkdir()
    (std_dir / "gb7718.txt").write_text("净含量应当以法定计量单位标示。\n\n配料表应当按照加入量递减顺序排列。", encoding="utf-8")

    monkeypatch.setattr("llm.rag._embed_fn", lambda: lambda texts: [np.zeros(8, dtype=float) for _ in texts])

    db_dir = tmp_path / "db"
    build_index(std_dir, db_dir)
    docs = retrieve("净含量", top_k=1, db_dir=db_dir)
    assert len(docs) == 1
    assert "净含量" in docs[0]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_rag.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `llm/rag.py`**

```python
import re
from pathlib import Path

import chromadb

_PARA_SPLIT = re.compile(r"\n\s*\n")


def chunk_standard(text: str) -> list[str]:
    parts = _PARA_SPLIT.split(text)
    return [p.strip() for p in parts if len(p.strip()) >= 10]


def _embed_fn():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3")
    return model.encode


def build_index(standards_dir: str | Path, db_dir: str | Path = "data/vector_db") -> None:
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection("gb_standards")
    chunks, ids, metas = [], [], []
    for path in sorted(Path(standards_dir).glob("*.txt")):
        for i, chunk in enumerate(chunk_standard(path.read_text(encoding="utf-8"))):
            chunks.append(chunk)
            ids.append(f"{path.stem}_{i}")
            metas.append({"source": path.stem})
    embed = _embed_fn()
    collection.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embed(chunks).tolist())


def retrieve(query: str, top_k: int = 3, db_dir: str | Path = "data/vector_db") -> list[str]:
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection("gb_standards")
    embed = _embed_fn()
    result = collection.query(query_embeddings=[embed([query]).tolist()[0]], n_results=top_k)
    return result["documents"][0]
```

- [ ] **Step 4: 实现 `scripts/build_index.py`**

```python
from pathlib import Path

from llm.rag import build_index

if __name__ == "__main__":
    project = Path(__file__).resolve().parent.parent
    build_index(project / "data" / "standards", project / "data" / "vector_db")
    print("向量库构建完成")
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_rag.py -v`
Expected: 1 passed

- [ ] **Step 6: 真机建库**

把 GB 7718-2025、GB 28050-2025、GB 2760-2024 全文保存为 `data/standards/` 下的 txt 文件，然后：

```powershell
python scripts/build_index.py
```

Expected: 输出"向量库构建完成"。首次运行会下载 bge-m3 模型（需要网络，几十秒到几分钟）。

- [ ] **Step 7: 提交**

```bash
git add llm scripts tests
git commit -m "feat: rag retrieval over gb standards"
```

---

### Task 11: 报告生成

**Files:**
- Create: `report/__init__.py`（空文件）
- Create: `report/report_generator.py`
- Create: `report/template.html`
- Test: `tests/test_report.py`

- [ ] **Step 1: 写失败测试 `tests/test_report.py`**

```python
from parser.models import LabelData
from rules.base import RiskLevel, RuleResult, RuleStatus
from report.report_generator import render_report


def test_render_report_contains_results_and_disclaimer():
    label = LabelData(net_content="500")
    results = [RuleResult("R001", "净含量", RuleStatus.FAIL, RiskLevel.RED, "GB 7718-2025", "净含量格式不规范")]
    html = render_report(label, results)
    assert "净含量格式不规范" in html
    assert "仅供参考" in html
    assert "R001" in html
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `report/report_generator.py`**

```python
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from parser.models import LabelData
from rules.base import RuleResult

_TEMPLATE_DIR = Path(__file__).resolve().parent


def render_report(label: LabelData, results: list[RuleResult]) -> str:
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))
    template = env.get_template("template.html")
    fail_count = sum(1 for r in results if r.status.value == "fail")
    suspect_count = sum(1 for r in results if r.status.value == "suspect")
    return template.render(
        label=label,
        results=results,
        fail_count=fail_count,
        suspect_count=suspect_count,
        total=len(results),
    )
```

- [ ] **Step 4: 实现 `report/template.html`**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>食品标签合规审查报告</title>
<style>
  body { font-family: "Microsoft YaHei", sans-serif; max-width: 800px; margin: 24px auto; padding: 0 16px; color: #222; }
  h1 { font-size: 22px; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 10px; color: #fff; font-size: 12px; margin-right: 6px; }
  .red { background: #d33; }
  .yellow { background: #e8a100; }
  .green { background: #2a8c3e; }
  table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px; }
  th, td { border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }
  th { background: #f2f2f2; }
  .disclaimer { margin-top: 24px; padding: 12px; background: #fff7e6; border-left: 4px solid #e8a100; font-size: 13px; }
</style>
</head>
<body>
<h1>🏷️ 食品标签合规审查报告</h1>
<p><strong>审查项：</strong>{{ total }} 项，其中 <span class="badge red">违规 {{ fail_count }}</span>
<span class="badge yellow">存疑 {{ suspect_count }}</span>
<span class="badge green">通过 {{ total - fail_count - suspect_count }}</span></p>
<p><strong>净含量：</strong>{{ label.net_content or "—" }} ｜
<strong>配料：</strong>{{ label.ingredients | join("、") or "—" }}</p>
<table>
  <tr><th>规则</th><th>结论</th><th>风险</th><th>说明</th><th>依据</th></tr>
  {% for r in results %}
  <tr>
    <td>{{ r.rule_id }} {{ r.rule_name }}</td>
    <td>{{ r.status.value }}</td>
    <td><span class="badge {{ r.level.value }}">{{ r.level.value }}</span></td>
    <td>{{ r.message }}</td>
    <td>{{ r.basis }}</td>
  </tr>
  {% endfor %}
</table>
<div class="disclaimer">⚠️ 本报告由 AI 辅助生成，仅供参考，正式合规审查需由专业人员依据官方标准文本复核。</div>
</body>
</html>
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_report.py -v`
Expected: 1 passed

- [ ] **Step 6: 提交**

```bash
git add report tests
git commit -m "feat: html report generation"
```

---

### Task 12: FastAPI 接口

**Files:**
- Create: `app/__init__.py`（空文件）
- Create: `app/api.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: 写失败测试 `tests/test_api.py`**

```python
from fastapi.testclient import TestClient

from app import api


def test_review_endpoint(monkeypatch):
    monkeypatch.setattr("app.api.extract_text", lambda p: "净含量：500g\n生产日期：2026-08-01\n保质期：12个月\n配料：小麦粉、白砂糖\nSC12345678901234")
    monkeypatch.setattr("app.api.semantic_review", lambda label: [])

    client = TestClient(api.app)
    resp = client.post("/api/review", files={"file": ("label.png", b"fake", "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"]["net_content"] == "500g"
    assert body["results"]
    assert body["report_html"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `app/api.py`**

```python
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

from llm.semantic_review import semantic_review
from ocr.ocr_service import extract_text
from parser.label_parser import parse
from report.report_generator import render_report
from rules.engine import RuleEngine
from rules.r1_net_content import NetContentRule
from rules.r2_date_labeling import DateLabelingRule
from rules.r3_ingredient_list import IngredientListRule
from rules.r4_allergen import AllergenRule
from rules.r5_nutrition_table import NutritionTableRule
from rules.r6_nutrition_claim import NutritionClaimRule
from rules.r7_sc_number import ScNumberRule
from rules.r8_storage_condition import StorageConditionRule
from rules.r9_sensitive_word import SensitiveWordRule

app = FastAPI(title="食品标签合规审查助手")

_engine = RuleEngine()
for rule in (
    NetContentRule(), DateLabelingRule(), IngredientListRule(), AllergenRule(),
    NutritionTableRule(), NutritionClaimRule(), ScNumberRule(),
    StorageConditionRule(), SensitiveWordRule(),
):
    _engine.register(rule)


@app.post("/api/review")
async def review(file: UploadFile = File(...)):
    suffix = Path(file.filename or "label.png").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = extract_text(tmp_path)
        label = parse(text)
        results = _engine.run(label) + semantic_review(label)
        return {
            "text": text,
            "label": asdict(label),
            "results": [asdict(r) for r in results],
            "report_html": render_report(label, results),
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_api.py -v`
Expected: 1 passed

- [ ] **Step 5: 提交**

```bash
git add app tests
git commit -m "feat: fastapi review endpoint"
```

---

### Task 13: Streamlit 前端

**Files:**
- Create: `app/ui.py`（UI 为手动验证，不写单元测试）

- [ ] **Step 1: 实现 `app/ui.py`**

```python
import requests
import streamlit as st

st.set_page_config(page_title="食品标签合规审查助手", page_icon="🏷️")
st.title("🏷️ 食品标签合规审查助手")
st.caption("上传预包装食品标签照片，自动审查 GB 7718 / GB 28050 合规性。结果仅供参考，需人工复核。")

uploaded = st.file_uploader("上传标签照片（支持 jpg/png）", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="已上传的标签", width=400)
    if st.button("开始审查"):
        with st.spinner("正在识别与审查，请稍候…"):
            resp = requests.post(
                "http://127.0.0.1:8000/api/review",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=180,
            )
            if resp.status_code == 200:
                body = resp.json()
                st.markdown(body["report_html"], unsafe_allow_html=True)
                st.download_button("下载审查报告（HTML）", body["report_html"], file_name="report.html", mime="text/html")
            else:
                st.error(f"审查失败：HTTP {resp.status_code}")
```

- [ ] **Step 2: 手动端到端验证**

```powershell
# 终端 1：启动后端
uvicorn app.api:app --reload --port 8000

# 终端 2：启动前端
streamlit run app/ui.py
```

Expected: 浏览器打开 Streamlit 页面，上传一张标签照片，点击"开始审查"后显示审查报告，可下载 HTML。

- [ ] **Step 3: 提交**

```bash
git add app
git commit -m "feat: streamlit ui"
```

---

### Task 14: 测试集与端到端评估

**Files:**
- Create: `data/test_labels.json`（先建 5 条种子数据，实拍后扩充到 20~30 条）
- Create: `scripts/evaluate.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: 建测试集标注文件 `data/test_labels.json`（先 5 条种子）**

```json
[
  {"id": "T001", "image": "T001.jpg", "expected": {"R001": "pass", "R002": "pass", "R004": "pass", "R005": "pass", "R007": "pass", "R009": "pass"}},
  {"id": "T002", "image": "T002.jpg", "expected": {"R001": "fail", "R004": "fail", "R005": "fail"}},
  {"id": "T003", "image": "T003.jpg", "expected": {"R001": "pass", "R006": "fail", "R009": "suspect"}},
  {"id": "T004", "image": "T004.jpg", "expected": {"R001": "pass", "R002": "fail", "R007": "fail"}},
  {"id": "T005", "image": "T005.jpg", "expected": {"R001": "pass", "R003": "pass", "R004": "fail"}}
]
```

标注规则：`expected` 中 `pass` 表示该规则应通过，`fail`/`suspect` 表示应非通过。

- [ ] **Step 2: 写冒烟测试 `tests/test_pipeline.py`**

```python
from parser.label_parser import parse
from rules.engine import RuleEngine
from rules.r1_net_content import NetContentRule
from rules.r9_sensitive_word import SensitiveWordRule

SAMPLE = "净含量：500g\n配料：小麦粉\nSC12345678901234\n本品为顶级配方"


def test_pipeline_without_llm():
    engine = RuleEngine()
    engine.register(NetContentRule())
    engine.register(SensitiveWordRule())
    label = parse(SAMPLE)
    results = engine.run(label)
    status = {r.rule_id: r.status.value for r in results}
    assert status["R001"] == "pass"
    assert status["R009"] == "suspect"
```

- [ ] **Step 3: 运行冒烟测试确认通过**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 1 passed

- [ ] **Step 4: 实现 `scripts/evaluate.py`**

```python
import json
import sys
from pathlib import Path

from llm.semantic_review import semantic_review
from ocr.ocr_service import extract_text
from parser.label_parser import parse
from rules.engine import RuleEngine
from rules.r1_net_content import NetContentRule
from rules.r2_date_labeling import DateLabelingRule
from rules.r3_ingredient_list import IngredientListRule
from rules.r4_allergen import AllergenRule
from rules.r5_nutrition_table import NutritionTableRule
from rules.r6_nutrition_claim import NutritionClaimRule
from rules.r7_sc_number import ScNumberRule
from rules.r8_storage_condition import StorageConditionRule
from rules.r9_sensitive_word import SensitiveWordRule

ENGINE = RuleEngine()
for rule in (
    NetContentRule(), DateLabelingRule(), IngredientListRule(), AllergenRule(),
    NutritionTableRule(), NutritionClaimRule(), ScNumberRule(),
    StorageConditionRule(), SensitiveWordRule(),
):
    ENGINE.register(rule)


def main(test_file: str, images_dir: str) -> None:
    data = json.loads(Path(test_file).read_text(encoding="utf-8"))
    stats: dict[str, dict[str, int]] = {}
    for case in data:
        text = extract_text(str(Path(images_dir) / case["image"]))
        label = parse(text)
        results = ENGINE.run(label) + semantic_review(label)
        got = {r.rule_id: r.status.value for r in results}
        for rule_id, want in case["expected"].items():
            stats.setdefault(rule_id, {"tp": 0, "fn": 0})
            got_val = got.get(rule_id, "missing")
            if (want == "pass") == (got_val == "pass"):
                stats[rule_id]["tp"] += 1
            else:
                stats[rule_id]["fn"] += 1
    print("规则准确率：")
    for rule_id, s in sorted(stats.items()):
        total = s["tp"] + s["fn"]
        print(f"  {rule_id}: {s['tp']}/{total} = {s['tp'] / max(total, 1):.0%}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 5: 实拍测试集并评估**

超市拍 20~30 张常见预包装食品标签（正反面），按 `data/test_labels.json` 格式标注，然后：

```powershell
python scripts/evaluate.py data/test_labels.json data/test_images
```

Expected: 输出每条规则的准确率。低于 90% 的规则，回 Task 5~8 调整对应规则（多为解析器漏字段，优先改 `parser/label_parser.py` 的正则）。

- [ ] **Step 6: 提交**

```bash
git add data scripts tests
git commit -m "feat: test set and evaluation script"
```

---

### Task 15: 交付物收尾

**Files:**
- Modify: `README.md`
- Create: `docs/demo-script.md`（演示视频脚本）
- Create: `docs/blog-outline.md`（技术博客大纲）

- [ ] **Step 1: 补全 README.md（必含章节）**

1. 标题 + 一句话简介 + 效果截图
2. 架构图（mermaid 流程图：照片 → OCR → 解析 → 规则引擎 + LLM → 报告）
3. 快速开始（安装、建向量库、起后端、起前端，四条命令）
4. 规则清单表（规则 ID / 名称 / 判定逻辑 / 依据）
5. 评估结果表（从 Task 14 输出粘贴）
6. 免责声明（AI 辅助，仅供参考，需人工复核）

- [ ] **Step 2: 写演示视频脚本 `docs/demo-script.md`**

3 分钟结构：开场 15 秒（痛点：人工审标签慢）→ 45 秒（拍一张标签 → 出报告）→ 60 秒（展示报告：风险等级、违规点、依据条款）→ 30 秒（讲规则引擎+大模型架构）→ 30 秒（收尾：可扩展方向）。

- [ ] **Step 3: 写技术博客大纲 `docs/blog-outline.md`**

主题《如何把 GB 7718 规则工程化 + 用大模型兜底》：背景 → 为什么选规则引擎+LLM 混合 → 国标如何拆成规则 → OCR/解析的坑 → LLM 防幻觉设计 → 评估结果 → 踩坑总结。

- [ ] **Step 4: 最终提交**

```bash
git add .
git commit -m "docs: readme, demo script, blog outline"
```

---

## 里程碑对照

| 时间 | 对应任务 | 出口标准 |
| --- | --- | --- |
| 暑假第 1 周 | Task 0~4 | 解析器与规则引擎框架跑通，单元测试全绿 |
| 暑假第 2 周 | Task 5~8 + Task 14 标注模板 | 9 条确定性规则全部实现并测试通过 |
| 暑假第 3 周 | Task 9~13 + Task 14 实拍评估 | 网页端到端可用，20+ 真实标签评估出准确率 |
| 大三上 9~10 月 | 扩展层 C（另立计划） | 抽检通报风险分析 + 规则扩充 + 准确率提升 |
| 大三上 11~12 月 | Task 15 + 扩展层收尾 | GitHub 仓库、演示视频、技术博客上线 |

## 后续扩展（不在本计划内）

- 扩展层 C：市场监管总局抽检通报爬取与"品类 × 违规项目"风险分析（另立计划）
- 特殊膳食标签（GB 13432）、进口食品标签对照、批量审查
- 正面+背面双图上传（MVP 先支持单图）
- 规则扩充：GB 2760 添加剂合法性数据库、营养声称液体/固体分档


## 实现偏差记录（2026-08-11 执行时）

1. 包名 `parser` 改为 `label_parser`：Python 3.9 标准库存在同名内置模块，会遮蔽项目包（3.10+ 无此问题）。所有 `from parser.models import ...` 改为 `from label_parser.models import ...`，解析器模块为 `label_parser/parser.py`。
2. 依赖锁定：`numpy==1.26.4`（paddlepaddle 2.6.2 与 numpy 2.x ABI 不兼容）、`protobuf==3.20.2`（chromadb 默认会把 protobuf 升到 6.x 导致 paddle 无法导入）。
3. `sentence-transformers`（含 torch，约 2~3GB）延后安装：拆到 `requirements-rag.txt`，真机建向量库前再装。
4. 执行环境：基于 `E:\anaconda3\envs\foodai`（Python 3.9.25）创建虚拟环境 `.venv`。
5. 开发分支：`codex/label-compliance-mvp`（未直接在主分支开发）。
6. `R006` 声称项目缺数据时返回 FAIL 而非 SUSPECT（GB 28050 要求声称涉及项目必须标示）。
7. 仓库 Git 身份沿用主仓库占位配置（Developer），可自行改为真实姓名邮箱。
## 标准核对记录（2026-08-11，依据官方文本）

已用 GB 7718-2025 / GB 28050-2025 官方 PDF 核对并修正：

1. **实施日期更正**：两标准均为 2025-03-16 发布、**2027-03-16 实施**（原设计文档误写为 2026-09-01）。现行有效版本为 GB 7718-2011、GB 28050-2011。
2. **R005 营养表**：现行强制 5 项（能量/蛋白质/脂肪/碳水化合物/钠）；GB 28050-2025 新增强制"饱和脂肪、糖"及 NRV%，规则实现为"现行 5 项缺失=违规，新版新增项缺失=存疑（注明实施日期）"。
3. **R002 日期**：GB 7718-2025 要求标示"保质期到期日"（格式见附录 A.2，含 YYYYMMDD 紧凑格式）；生产日期可因保质期≥6 个月等原因省略。规则改为"到期日缺失=存疑（新版要求），两者皆缺=违规"。
4. **R004 致敏物质**：强制清单为 8 类（麸质谷物/甲壳纲类动物/鱼类/蛋类/花生/大豆/乳及乳制品（含乳糖）/坚果）；芝麻、芹菜属自愿标示，已从强制关键字中移除，新增"斯佩耳特小麦、乳糖、甲壳纲"。
5. **R006 声称阈值**：核对表 C.1 确认——低脂 ≤3g/100g(固体)、低糖 ≤5g、无糖 ≤0.5g、低钠 ≤120mg、极低钠 ≤40mg、高钙 ≥30%NRV（钙 NRV=800mg，即 ≥240mg）。新增"无脂 ≤0.5g""极低钠 ≤40mg"两条。
6. **旧标准编号更正**：营养标签旧标准为 GB 28050-2011（非 2013）。