# 苹果表面缺陷识别系统 — 学习文档

## 一、项目总体思路

本项目通过计算机视觉技术，自动检测苹果表面的缺陷（腐烂、磕伤、斑点等），
并根据缺陷面积占比对苹果进行品质分级。整个流程分为以下核心步骤：

1. **图像加载与预处理** - 缩放、降噪、颜色空间转换
2. **苹果区域分割** - 从背景中分离出苹果本身
3. **缺陷检测** - 在苹果区域内查找颜色异常的像素
4. **品质分级** - 根据缺陷占比判定等级（优质/合格/不合格）
5. **结果可视化** - 在图上标注缺陷和等级信息
6. **Web 服务** - 通过浏览器上传图片并查看结果
7. **SVM 模型** - 可选的机器学习辅助判定

---

## 二、技术栈

| 技术 | 用途 |
|------|------|
| Python 3 | 编程语言 |
| OpenCV (cv2) | 图像处理核心库 |
| NumPy | 数组运算、图像矩阵操作 |
| PIL / Pillow | 中文文字渲染到图片 |
| Flask | Web 服务器框架 |
| SQLAlchemy | 数据库 ORM（保存检测记录） |
| scikit-learn (SVM) | 机器学习分类模型 |
| Bootstrap | 前端 UI 框架 |

---

## 三、项目文件结构

```
E:/apple/
|-- apple_defect_detector.py   # 核心检测模块（所有算法）
|-- app.py                      # Flask Web 服务器
|-- train_model.py              # SVM 模型训练脚本
|-- models.py                   # 数据库模型定义
|-- requirements.txt            # Python 依赖列表
|-- apple_svm_model.dat         # 训练好的 SVM 模型文件
|-- apple_svm_model.pkl         # SVM 模型备份
|--
|-- templates/
|   |-- base.html              # 网页基础模板
|   |-- index.html             # 主页（上传 + 检测）
|   |-- history.html           # 历史记录页
|   |-- detail.html            # 单条记录详情页
|--
|-- static/
|   |-- uploads/               # 上传图片 + 结果图片存放
|--
|-- images/
|   |-- test_good_apple.jpg    # 测试用：好苹果
|   |-- test_bad_apple.jpg     # 测试用：坏苹果
|--
|-- output/                    # 命令行模式输出目录
|-- instance/                  # SQLite 数据库存放
|-- kaggle_data/               # Kaggle 数据集（训练 SVM 用）
|-- DOCUMENTATION.md           # 本学习文档
```

---

## 四、核心算法详解

### 4.1 图像预处理 `preprocess()`

```python
def preprocess(img):
    # 1. 等比例缩放到统一宽度 500px
    # 2. 双边滤波去噪（保留边缘的同时降噪）
    return img_blurred, img_resized
```

**为什么这么做？**
- 统一图片尺寸使后续处理的参数保持一致
- 双边滤波比高斯模糊更能保留苹果边缘细节

### 4.2 苹果区域分割 `segment_apple()`

这是最重要的一步：要把苹果从背景中准确分离出来。

```python
def segment_apple(img_bgr):
    # 步骤 1：霍夫圆检测 — 找到图片中的圆形物体
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                                dp=1.2, minDist=100,
                                param1=100, param2=35,
                                minRadius=30, maxRadius=250)

    # 步骤 2：在圆内用 HSV 颜色过滤，只保留苹果色像素
    # 这样即使圆画大了，背景也不会被算进苹果区域
    for lo, up in APPLE_HSV_RANGES:
        color_mask += cv2.inRange(hsv, lo, up)
    mask = circle_mask & color_mask

    # 步骤 3：排除肤色（防止手持时手指被识别）
    skin_mask = cv2.inRange(hsv, SKIN_HSV_LOWER, SKIN_HSV_UPPER)
    mask = mask & ~skin_mask

    # 步骤 4：形态学操作清理边缘
    mask = cv2.morphologyEx(mask, MORPH_CLOSE, kernel)
    return mask, apple_contour
```

**关键技术点：**

| 技术 | 作用 |
|------|------|
| HoughCircles | 检测圆形物体，返回圆心和半径 |
| HSV 颜色空间 | 比 RGB 更接近人眼对颜色的感知，易于设定颜色范围 |
| APPLE_HSV_RANGES | 定义苹果颜色范围（红/绿/黄/青苹果都覆盖） |
| SKIN_HSV_LOWER/UPPER | 肤色范围，用于排除手持图片中的手 |
| 形态学闭运算 | 填充苹果内部的细小空洞 |
| 形态学腐蚀 | 让苹果边缘稍微收缩，去掉背景杂边 |

**为什么用霍夫圆 + HSV 结合，而不是只用 HSV？**
因为背景中可能有和苹果颜色相近的物体（如木桌、纸箱），
霍夫圆先框定位置，HSV 再精确取色，两者结合更可靠。

### 4.3 缺陷检测 `detect_defects()`

```python
def detect_defects(img_bgr, apple_mask, sensitivity=5, min_area=200):
    # 步骤 1：在 HSV 空间中查找颜色异常的像素
    for lo, up in DEFECT_HSV_RANGES:
        # sensitivity 控制 V 通道的缩放
        v_scale = 0.6 + (sensitivity - 1) * 0.1
        scaled_lo = [lo[0], lo[1], lo[2] * v_scale]
        defect_mask += cv2.inRange(hsv, scaled_lo, scaled_up)

    # 步骤 2：只保留苹果区域内的缺陷
    defect_mask &= apple_mask

    # 步骤 3：过滤掉太小的噪点
    contours = findContours(defect_mask)
    defect_contours = [c for c in contours if area(c) >= min_area]
    return filtered_mask, defect_contours
```

**缺陷 HSV 范围说明：**

| 范围 | 检测目标 |
|------|---------|
| H 5-40, S 40-200, V 25-150 | 棕色腐烂区域 |
| H 0-180, S 0-150, V 0-8 | 极暗区域（严重腐烂） |
| H 5-25, S 25-130, V 30-80 | 深棕色斑点 |
| H 150-180, S 40-200, V 10-60 | 深红色淤伤 |

**灵敏度参数 (1-10)：**
- 1 = 最严格（`v_scale=0.6`）：只检测最明显的缺陷
- 5 = 默认（`v_scale=1.0`）：平衡模式
- 10 = 最宽松（`v_scale=1.4`）：不容易漏检

### 4.4 品质分级 `classify_quality()`

```python
def classify_quality(defect_pixels, apple_pixels,
                    premium_th=0.05, standard_th=0.15):
    defect_ratio = defect_pixels / apple_pixels
    if defect_ratio < premium_th:     # < 5%  → 优质
        return 'Premium'
    elif defect_ratio <= standard_th:  # 5%-15% → 合格
        return 'Standard'
    else:                              # > 15%  → 不合格
        return 'Reject'
```

**阈值可以在 Web 界面实时调节：**
- 优质阈值：1%-15%（默认 5%）
- 合格阈值：5%-30%（默认 15%）

### 4.5 结果可视化 `visualize_result()`

```python
def visualize_result(img_bgr, apple_contour, defect_contours,
                   defect_mask, grade, defect_ratio, grade_color):
    # 1. 绿色轮廓线标出苹果边界
    cv2.drawContours(result, [apple_contour], -1, GREEN, 2)
    # 2. 红色半透明标记缺陷区域
    overlay[defect_mask > 0] = RED
    result = cv2.addWeighted(overlay, 0.5, result, 0.5, 0)
    # 3. 用 PIL + 微软雅黑 渲染中文文字
    from PIL import ImageFont, ImageDraw
    draw.text((15, 8), '等级: ' + grade, ...)
    draw.text((15, 40), '缺陷占比: xx%', ...)
    draw.text((15, 62), '缺陷数量: xx', ...)
    return result
```

**注意：** OpenCV 的 `cv2.putText()` 不支持中文。
所以用 PIL 加载 Windows 系统字体 `msyh.ttc`（微软雅黑）来绘制中文。
如果字体加载失败，自动降级为英文显示。

---

## 五、SVM 机器学习模型

### 5.1 为什么需要 SVM？
HSV 阈值法是手工规则，对复杂场景（不同光照、不同品种）适应性有限。
SVM 通过学习大量样本，自动寻找最优分类边界，可作为辅助判定。

### 5.2 特征提取（14 维特征向量）

| # | 特征 | 含义 |
|---|------|------|
| 1-2 | H 通道均值/标准差 | 苹果整体色调 |
| 3-4 | S 通道均值/标准差 | 颜色饱和度 |
| 5-6 | V 通道均值/标准差 | 亮度分布 |
| 7 | 苹果面积占比 | 苹果占画面比例 |
| 8-10 | 3 个 HSV 范围的缺陷比例 | 各类缺陷占比 |
| 11-12 | 拉普拉斯纹理均值/标准差 | 表面粗糙程度 |
| 13 | 圆形度 | 苹果形状接近圆的程度 |
| 14 | 凸度（Solidity） | 苹果饱满程度 |

### 5.3 训练流程

1. 从 Kaggle 下载苹果数据集（新鲜苹果 + 腐烂苹果）
2. 对每张图片提取 14 维特征
3. 正负样本平衡（新鲜和腐烂数量相等）
4. 训练 RBF 核 SVM 分类器
5. 保存模型为 `apple_svm_model.dat`

### 5.4 注意事项
SVM 模型在 Web 界面中仅供参考，不作为最终判定标准。
因为 SVM 训练数据来自 Kaggle 数据集，
而 HSV 检测基于实际拍摄图片，两者环境不同。

---

## 六、Web 应用结构（Flask）

### 6.1 路由说明

| 路由 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 主页，上传检测界面 |
| `/upload` | POST | 接收图片，执行检测，返回 JSON 结果 |
| `/history` | GET | 历史检测记录 |
| `/detail/<id>` | GET | 单条检测详情 |
| `/delete/<id>` | POST | 删除记录 |
| `/api/stats` | GET | 统计数据 |

### 6.2 前端上传流程

1. 用户拖拽或点击选择图片
2. JavaScript 读取调参面板的数值（灵敏度/面积/阈值）
3. `FormData` 封装图片 + 参数 → `POST /upload`
4. 后端执行检测流程，返回 JSON
5. JS 更新页面：显示原图、结果图、等级、缺陷信息

### 6.3 调参面板

面板展开后可调节：
- **检测灵敏度** (1-10)：控制缺陷检测的严格程度
- **最小缺陷面积** (50-500px)：过滤掉太小的噪点
- **优质阈值** (1%-15%)：低于此值评为优质
- **合格阈值** (5%-30%)：高于此值评为不合格

点击"恢复默认"重置为推荐值。

---

## 七、运行指南

### 7.1 环境要求

```
Python 3.8+
opencv-python
numpy
Pillow
flask
flask-sqlalchemy
```

### 7.2 启动 Web 服务

```bash
cd E:/apple
python app.py
# 访问 http://127.0.0.1:5020
```

### 7.3 命令行模式（单张检测）

```bash
python apple_defect_detector.py images/test_good_apple.jpg
python apple_defect_detector.py images/test_bad_apple.jpg
# 也可以检测整个目录：
python apple_defect_detector.py images/
```

### 7.4 训练 SVM 模型

```bash
python train_model.py
```

### 7.5 在 PyCharm 中打开

1. 打开 PyCharm
2. `File` → `Open` → 选择 `E:/apple` 文件夹
3. 确保 Python 解释器已安装 OpenCV、Flask 等依赖
4. 运行 `app.py` 启动 Web 服务

---

## 八、常见问题

### Q: 为什么好苹果也被判为不合格？
**A：** 常见原因：
1. 背景被误识别为苹果区域 → 解决方案：HSV 颜色过滤 + 圆范围从 1.3x 缩小到 1.15x
2. 苹果柄/花萼被识别为缺陷 → 解决方案：收紧暗色缺陷范围（V<60 改为 V<8）
3. 阴影/反光被误识别 → 解决方案：提高饱和度下限（S<30 改为 S<40）
4. 手持图片手指被识别 → 解决方案：新增肤色排除（SKIN_HSV 范围）

### Q: 检测结果图上中文显示为问号？
**A：** `cv2.putText()` 不支持中文。代码已改用 PIL + 微软雅黑字体渲染中文。
确保系统中有 `C:/Windows/Fonts/msyh.ttc` 文件。

### Q: 调节面板没反应？
**A：** 旧版本中调节参数没有发送到服务器。已修复：前端通过 FormData 附带参数，
后端 `detect_defects()` 和 `classify_quality()` 支持接收这些参数。

### Q: 显示"检测出错"？
**A：** 常见原因：HTML 中缺少 `svmDisplay` 元素，导致 JavaScript 报错。
已修复该问题。如果再次出现，检查浏览器控制台（F12）查看具体错误。

---

## 九、改进方向

1. **深度学习模型**：用 YOLO / Faster-RCNN 代替 HSV 阈值法，检测更准确
2. **更多缺陷类型**：扩展 HSV 范围覆盖更多缺陷（霉斑、虫眼、冻伤）
3. **多角度检测**：拍摄多张不同角度的图片综合判断
4. **批量处理**：支持批量上传和 CSV 导出
5. **实时视频**：接入摄像头进行实时检测

---

## 十、参数速查表（核心常量）

| 常量 | 默认值 | 说明 |
|------|--------|------|
| IMAGE_WIDTH | 500 | 预处理缩放宽度 |
| PREMIUM_THRESHOLD | 0.05 (5%) | 优质缺陷上限 |
| STANDARD_THRESHOLD | 0.15 (15%) | 合格缺陷上限 |
| APPLE_HSV_RANGES | 5 组范围 | 覆盖红/绿/黄/金色/亮白苹果 |
| DEFECT_HSV_RANGES | 4 组范围 | 覆盖棕/黑/深红缺陷 |
| SKIN_HSV_LOWER | [0,18,60] | 肤色排除下限 |
| SKIN_HSV_UPPER | [20,120,255] | 肤色排除上限 |
| Hough minRadius | 30 | 最小苹果半径(px) |
| Hough maxRadius | 250 | 最大苹果半径(px) |
| 默认灵敏度 | 5 | 范围 1-10 |
| 默认最小缺陷面积 | 200 px | 过滤小噪点 |

---

*文档版本: 2026-07-27*