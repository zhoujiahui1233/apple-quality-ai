# 电子鼻（E-Nose）项目（ESP32 免焊版）

用 ESP32 + MQ 气体传感器阵列（经 ADS1115 高精度采样）+ BME680 + SGP30，
采集食品的挥发性气体"气味指纹"，配合机器学习判断新鲜度/腐败程度。
与本项目（智果 AI 品控系统）互补：视觉检测看外观，电子鼻闻内部生化变化（生物性危害）。

## 目录

- `protocol.md` — 实验方案（样本设计、客观分级标准、环境控制、12 周时间表）
- `enose_sketch/enose_sketch.ino` — ESP32 固件（传感器读数 → 串口 CSV）
- `collect.py` — Python 上位机（串口 → CSV，带样品标签）
- `preprocess.py` — 数据预处理与特征提取（S-G 滤波 → 60 维特征）
- `plsda.py` — PLS-DA 分类器（偏最小二乘判别，食品/电子鼻论文标配，独立模块便于部署加载）
- `train.py` — 建模：分类（RF / SVM / KNN / LR / PLS-DA）+ 回归（PLSR / RF / SVR）
- `predict_freshness.py` — 单样本预测（分类：新鲜度等级；回归：剩余货架期/放置时间）

## 采购清单（B 档详细版，约 180~250 元，全部免焊）

> 淘宝直接复制「搜索关键词」就能找到；全部选「模块版 / 带排针」，到手即插即用。

| # | 器件 | 具体型号/规格 | 参考价 | 搜索关键词 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 1 | 主控 | ESP32 DevKitC V4（模组 ESP32-WROOM-32，双核 240MHz，4MB Flash，WiFi+蓝牙） | 25~35 | `ESP32 开发板 CH340 带排针` | CH340 版免驱好装；选「已焊排针」版本 |
| 2 | 气体传感器 | MQ-3 模块（酒精/乙醇，LM393 比较器板，AOUT 模拟输出） | 5~10 | `MQ-3 气体传感器模块` | 用 AOUT 接 ADS1115，DOUT 不用 |
| 3 | 气体传感器 | MQ-135 模块（氨气/空气质量，食品腐败标志气体） | 5~10 | `MQ-135 气体传感器模块` | 同上 |
| 4 | 气体传感器 | MQ-2 模块（可燃气体/丙烷/烟雾） | 5~10 | `MQ-2 气体传感器模块` | 同上 |
| 5 | 气体传感器 | MQ-7 模块（一氧化碳） | 5~10 | `MQ-7 气体传感器模块` | 同上 |
| 6 | 数字气体/环境 | BME680 模块（GY-BME680，Bosch，I2C 地址 0x76/0x77；温度/湿度/气压/VOC 四合一） | 40~55 | `GY-BME680 模块` | 重复性比 MQ 好，是数据稳定性的关键 |
| 7 | 数字气体 | SGP30 模块（Sensirion，I2C 0x58；eCO2 + tVOC） | 25~35 | `SGP30 模块` | 只接 3.3V，不能接 5V；首次上电基线校准 10~15 分钟 |
| 8 | ADC | ADS1115 模块（TI 16 位 4 通道，I2C 0x48，可编程增益 PGA） | 12~18 | `ADS1115 模块` | 接 4 路 MQ 模拟输出，读数从 10 位升 16 位 |
| 9 | 显示 | OLED SSD1306 0.96 寸（I2C 0x3C，128×64） | 12~15 | `0.96 OLED SSD1306 I2C` | 蓝/白字都行 |
| 10 | 稳压 | AMS1117-3.3 模块（输入 4.5~12V，输出 3.3V，1A） | 2~4 | `AMS1117-3.3 模块` | 从 5V 稳压给 4 个 MQ 供电，避免拉垮 ESP32 3V3 且不超 ADC 量程 |
| 11 | 面包板 | MB-102 面包板（830 孔，带 2 条电源轨） | 8~15 | `MB-102 面包板` | 所有插接都在这里完成 |
| 12 | 杜邦线 | 公对公 ×20、公对母 ×40（母头接模块排针） | 5~10 | `杜邦线 公对公 公对母 套装` | 建议买彩色的，好查线 |
| 13 | 电源 | Micro-USB / Type-C 数据线 + 5V/2A 充电头或移动电源 | 家里有或 15 | — | 注意看板子接口类型 |
| 14 | 采样容器 | 带盖密封玻璃罐/保鲜盒 | 家里有或 10 | `顶空采样瓶 250ml`（可选） | 实验时用保鲜膜/硅胶垫辅助密封 |
| 15 | 排查工具（推荐） | 数字万用表 | 20~40 | `数字万用表` | 新手排查接线必备，测 3.3V/5V 是否到位 |

> 可选升级：野外/无线采集加 `18650 电池盒 + 5V 升压模块（MH-MINI，约 10 元）`。
> A 档备选（约 150~200 元）：把主控换成 `Arduino Nano 已焊排针`（15~25 元），去掉 ESP32 的 WiFi 能力，
> 其余器件相同，固件把引脚映射改成 Nano 模拟引脚即可。
## 接线（全部免焊：模块排针插面包板 + 杜邦线）

```
ESP32                   传感器
-----                   ------
5V    --> AMS1117-3.3 IN（给 4 个 MQ 模块 VCC 供电，AOUT 不会超 ADC 量程）
3V3   --> ADS1115/BME680/SGP30/OLED VDD（数字模块电流小，直连安全）
GND   --> 所有模块 GND 共地

I2C 总线（并联 4 个数字模块）：
GPIO21(SDA) -> ADS1115 SDA / BME680 SDA / SGP30 SDA / OLED SDA
GPIO22(SCL) -> ADS1115 SCL / BME680 SCL / SGP30 SCL / OLED SCL

MQ 模拟通道（AOUT -> ADS1115 输入）：
MQ-3   AOUT -> ADS1115 AIN0
MQ-135 AOUT -> ADS1115 AIN1
MQ-2   AOUT -> ADS1115 AIN2
MQ-7   AOUT -> ADS1115 AIN3
```

## 快速开始

1. 安装 Arduino IDE，在「文件 → 首选项 → 附加开发板管理器网址」填入
   `https://dl.espressif.com/dl/package_esp32_index.json`，再在开发板管理器安装
   "esp32 by Espressif Systems"。
2. 库管理器安装：`Adafruit ADS1X15`、`Adafruit BME680`、`Adafruit SGP30 Sensor`、
   `Adafruit SSD1306`、`Adafruit GFX Library`。
3. 打开 `enose_sketch.ino`，选择板型（ESP32 Dev Module）和端口，烧录。
4. 打开串口监视器（9600 波特率）确认输出格式：
   `ts_ms,MQ1,MQ2,MQ3,MQ4,tempC,humPct,pressPa,gasOhm,eco2,tvoc`
   - 若打印 `XXX_ERROR`，检查对应模块接线/地址（BME680 地址 0x76，不行改 0x77）。
5. 电脑上装 pyserial：`pip install pyserial`。
6. 查串口号（Windows 设备管理器 → 端口 COMx），开始采集：

```bash
python collect.py --port COM3 --label apple_fresh --out data/apple_day0.csv
```


## 数据流水线（推荐顺序）

```bash
# 1) 采集：每颗样品一次顶空采样，输出到 data/raw/
python collect.py --port COM3 --label fresh --out data/raw/s001_fresh_day0.csv

# 2) 预处理：S-G 滤波 + 每通道 6 特征（start/steady/range/slope1/auc/ratio）× 10 通道 = 60 维
python preprocess.py data/raw --out data/features.csv
# 可选：空罐基线差分校正（每次实验先测空罐）
python preprocess.py data/raw --out data/features.csv --baseline data/raw/baseline_day1.csv

# 3) 多模型对比：RF / SVM / KNN / LR / PLS-DA 五模型分组交叉验证（推荐先跑这个）
python train.py data/features.csv --compare --cv group

# 4) 单模型详细报告：混淆矩阵 / 分类报告 / RF 特征重要性，并存模型
python train.py data/features.csv --model rf --cv group --save enose/models/freshness_rf.pkl
python train.py data/features.csv --model plsda --cv group --save enose/models/freshness_plsda.pkl

# 5) 回归建模：预测剩余货架期/放置小时数（PLSR / RF / SVR）
#    特征表需有 hours 列（每行样本的实际放置时间，小时）
python train.py data/features.csv --mode regress --model plsr --target hours --cv group --save enose/models/shelf_plsr.pkl
python train.py data/features.csv --mode regress --model rf --target hours --cv group --save enose/models/shelf_rf.pkl

# 6) 单样本预测（网页「电子鼻/嗅觉」同款逻辑，可先用单个 CSV 试跑）
python enose/predict_freshness.py data/raw/s001_fresh_day0.csv --model enose/models/freshness_rf.pkl
python enose/predict_freshness.py data/raw/s001_fresh_day0.csv --model enose/models/shelf_plsr.pkl
```

样本命名规范（`preprocess.py` 会从文件名自动推断标签）：
`s001_fresh_day0.csv`、`s002_mid_day1.csv`、`s003_rotten_day3.csv`
（文件名含关键词 `fresh` / `mid` / `rotten` 即作为类别标签；若 CSV 内有 `label` 列则以列为准）。

> 完整实验设计（样本数、分级标准、环境控制、质控、时间表）见 `protocol.md`。

## 实验方法（顶空采样）

1. 传感器通电预热 10~15 分钟（SGP30 首次上电有校准期，读数逐渐稳定后再测）。
2. 测"空罐基线"：密封罐不放样品，采集 1~2 分钟，存为 `baseline.csv`。
3. 把样品放入密封罐，静置 3~5 分钟让气体积累，再采集 1~2 分钟。
4. 同一样品连测多天（第 0 / 1 / 3 / 5 / 7 天），记录每天的 label
   （如 `apple_day0` … `apple_day7`），得到"新鲜到腐败"的时间序列。
5. 采集时同步记录温度、湿度（BME680 已带，固件已输出）。
6. 可选：鲜切苹果（切片）采样——切片挥发物释放比整果强，信噪比更高，详见 `protocol.md` §9；系统页面 `/slice` 可自动计算褐变面积占比与等级。

## 建模

```bash
# 分类（label 列：fresh / mid / rotten ...）
python train.py data/all_days.csv --model rf            # 默认 group 分组交叉验证
python train.py data/all_days.csv --model plsda         # 论文常用的 PLS-DA
python train.py data/all_days.csv --compare             # 五模型对比
python train.py data/all_days.csv --model svm --cv loo  # 留一交叉验证（样本少时用）

# 回归（预测放置小时数，特征表需要有 hours 列）
python train.py data/all_days.csv --mode regress --model plsr --target hours
python train.py data/all_days.csv --mode regress --model rf --target hours --compare
```

**防泄漏评估（重要）**：每个样本加一列 `group`（采集批次/日期），训练时加 `--cv group`，
按批次分组做交叉验证——同批次的样本互相高度相似，若混进训练集和验证集，
准确率会虚高。系统默认就是 `--cv group`；没有 group 列会自动回退普通 K 折并提示。

**模型选择建议**：
- 分类：样本多（>100）用 RF（稳健、可看特征重要性）；样本少或论文需要解释性用 PLS-DA；SVM 适合小样本高维。
- 回归（剩余货架期）：PLS 系列在"多变量高度共线"的传感器数据上表现稳定且可解释（载荷系数）；RF 回归适合非线性、数据量大时；SVR 做对比基线。
- 评估：样本 < 50 用 `--cv loo`（留一）；按批次采集用 `--cv group`；常规用默认 group。

**模型文件（系统自动加载）**：
- `enose/models/freshness_rf.pkl` / `freshness_rf_slice.pkl` — 分类（外观模块 / 切片模块）
- `enose/models/shelf_plsr.pkl` / `shelf_plsr_slice.pkl`（或 shelf_rf）— 回归剩余货架期
- 网页「电子鼻 / 嗅觉」上传 CSV 后会自动同时显示分类等级与预测小时数。

## 新手避坑

- **温湿度影响巨大**：必须同步记录，建模时作为特征或做校正，否则数据不可用。
- **预热稳定**：通电后等 10~15 分钟，读数平稳了再测。
- **基线一致**：每次实验先测空罐，用"样品值 − 基线"做特征。
- **传感器阵列才有意义**：单个传感器区分不了气味，至少 3~4 个不同敏感谱的。
- **MQ 供电注意**：MQ 模块 AOUT 最高接近 VCC，VCC 超过 3.3V 会超 ADS1115 量程，
  因此 MQ 阵列统一用 AMS1117-3.3 从 5V 稳压供电，不要直连 ESP32 3V3（电流也不够）。
- **SGP30 校准**：首次上电前 10~15 分钟数值漂移属正常，正式实验前先预热。
- **数据标记要规范**：label 用统一的命名（如 fresh/half/rotten），训练前检查类别。

## 后续可扩展（自己摸索的方向）

- 与视觉检测融合：同一样品同时拍外观 + 测气味，做多模态分类（网页「切片/综合状态」已预留）
- 启用 ESP32 WiFi：固件预留思路，把每行数据 POST 到智果系统的
  `http://<电脑IP>:5020/api/enose/push`，即可在网页端实时看气味曲线
- 采集更多天数（第 0/1/3/5/7 天）扩充回归样本，让货架期预测更准
- 时序滑窗特征（对每行数据滑窗提取稳态/斜率/面积）与 1D-CNN/LSTM 对比

## 版本说明

- 当前固件为 ESP32 + ADS1115 + BME680 + SGP30（B 档免焊版）。
- 旧版 Arduino Nano（MQ + DHT11）固件已替换，如需回退用 git 历史。

## 无线采集（充电宝供电 + WiFi，方式一：连路由器）

不用 USB 线也能采集：充电宝给 ESP32 供电，ESP32 连家里路由器，把数据通过 WiFi 发到电脑上运行的 `collect_wifi.py`。

数据格式和 `collect.py` 完全一样，训练/分析流程不用改。

### 一次性准备（改一次即可）
1. 打开 `enose_sketch/enose_sketch.ino`，在文件顶部"WiFi 配置区"填三样：
   - `WIFI_SSID` = 你家 WiFi 名字
   - `WIFI_PASS` = 你家 WiFi 密码
   - `PC_HOST_IP` = 电脑的局域网 IP（先运行一次 `collect_wifi.py`，它会打印出来）
2. 重新烧录固件（Arduino IDE 上传）。

### 每次实验步骤
1. 电脑连上同一个 WiFi。
2. 先运行接收端：`python collect_wifi.py --label 你的样品标签`
3. 给 ESP32 上电（充电宝 USB 口），等 OLED 显示 `W:OK C:OK`（W=连上路由器，C=连上电脑）。
4. 窗口开始滚动数据，Ctrl+C 停止，CSV 存到指定文件。

### 注意
- 首次运行接收端，Windows 防火墙弹窗请选"允许访问"（专用网络）。
- 电脑 IP 是路由器分配的，可能变化；连不上时用 `ipconfig` 重新查，改回固件里。
- 充电宝建议选有"小电流模式"的，防止中途自动断电；充电宝放箱内时放远一点，避免影响温度读数。