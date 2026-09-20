// ============================================================
//  电子鼻采集固件（ESP32 免焊版）enose_sketch.ino —— 学习注释版
//  用途：MQ 气体阵列(经ADS1115) + BME680 + SGP30 + OLED
//        串口输出 CSV + WiFi 传输（方式一：连路由器）
//
//  【学习导读：这套固件在干什么】
//  简单说就是"轮询 + 输出"：每隔 1 秒，把所有传感器的读数读一遍，
//  拼成一行 CSV 从两个通道发出去：
//    1) USB 串口（插线时）：电脑上的 collect.py 接收；
//    2) WiFi（充电宝供电时）：ESP32 连家里路由器，把同一行数据发到
//       电脑上运行的 collect_wifi.py（电脑也要连同一个 WiFi）。
//  两个通道同时生效、数据一样，插不插 USB 都能用。
//
//  数据流（记熟这个就对了一半）：
//    MQ-2/3/7/135 模拟电压 --ADS1115(16位ADC)--> ESP32
//    BME680(温湿度/气压/VOC) ----I2C----> ESP32
//    SGP30(eCO2/TVOC) -----------I2C----> ESP32
//    OLED(屏幕显示) --------------I2C----> ESP32
//    ESP32 --USB串口/WiFi(TCP)--> 电脑 collect.py / collect_wifi.py --CSV文件--> 训练/分析
//
//  接线（全部用模块版，面包板 + 杜邦线，零焊接）：
//    ESP32 5V  --> AMS1117-3.3 模块 IN
//    AMS1117-3.3 OUT --> 4 个 MQ 模块 VCC（供电充足且不超 ADC 量程）
//    ADS1115   SDA->GPIO21  SCL->GPIO22  VDD->3V3  GND->GND
//              AIN0<-MQ-2 AOUT   AIN1<-MQ-3 AOUT
//              AIN2<-MQ-7 AOUT   AIN3<-MQ-135 AOUT
//    BME680    SDA->GPIO21 SCL->GPIO22 VIN->3V3 GND->GND
//    SGP30     SDA->GPIO21 SCL->GPIO22 VIN->3V3 GND->GND
//    OLED      SDA->GPIO21 SCL->GPIO22 VIN->3V3 GND->GND
//
//  依赖库（Arduino IDE -> 库管理器安装）：
//    Adafruit ADS1X15 / BME680 / SGP30 / SSD1306 / GFX
//   （WiFi 是 ESP32 自带的库，不用额外安装）
//
//  输出格式（每 SAMPLE_INTERVAL_MS 一行）：
//    ts_ms,MQ1,MQ2,MQ3,MQ4,tempC,humPct,pressPa,gasOhm,eco2,tvoc
//  列含义（写论文/分析数据时照着看）：
//    ts_ms    = 开机以来的毫秒数（时间轴）
//    MQ1..4   = 4 路气体传感器的原始读数（0~32767，越大=气体越多）
//    tempC    = 温度（℃）
//    humPct   = 相对湿度（%）
//    pressPa  = 气压（Pa）
//    gasOhm   = BME680 气体传感器电阻（Ω，VOC 越多电阻越小）
//    eco2     = SGP30 估算的 CO2 浓度（ppm）
//    tvoc     = SGP30 估算的总挥发性有机物（ppb）
// ============================================================

// ---- 头文件：引入各模块的驱动库 ----
#include <Wire.h>               // I2C 通信底层库（ESP32 自带）
#include <WiFi.h>               // WiFi 连接（ESP32 自带，不用装）
#include <WiFiClient.h>         // TCP 客户端：把数据发到电脑
#include <Adafruit_ADS1X15.h>   // ADS1115 模数转换器驱动
#include <Adafruit_BME680.h>    // BME680 四合一传感器驱动
#include <Adafruit_SGP30.h>     // SGP30 空气质量传感器驱动
#include <Adafruit_SSD1306.h>   // OLED 屏幕驱动
#include <Adafruit_GFX.h>       // OLED 绘图基础库（画字画图）

// ================= WiFi 配置区（方式一：连路由器）=================
// 用充电宝 + WiFi 前，把下面三样改成你自己的：
const char* WIFI_SSID   = "Xiaomi_1ECE";          // ① 你家 WiFi 名字
const char* WIFI_PASS   = "zfd007895";         // ② 你家 WiFi 密码
const char* PC_HOST_IP  = "192.168.31.111";       // ③ 电脑的局域网IP
                                                 //   先运行 collect_wifi.py 会打印出来
const uint16_t PC_PORT  = 9000;                  // 端口，和 collect_wifi.py 的 --port 一致
// ================================================================

// 断线后每隔多久重试连 WiFi / 连电脑（毫秒）
const unsigned long WIFI_RETRY_MS    = 5000;
const unsigned long CONN_RETRY_MS    = 2000;

WiFiClient client;   // TCP 客户端：连接电脑上的接收程序

// ---- 采样间隔（毫秒），默认 1 秒一行 ----
// 1000 毫秒 = 1 秒。改小=数据更密但文件更大；改大=更省电。
const unsigned long SAMPLE_INTERVAL_MS = 1000;

// ---- ESP32 默认 I2C 引脚 ----
// I2C 只需要两根线：SDA（数据）和 SCL（时钟），所有 I2C 模块并联在这两根线上，
// 靠不同的"地址"区分是谁。这就是为什么那么多模块只用了 2 个引脚。
#define SDA_PIN 21
#define SCL_PIN 22

// ---- OLED SSD1306 0.96寸 ----
#define OLED_ADDR 0x3C           // OLED 的 I2C 地址（0x3C 是标准默认值）
Adafruit_SSD1306 display(128, 64, &Wire, -1);

// ---- ADS1115：读 4 路 MQ 模块输出 ----
// 为什么要它？ESP32 自带的 ADC 精度差、有非线性误差，测气体这种"细微变化"
// 不够用。ADS1115 是 16 位高精度 ADC，专门负责把 MQ 的模拟电压变成数字。
Adafruit_ADS1115 ads;
const int N_MQ = 4;                       // 一共 4 个 MQ 传感器
const char* MQ_NAMES[] = {"MQ2", "MQ3", "MQ7", "MQ135"};  // 名字，OLED 显示用

// ---- BME680：温度/湿度/气压/VOC 四合一 ----
Adafruit_BME680 bme;

// ---- SGP30：eCO2 / TVOC ----
Adafruit_SGP30 sgp;

// ---- 工具函数：把一行数据同时发到串口和 WiFi ----
// 充电宝供电时看不到串口，WiFi 是唯一通道；插着 USB 时两个都看得到。
void sendLine(const char* line) {
  Serial.println(line);              // USB 串口
  if (client.connected()) {          // WiFi 已连上电脑才发
    client.println(line);
  }
}

// ---- 工具函数：开始连 WiFi（异步，不卡住采集）----
// WiFi.begin() 一调用就返回，真正的连接在后台进行，
// 所以就算 WiFi 连不上，传感器采集也照常跑（只是暂时发不出去）。
void startWifi() {
  WiFi.mode(WIFI_STA);              // 站点模式：去连路由器
  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

// setup()：上电后只运行一次。做"初始化"——开串口、开 I2C、挨个唤醒传感器。
// 如果某个传感器没接好/地址不对，会通过串口打印 XXX_ERROR，方便排查。
void setup() {
  Serial.begin(9600);   // 波特率，必须和 collect.py 默认值一致（都是 9600）
  Wire.begin(SDA_PIN, SCL_PIN);   // 启动 I2C 总线，指定引脚

  startWifi();   // 后台开始连 WiFi，不阻塞下面的初始化

  // OLED（可选显示，失败不阻塞）
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("OLED_ERROR");
  } else {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("ENOSE booting...");
    display.display();
  }

  // ADS1115 默认地址 0x48；MQ AOUT 为 0~3.3V，用 ±4.096V 量程安全
  if (!ads.begin(0x48)) {
    Serial.println("ADS1115_ERROR");
  }
  ads.setGain(GAIN_ONE);  // ±4.096V：量程越大越"不灵敏"但越安全

  // BME680 地址自动检测：0x76 失败则试 0x77
  bool bmeOk = bme.begin(0x76);
  if (bmeOk) {
    Serial.println("BME680_OK_0x76");
  } else {
    bmeOk = bme.begin(0x77);
    if (bmeOk) {
      Serial.println("BME680_OK_0x77");
    } else {
      Serial.println("BME680_ERROR");
    }
  }
  bme.setTemperatureOversampling(BME680_OS_8X);  // 温度 8 倍过采样
  bme.setHumidityOversampling(BME680_OS_2X);     // 湿度 2 倍
  bme.setPressureOversampling(BME680_OS_4X);     // 气压 4 倍
  bme.setIIRFilterSize(BME680_FILTER_SIZE_3);    // 数字滤波，减少毛刺
  bme.setGasHeater(320, 150);   // 气体加热 320℃、150ms

  // SGP30（上电后前 10~15 分钟为基线校准期，读数逐渐稳定）
  if (!sgp.begin()) {
    Serial.println("SGP30_ERROR");
  }

  // 给电脑发两个信号（只在串口发，WiFi 接收端会自动过滤掉这类行）：
  // 1) ENOSE_READY：collect.py 识别到就跳过这行（它只收数据行）
  // 2) 表头：CSV 第一行，告诉电脑每列是什么
  Serial.println("ENOSE_READY,warm_up_5min_then_start");
  delay(2000);
  Serial.println("ts_ms,MQ1,MQ2,MQ3,MQ4,tempC,humPct,pressPa,gasOhm,eco2,tvoc");
}

// loop()：不断循环执行。这里是核心：读数据 -> 发出去 -> 刷新屏幕。
void loop() {
  // static = 变量在两次循环之间"记住"上一次的值
  static unsigned long last = 0;     // 上次采样的时间
  static unsigned long lastHum = 0;  // 上次给 SGP30 校准湿度的时间
  static unsigned long lastWifiTry = 0;   // 上次尝试连 WiFi 的时间
  static unsigned long lastConnTry = 0;   // 上次尝试连电脑的时间
  unsigned long now = millis();      // millis() = 开机到现在经过的毫秒数

  // ---- WiFi 维护（不阻塞采集）----
  // 没连上 WiFi：每 WIFI_RETRY_MS 重试一次
  if (WiFi.status() != WL_CONNECTED) {
    if (now - lastWifiTry >= WIFI_RETRY_MS) {
      lastWifiTry = now;
      WiFi.disconnect();
      startWifi();
    }
  } else if (!client.connected()) {
    // WiFi 已连上但没连上电脑：每 CONN_RETRY_MS 重连一次
    // 常见原因：电脑没开接收程序 / 电脑 IP 填错了 / 防火墙拦截
    if (now - lastConnTry >= CONN_RETRY_MS) {
      lastConnTry = now;
      client.stop();
      client.connect(PC_HOST_IP, PC_PORT);
    }
  }

  // 非阻塞定时：不到 1 秒就跳过本次循环，避免用 delay() 卡住整块板。
  if (now - last < SAMPLE_INTERVAL_MS) return;
  last = now;

  // ---- MQ 阵列：ADS1115 单端 4 路，原始 ADC 0~32767 ----
  int16_t mq[N_MQ];
  for (int i = 0; i < N_MQ; i++) {
    mq[i] = ads.readADC_SingleEnded(i);
  }

  // ---- BME680：温度/湿度/气压/气体电阻 ----
  // -999 是"占位符"：传感器读数失败时就用它占位，
  // 这样 CSV 格式不乱，电脑端能识别出这行数据无效。
  float t = -999.0f, h = -999.0f, p = -999.0f, g = -999.0f;
  if (bme.performReading()) {
    t = bme.temperature;
    h = bme.humidity;
    p = bme.pressure;
    g = bme.gas_resistance;     // 气体电阻：VOC 越多电阻越小
  }

  // ---- SGP30：eCO2 / tVOC ----
  int eco2 = -999, tvoc = -999;
  if (sgp.IAQmeasure()) {
    eco2 = sgp.eCO2;
    tvoc = sgp.TVOC;
  }
  // 用 BME680 湿度校准 SGP30（官方建议每 60s 更新一次）
  if (now - lastHum >= 60000 && h >= 0) {
    lastHum = now;
    sgp.setHumidity(h);
  }

  // ---- 拼一行 CSV，串口和 WiFi 各发一份 ----
  char buf[96];
  snprintf(buf, sizeof(buf),
           "%lu,%d,%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%d,%d",
           (unsigned long)now, mq[0], mq[1], mq[2], mq[3],
           t, h, p, g, eco2, tvoc);
  sendLine(buf);

  // ---- OLED 概览：5 行布局 + 第 8 行 WiFi 状态 ----
  if (display.getBuffer() != NULL) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.print("T ");
    if (t > -900) { display.print(t, 1); display.print("C "); } else { display.print("-- "); }
    display.print("H ");
    if (h > -900) { display.print(h, 1); display.print("%"); } else { display.print("--"); }
    display.setCursor(0, 8);
    display.print("P ");
    if (p > -900) { display.print(p / 100.0f, 1); display.print("hPa"); } else { display.print("--"); }
    display.setCursor(0, 16);
    display.print("eCO2 "); display.print(eco2);
    display.print(" TVOC "); display.print(tvoc);
    for (int i = 0; i < N_MQ; i++) {
      display.setCursor(0, 24 + i * 8);
      display.print("MQ"); display.print(i + 1);
      display.print(" "); display.print(mq[i]);
      display.print(" ("); display.print(MQ_NAMES[i]); display.print(")");
    }
    // 第 8 行：WiFi 状态。W:OK = 连上路由器，C:OK = 数据已连到电脑。
    display.setCursor(0, 56);
    if (WiFi.status() == WL_CONNECTED) {
      display.print("W:OK ");
      display.print(client.connected() ? "C:OK" : "C:--");
    } else {
      display.print("W:-- try");
    }
    display.display();
  }
}