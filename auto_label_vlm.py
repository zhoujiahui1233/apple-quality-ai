# -*- coding: utf-8 -*-
"""
VLM 自动预标注（苹果外观缺陷）
================================
用多模态大模型给未标注图片批量生成"缺陷/苹果候选框"，输出：
  - YOLO 格式标签（可直接并入 defect_dataset_yolo 重训）
  - 可视化预览图（快速目检）
  - summary.csv / review 清单（高置信度自动入库、低置信度人工复核）

默认模型：智谱 GLM-4V-Flash（永久免费，国内直连，OpenAI 兼容接口）
注册拿 Key（1 分钟）：https://open.bigmodel.cn/

用法示例（Windows 命令行里用双反斜杠或正斜杠均可）：
  python auto_label_vlm.py --source E:/apple/yolo_pseudo/images --out E:/apple/auto_label_output
  python auto_label_vlm.py --source <图片目录> --out <输出目录> --limit 20   # 先试 20 张
  python auto_label_vlm.py --source <图片目录> --out <输出目录> --resume     # 断点续跑
  python auto_label_vlm.py --selftest                                       # 不联网自检

Key 配置（三选一，推荐第 1 种）：
  1) 与本文件同目录放 vlm_config.json：
       {"api_key":"你的key","base_url":"https://open.bigmodel.cn/api/paas/v4",
        "model":"glm-4v-flash","source":"","out":"","high_conf":0.6}
  2) 环境变量 VLM_API_KEY / VLM_BASE_URL / VLM_MODEL
  3) 命令行 --api-key / --base-url / --model

想换别的模型（Qwen-VL / GPT / Gemini，只要 OpenAI 兼容接口）：
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --model qwen-vl-max
"""
import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np
import requests

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODELS = ["glm-4v-flash", "glm-4v-plus", "glm-4.1v-thinking-flash"]
DEFAULT_CLASSES = ["defect"]  # 单一"缺陷"类，与 defect_dataset_yolo 的 nc=1 一致

# 细分缺陷类型（当 --defect-types 给出时启用）。英文名映射到系统中文名。
TYPE_CN = {
    "bruise": "碰伤/压伤",
    "rot": "腐烂",
    "scab": "疮痂/黑斑",
    "wound": "伤口",
    "mold": "霉斑",
    "defect": "缺陷",
}


def load_config(args) -> dict:
    """合并 vlm_config.json + 环境变量 + 命令行参数（命令行优先）。"""
    cfg = {"api_key": "", "base_url": DEFAULT_BASE_URL, "model": "",
           "models": list(DEFAULT_MODELS), "source": "", "out": "",
           "high_conf": 0.6, "max_side": 1100, "sleep": 0.4,
           "coord_scale": 1,
           "defect_types": []}
    cfg_file = SCRIPT_DIR / "vlm_config.json"
    if cfg_file.exists():
        try:
            user = json.loads(cfg_file.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in user.items() if v not in (None, "")})
        except Exception as e:
            print("[警告] vlm_config.json 读取失败:", e)
    cfg["api_key"] = os.environ.get("VLM_API_KEY") or cfg["api_key"]
    cfg["base_url"] = os.environ.get("VLM_BASE_URL") or cfg["base_url"]
    cfg["model"] = os.environ.get("VLM_MODEL") or cfg["model"]

    # 命令行覆盖
    for attr in ("api_key", "base_url", "model", "source", "out"):
        val = getattr(args, attr)
        if val:
            cfg[attr] = val
    if args.high_conf is not None:
        cfg["high_conf"] = args.high_conf
    if args.defect_types:
        cfg["defect_types"] = [t.strip().lower() for t in args.defect_types.split(",") if t.strip()]
    if cfg["model"]:
        cfg["models"] = [cfg["model"]] + [m for m in cfg["models"] if m != cfg["model"]]
    # Qwen 系列模型默认用 0~1000 坐标；GLM 用 0~1
    if not cfg.get("coord_scale") or cfg["coord_scale"] == 1:
        joined = " ".join(cfg["models"]).lower()
        cfg["coord_scale"] = 1000 if "qwen" in joined else 1
    return cfg


def read_image(path: Path) -> np.ndarray:
    """兼容中文路径读图。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def to_b64_jpeg(img: np.ndarray, max_side: int) -> str:
    """缩放到 max_side 内并编码为 base64 JPEG（控制 token 成本）。"""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("JPEG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_prompt(defect_types: list, coord_scale: int) -> str:
    if defect_types:
        allowed = ",".join(defect_types)
        hint = (
            "缺陷类型(type)只能从以下选择: " + allowed + "；"
            "对应外观提示：bruise=碰伤/压伤(果皮凹陷变色), rot=腐烂(软烂/发黑/霉斑), "
            "scab=疮痂/黑斑/星状病斑, wound=伤口/裂口/虫孔。类型拿不准就写 defect。"
        )
    else:
        allowed = "defect"
        hint = "所有缺陷统一 type 用 defect（不需要细分类型）。"
    if coord_scale and coord_scale > 1:
        coord_note = ("坐标一律用 0~{coord_scale} 的整数（0 表示图片左/上边缘，"
                      f"{coord_scale} 表示右/下边缘），不要用 0~1 小数。")
    else:
        coord_note = "坐标一律用 0~1 的小数（最多 4 位）。"
    return f"""你是苹果外观缺陷标注专家。请仔细看这张图，标出图中每一个苹果，以及苹果表面每一处真正的缺陷。

规则：
1. 只标苹果果实表面的真正缺陷：碰伤、腐烂、疮痂/黑斑、伤口/裂口、霉斑等。
2. 严禁标注：阴影、高光/反光、苹果梗(果柄)、果萼/花萼凹陷、背景物体、果核区域（如果是切片图）。
3. 同一苹果上多个独立缺陷要分开框；密集小斑无法逐个分辨时，框住整片密集区域并在 conf 上适当降分。
4. 框要贴住缺陷实际占用范围，不要外扩到旁边好果肉，也不要漏掉边缘的小斑。
5. 一个苹果通常一个框；框住苹果完整可见轮廓（含果柄坑不影响）。
6. 无苹果或苹果完全干净无缺陷时，对应列表返回空。

{hint}

只输出 JSON，禁止输出任何解释文字。格式：
{{"apples":[{{"box":[x1,y1,x2,y2]}}],"defects":[{{"type":"{allowed}","box":[x1,y1,x2,y2],"conf":0.9}}]}}
要求：{coord_note}（x1<x2,y1<y2）；conf 是 0~1 的置信度（>=0.5 才算有把握的缺陷）。"""


def call_vlm(cfg: dict, img_b64: str, prompt: str, log_path: Path) -> dict:
    """调用 OpenAI 兼容视觉接口，带模型回退与重试。返回解析后的 dict。"""
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + img_b64}},
            ],
        }],
        "temperature": 0.0,
    }
    last_err = None
    for model in cfg["models"]:
        payload["model"] = model
        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=180)
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                    log_path.write_text(text, encoding="utf-8")
                    parsed = parse_json(text)
                    parsed["_model"] = model
                    return parsed
                if resp.status_code in (401, 403):
                    raise RuntimeError(f"鉴权失败({resp.status_code})，请检查 api_key。"
                                       f" 响应: {resp.text[:200]}")
                if resp.status_code == 404:
                    last_err = f"模型 {model} 不存在或不可用"
                    break  # 换下一个模型
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:150]}"
                    time.sleep(5 * (attempt + 1))
                    continue
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                break
            except requests.RequestException as e:
                last_err = f"网络错误: {e}"
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"所有模型调用失败。最后错误: {last_err}")


def parse_json(text: str) -> dict:
    """从模型输出里稳健提取 JSON。"""
    text = (text or "").strip()
    # 去掉可能的 ```json 围栏
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    candidates = []
    try:
        candidates.append(json.loads(text))
    except Exception:
        pass
    for m in re.finditer(r"\{.*\}", text, flags=re.S):
        try:
            candidates.append(json.loads(m.group(0)))
        except Exception:
            continue
    if not candidates:
        # ---- 容错：模型把 JSON 写坏了时，用正则抢救 4 元数组框 ----
        return salvage_boxes(text)
    # 优先选含 apples/defects 键的
    for c in candidates:
        if isinstance(c, dict) and ("apples" in c or "defects" in c):
            return c
    return candidates[-1]


def salvage_boxes(text: str) -> dict:
    """
    容错解析：当模型输出类似
      "apples [{ "box": [0.1,0.2,0.3,0.4}], "defects": []
    这种坏 JSON 时，按 "apples/defects" 关键字分段，把每一段里的
    4 元数字数组当作框抢救出来。
    """
    box_re = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*"
                        r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]")
    # 找出 apples/defects 关键字的出现位置（谁在前就归谁，直到下一个关键字）
    marks = []
    for kw in ("apples", "defects"):
        for m in re.finditer(r'"?\s*%s\s*"?\s*[:=]?\s*[\[{]' % kw, text, flags=re.I):
            marks.append((m.start(), kw))
    marks.sort()
    if not marks:
        # 关键字都没了：把全文所有框都当 defects（保守）
        boxes = [{"box": [float(v) for v in m.groups()], "type": "defect", "conf": 0.5}
                 for m in box_re.finditer(text)]
        return {"apples": [], "defects": boxes}

    sections = []
    for i, (pos, kw) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        sections.append((kw, text[pos:end]))

    apples, defects = [], []
    for kw, seg in sections:
        boxes = [{"box": [float(v) for v in m.groups()]} for m in box_re.finditer(seg)]
        if kw == "apples":
            apples.extend(boxes)
        else:
            defects.extend({**b, "type": "defect", "conf": 0.5} for b in boxes)
    if not apples and not defects:
        raise ValueError("模型输出中没有可解析的 JSON")
    return {"apples": apples, "defects": defects}


def clamp_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 - x1 < 0.002 or y2 - y1 < 0.002:
        return None  # 太小，丢弃
    return x1, y1, x2, y2


def normalize_results(data: dict, classes: list, defect_types: list,
                      img_w: int, img_h: int, coord_scale: int = 1) -> tuple:
    """
    把 VLM 原始输出整理成:
      apples: [(cx,cy,w,h)]  YOLO 归一化
      defects: [(cls_id, cx, cy, w, h, conf, type_str)]
    并做边界 clamp 与合理性过滤。
    """
    apples, defects = [], []
    raw_apples = [a for a in (data.get("apples") or []) if isinstance(a, dict)]
    raw_defects = [d for d in (data.get("defects") or []) if isinstance(d, dict)]
    # 自动判断模型实际用的坐标尺度（0~1000 还是 0~1）
    vals = []
    for box in [a.get("box") for a in raw_apples] + [d.get("box") for d in raw_defects]:
        try:
            vals += [abs(float(v)) for v in box]
        except Exception:
            pass
    eff_scale = coord_scale if (coord_scale and coord_scale > 1
                                and vals and max(vals) > 1.5) else 1
    # 类型 -> id。多类时 id 从 1 开始（0 保留给单类 defect 场景为兼容原有数据集）
    type_map = {t: i + 1 for i, t in enumerate(defect_types)} if defect_types else {}

    def norm_box(box):
        try:
            if len(box) != 4:
                return None
            b = [float(v) / eff_scale for v in box]
        except Exception:
            return None
        cb = clamp_box(b, img_w, img_h)
        if cb is None:
            return None
        x1, y1, x2, y2 = cb
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1
        return cx, cy, w, h

    for a in raw_apples:
        nb = norm_box(a.get("box"))
        if nb:
            apples.append(nb)

    for d in raw_defects:
        nb = norm_box(d.get("box"))
        if not nb:
            continue
        t = str(d.get("type", "defect")).strip().lower()
        if defect_types and t not in type_map:
            # 未知类型：归为 defect（id 0）
            cls_id = 0
        else:
            cls_id = type_map.get(t, 0)
        try:
            conf = float(d.get("conf", 0.5))
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        defects.append((cls_id, nb[0], nb[1], nb[2], nb[3], conf, t))
    return apples, defects


def save_yolo_txt(label_dir: Path, stem: str, defects: list) -> Path:
    label_dir.mkdir(parents=True, exist_ok=True)
    txt = label_dir / f"{stem}.txt"
    lines = []
    for cls_id, cx, cy, w, h, conf, t in defects:
        lines.append(f"{cls_id} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}")
    txt.write_text("\n".join(lines), encoding="utf-8")
    return txt


def draw_viz(img: np.ndarray, apples: list, defects: list,
             apple_cls: tuple, defect_cls: tuple, classes: list) -> np.ndarray:
    h, w = img.shape[:2]
    vis = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for cx, cy, bw, bh in apples:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), apple_cls, 2)
    for cls_id, cx, cy, bw, bh, conf, t in defects:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), defect_cls, 3)
        name = classes[cls_id] if 0 <= cls_id < len(classes) else t
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
        ty = y1 - 6 if y1 - 6 > th else y1 + th + 6
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 6, ty), defect_cls, -1)
        cv2.putText(vis, label, (x1 + 3, ty - 3), font, 0.55, (255, 255, 255), 2)
    return vis


def encode_write(img: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError(f"可视化图写入失败: {path}")
    buf.tofile(str(path))


def process_one(cfg: dict, img_path: Path, out_dir: Path, classes: list,
                defect_types: list) -> dict:
    rec = {"file": img_path.name, "status": "ok", "n_apple": 0, "n_defect": 0,
           "max_conf": 0.0, "types": "", "label": "", "viz": "", "msg": ""}
    try:
        img = read_image(img_path)
        h, w = img.shape[:2]
        b64 = to_b64_jpeg(img, int(cfg["max_side"]))
        log_dir = out_dir / "raw_responses"
        log_dir.mkdir(parents=True, exist_ok=True)
        data = call_vlm(cfg, b64, build_prompt(defect_types, int(cfg.get("coord_scale", 1))),
                        log_dir / f"{img_path.stem}.json")
        apples, defects = normalize_results(
            data, classes, defect_types, w, h, int(cfg.get("coord_scale", 1)))
        # 人工可读的细分类型回显
        type_show = ",".join(sorted({TYPE_CN.get(t, t) for _, _, _, _, _, _, t in defects}))
        rec.update(n_apple=len(apples), n_defect=len(defects),
                   max_conf=max([d[5] for d in defects], default=0.0),
                   types=type_show)

        label_dir = out_dir / "labels"
        viz_dir = out_dir / "viz"
        txt = save_yolo_txt(label_dir, img_path.stem, defects)
        rec["label"] = str(txt)
        if apples or defects:
            vis = draw_viz(img, apples, defects,
                           (0, 200, 0), (0, 0, 255), classes)
            viz_path = viz_dir / f"{img_path.stem}.jpg"
            encode_write(vis, viz_path)
            rec["viz"] = str(viz_path)
        else:
            rec["status"] = "none"  # VLM 认为无缺陷/无苹果
        if rec["status"] == "ok" and rec["max_conf"] < cfg["high_conf"]:
            rec["status"] = "review"
    except Exception as e:
        rec["status"] = "error"
        rec["msg"] = str(e)[:300]
    return rec


def collect_images(source: Path, limit: int, resume: bool, out_dir: Path) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in source.rglob("*") if p.suffix.lower() in exts)
    if resume:
        label_dir = out_dir / "labels"
        done = {p.stem for p in label_dir.glob("*.txt")} if label_dir.exists() else set()
        files = [p for p in files if p.stem not in done]
    if limit:
        files = files[:limit]
    return files


def main():
    ap = argparse.ArgumentParser(description="VLM 自动预标注（苹果缺陷）")
    ap.add_argument("--source", default="", help="未标注图片目录")
    ap.add_argument("--out", default="", help="输出目录（labels/viz/summary.csv）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（试水）")
    ap.add_argument("--resume", action="store_true", help="跳过已生成标签的图")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--high-conf", type=float, default=None,
                    help="高置信度阈值（默认 0.6）")
    ap.add_argument("--defect-types", default="",
                    help="细分类型，逗号分隔: bruise,rot,scab,wound（默认不分，全部 defect）")
    ap.add_argument("--selftest", action="store_true", help="不联网自检解析/绘图逻辑")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    cfg = load_config(args)
    if not cfg["api_key"]:
        print("=" * 60)
        print("缺少 API Key！")
        print("注册（免费）：https://open.bigmodel.cn/  → 创建 API Key")
        print("然后任选一种：")
        print("  1) 在本文件同目录建 vlm_config.json:")
        print('     {"api_key":"你的key"}')
        print("  2) 或设置环境变量 VLM_API_KEY")
        print("  3) 或加参数 --api-key 你的key")
        print("=" * 60)
        sys.exit(2)

    source = Path(cfg["source"])
    if not source.is_dir():
        print(f"图片目录不存在: {source}")
        print("示例：python auto_label_vlm.py --source E:\\apple\\yolo_pseudo\\images "
              "--out E:\\apple\\auto_label_output --limit 20")
        sys.exit(2)

    out_dir = Path(cfg["out"]) if cfg["out"] else SCRIPT_DIR / "auto_label_output"
    classes = list(DEFAULT_CLASSES) + cfg["defect_types"]
    print(f"[配置] base_url={cfg['base_url']}")
    print(f"[配置] models={cfg['models']}")
    print(f"[配置] 高置信阈值={cfg['high_conf']}  细分类型={cfg['defect_types'] or '无(全按 defect)'}")
    print(f"[输入] {source} -> [输出] {out_dir}")

    files = collect_images(source, args.limit, args.resume, out_dir)
    if not files:
        print("没有需要处理的图片（目录为空或已全部处理完）")
        return
    print(f"[开始] 共 {len(files)} 张待处理...\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.time()
    for i, fp in enumerate(files, 1):
        rec = process_one(cfg, fp, out_dir, classes, cfg["defect_types"])
        records.append(rec)
        icon = {"ok": "OK  ", "none": "无缺陷", "review": "复核", "error": "失败"}.get(
            rec["status"], rec["status"])
        print(f"[{i}/{len(files)}] {rec['file']} | {icon} | "
              f"苹果{rec['n_apple']} 缺陷{rec['n_defect']} "
              f"conf最高{rec['max_conf']:.2f} {rec['types']} {rec['msg']}")
        time.sleep(float(cfg.get("sleep", 0.4)))
        if i % 20 == 0:
            eta = (time.time() - t0) / i * (len(files) - i)
            print(f"    ...已 {i} 张，预计还需 {eta/60:.1f} 分钟")

    # ---- 汇总 ----
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        wr.writeheader()
        wr.writerows(records)

    ok_cnt = sum(1 for r in records if r["status"] == "ok")
    none_cnt = sum(1 for r in records if r["status"] == "none")
    review_cnt = sum(1 for r in records if r["status"] == "review")
    err_cnt = sum(1 for r in records if r["status"] == "error")

    review_txt = out_dir / "review_list.txt"
    with open(review_txt, "w", encoding="utf-8") as f:
        for r in records:
            if r["status"] in ("review", "none", "error"):
                f.write(r["file"] + "\n")

    print("\n" + "=" * 60)
    print(f"完成 {len(records)} 张 -> {out_dir}")
    print(f"  可直接入库(高置信): {ok_cnt}")
    print(f"  需人工复核(低置信/无缺陷/失败): {review_cnt + none_cnt + err_cnt}")
    print(f"  复核清单: {review_txt}")
    print(f"  明细:      {csv_path}")
    print("  标签:      " + str(out_dir / 'labels'))
    print("  预览图:    " + str(out_dir / 'viz'))
    print("\n下一步：")
    print("  1) 打开 viz 目录扫一眼框得对不对；")
    print("  2) 用 X-AnyLabeling / LabelImg 打开原图目录，输出目录指向 labels，删错框/补漏框；")
    print("  3) 复核完把 labels 合并进 E:\\apple\\defect_dataset_yolo 后到 Kaggle 重训。")
    print("=" * 60)


def run_selftest():
    """不联网验证：JSON 解析、坐标换算、绘图、写标签。"""
    print("[自检] 解析 JSON ...")
    fake = ('```json\n{"apples":[{"box":[0.1,0.1,0.9,0.9]}],'
            '"defects":[{"type":"defect","box":[0.2,0.2,0.4,0.35],"conf":0.9},'
            '{"type":"defect","box":[0.5,0.5,0.9,0.9,"bad"],"conf":0.3}]}\n```')
    data = parse_json(fake)
    assert "apples" in data and "defects" in data

    print("[自检] 坏 JSON 容错 ...")
    broken = ('```json\n{\n    "apples [{ "box": [0.5,0.7,0.6,0.8] }],\n'
              '    "defects": [{"box":[0.2,0.2,0.3,0.3]}] }\n```')
    data2 = parse_json(broken)
    assert len(data2["apples"]) == 1 and len(data2["defects"]) == 1, data2
    assert data2["apples"][0]["box"] == [0.5, 0.7, 0.6, 0.8]

    print("[自检] 坐标换算/过滤 ...")
    classes = ["defect"]
    apples, defects = normalize_results(data, classes, [], 1000, 1000)
    assert len(apples) == 1
    assert len(defects) == 1, f"应过滤 1 个坏框，实际 {len(defects)}"
    cx, cy, w, h = apples[0]
    assert abs(cx - 0.5) < 1e-6 and abs(w - 0.8) < 1e-6

    print("[自检] 0~1000 坐标自动识别 ...")
    data3 = {"apples": [{"box": [100, 100, 900, 900]}],
             "defects": [{"type": "defect", "box": [200, 200, 400, 350], "conf": 0.9}]}
    apples3, defects3 = normalize_results(data3, classes, [], 1000, 1000, coord_scale=1000)
    assert abs(apples3[0][0] - 0.5) < 1e-6 and abs(apples3[0][2] - 0.8) < 1e-6
    assert abs(defects3[0][1] - 0.3) < 1e-6 and abs(defects3[0][4] - 0.15) < 1e-6

    print("[自检] 写标签 ...")
    tmp = SCRIPT_DIR / "_vlm_selftest"
    txt = save_yolo_txt(tmp / "labels", "sample", defects)
    content = txt.read_text(encoding="utf-8").strip()
    assert content == "0 0.3000 0.2750 0.2000 0.1500", content

    print("[自检] 画可视化 ...")
    img = np.full((200, 300, 3), 240, dtype=np.uint8)
    vis = draw_viz(img, apples, defects, (0, 200, 0), (0, 0, 255), classes)
    encode_write(vis, tmp / "viz" / "sample.jpg")
    assert (tmp / "viz" / "sample.jpg").exists()

    print("[自检] 清理临时文件 ...")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("[自检] 全部通过 ✔  脚本可正常使用")


if __name__ == "__main__":
    main()
