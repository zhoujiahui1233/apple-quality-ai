# -*- coding: utf-8 -*-
"""
缺陷标注网页复核器
==================
逐张打开图片，把 VLM 自动生成的缺陷框草稿「过一遍」：
  - 点选框 -> 拖动移动 / 拉右下角缩放
  - Delete 或按钮删除选中框
  - 「加框」模式拖拽补漏检的缺陷
  - 自动保存 YOLO 标签，标记「已复核」

启动（在 E:\\apple 目录）：
  python review_tool.py
  然后浏览器打开 http://127.0.0.1:5060

默认读取：
  图片: E:\\apple\\defect_annotate\\images
  草稿: E:\\apple\\auto_label_output_qwen_eval\\labels
可用参数覆盖：
  --images <图片目录> --labels <标签目录> --port <端口>
"""
import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from flask import Flask, jsonify, request, send_file, Response

ROOT = Path(__file__).resolve().parent

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>缺陷标注复核器</title>
<style>
  :root { --bg:#101418; --panel:#171d24; --line:#2a333d; --fg:#dbe4ec; --muted:#8b98a5;
          --red:#ff5d5d; --green:#4ade80; --blue:#60a5fa; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Microsoft YaHei",system-ui,sans-serif; background:var(--bg); color:var(--fg); }
  #app { display:flex; height:100vh; }
  #main { flex:1; display:flex; flex-direction:column; min-width:0; }
  #stage { flex:1; position:relative; overflow:hidden; background:#0a0d10; min-height:60vh; }
  #stageMsg { position:absolute; left:16px; top:12px; z-index:5; color:#8b98a5;
              font-size:13px; background:rgba(10,13,16,.7); padding:4px 10px; border-radius:6px; }
  #stageMsg.err { color:#ff5d5d; }
  canvas { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
           max-width:100%; max-height:100%; cursor:crosshair; }
  #top { display:flex; align-items:center; gap:10px; padding:8px 14px; background:var(--panel);
         border-bottom:1px solid var(--line); flex-wrap:wrap; }
  button { background:#232b35; color:var(--fg); border:1px solid var(--line); padding:6px 12px;
           border-radius:6px; cursor:pointer; font-size:14px; }
  button:hover { background:#2d3742; }
  button.on { background:var(--blue); color:#0a0d10; border-color:var(--blue); }
  button.danger { color:var(--red); }
  .spacer { flex:1; }
  #status { color:var(--muted); font-size:13px; }
  #filename { font-size:14px; }
  #panel { width:320px; background:var(--panel); border-left:1px solid var(--line);
           display:flex; flex-direction:column; }
  #boxlist { flex:1; overflow:auto; padding:8px; }
  .boxitem { display:flex; justify-content:space-between; padding:6px 8px; margin-bottom:4px;
             border:1px solid var(--line); border-radius:6px; cursor:pointer; font-size:13px; }
  .boxitem.sel { border-color:var(--red); background:#2b1a1c; }
  .boxitem .del { color:var(--red); cursor:pointer; }
  #tips { padding:10px 12px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; line-height:1.8; }
  #counter { font-size:13px; color:var(--muted); }
  #reviewedTag { color:var(--green); font-weight:bold; }
  h3 { margin:10px 10px 4px; font-size:13px; color:var(--muted); font-weight:normal; }
</style>
</head>
<body>
<div id="app">
  <div id="main">
    <div id="top">
      <button id="prev">◀ 上一张</button>
      <button id="next">下一张 ▶</button>
      <span id="counter"></span>
      <span id="reviewedTag"></span>
      <span id="filename"></span>
      <div class="spacer"></div>
      <button id="btnAdd" title="点一下开启/关闭加框模式">加框模式: 关</button>
      <button id="btnDel" class="danger">删除选中</button>
      <button id="btnReset" title="放弃本张改动，重新读取">还原本张</button>
      <span id="status">已自动保存</span>
    </div>
    <div id="stage"><div id="stageMsg">加载中…</div><canvas id="cv"></canvas></div>
  </div>
  <div id="panel">
    <h3>本张缺陷框（点选后可拖动/缩放，Delete 删除）</h3>
    <div id="boxlist"></div>
    <div id="tips">
      拖动空白处：加框模式下手动补框<br>
      点框选中 → 拖动移动 / 拉右下角缩放手柄改大小<br>
      <b>Delete</b> 删除选中框 · 改动自动保存<br>
      右上角灰色圆点=整果级可疑框，注意判断
    </div>
  </div>
</div>
<script>
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
let files = [], idx = 0, boxes = [], sel = -1, addMode = false, img = null;
let drag = null, dirty = false;

function el(id){ return document.getElementById(id); }
function stageMsg(t, isErr){ const m = el('stageMsg'); m.textContent = t;
  m.className = isErr ? 'err' : ''; }

async function api(path, opts){
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function loadList(){
  try{
    stageMsg('正在读取图片列表…');
    const d = await api('/api/images');
    files = d.files;
    if (!files.length){ stageMsg('图片目录为空', true); return; }
    const saved = Number(new URLSearchParams(location.search).get('i') || 0);
    idx = Math.max(0, Math.min(saved, files.length - 1));
    await loadImage();
  }catch(err){
    stageMsg('读取失败: ' + err.message + ' —— 请刷新重试', true);
  }
}

async function loadImage(){
  try{
    const d = await api('/api/img/' + idx);
    boxes = d.boxes.map(b => ({...b}));
    sel = -1; dirty = false; addMode = false;
    el('btnAdd').classList.remove('on');
    el('btnAdd').textContent = '加框模式: 关';
    el('filename').textContent = '';
    drawStatus();
    stageMsg(`第 ${idx + 1} 张加载中…`);
    const im = new Image();
    im.onload = () => { img = im; draw();
      stageMsg(`${d.name} · ${im.naturalWidth}×${im.naturalHeight} · ${boxes.length} 个草稿框`); };
    im.onerror = () => stageMsg(`图片加载失败: ${d.name}`, true);
    im.src = '/api/imgfile/' + idx + '?v=' + Date.now();
  }catch(err){
    stageMsg('本张数据读取失败: ' + err.message, true);
  }
}

function boxPx(b){
  const s = canvasScale();
  return {
    x1: b.x * s.w, y1: b.y * s.h,
    x2: (b.x + b.w) * s.w, y2: (b.y + b.h) * s.h
  };
}

function canvasScale(){
  const iw = img ? img.naturalWidth : 1, ih = img ? img.naturalHeight : 1;
  const s = Math.min(cv.width / iw, cv.height / ih);
  return { w: s * iw, h: s * ih, k: s };
}

function draw(){
  if (!img) return;
  try{
    ctx.clearRect(0, 0, cv.width, cv.height);
    const cw = cv.width, ch = cv.height;
    const sc = canvasScale();
    const ox = (cw - sc.w) / 2, oy = (ch - sc.h) / 2;
    ctx.drawImage(img, ox, oy, sc.w, sc.h);
    ctx.save();
    ctx.translate(ox, oy);
    boxes.forEach((b, i) => {
      const p = boxPx(b);
      const isSel = i === sel;
      ctx.strokeStyle = isSel ? '#ffd700' : '#ff5d5d';
      ctx.lineWidth = isSel ? 3 : 2;
      ctx.strokeRect(p.x1, p.y1, p.x2 - p.x1, p.y2 - p.y1);
      if (isSel){
        ctx.fillStyle = '#ffd700';
        const hs = 8;
        ctx.fillRect(p.x2 - hs / 2, p.y2 - hs / 2, hs, hs);
      }
    });
    ctx.restore();
    renderList();
  }catch(err){
    stageMsg('绘制出错: ' + err.message, true);
  }
}

function renderList(){
  const L = el('boxlist');
  L.innerHTML = '';
  boxes.forEach((b, i) => {
    const d = document.createElement('div');
    d.className = 'boxitem' + (i === sel ? ' sel' : '');
    d.innerHTML = `<span>框 ${i + 1} · ${(b.w * 100).toFixed(1)}% × ${(b.h * 100).toFixed(1)}%</span><span class="del">✕</span>`;
    d.onclick = () => { sel = i; draw(); };
    d.querySelector('.del').onclick = (e) => { e.stopPropagation(); removeBox(i); };
    L.appendChild(d);
  });
}

function mousePos(e){
  const r = cv.getBoundingClientRect();
  const sc = canvasScale();
  const ox = (cv.width - sc.w) / 2, oy = (cv.height - sc.h) / 2;
  const px = e.clientX - r.left - ox, py = e.clientY - r.top - oy;
  const bx = px / sc.w, by = py / sc.h;
  return { bx, by, px, py };
}

function norm(bx, by){ return { x: Math.max(0, Math.min(1, bx)), y: Math.max(0, Math.min(1, by)) }; }

function hitTest(bx, by){
  for (let i = boxes.length - 1; i >= 0; i--){
    const b = boxes[i];
    if (bx >= b.x && bx <= b.x + b.w && by >= b.y && by <= b.y + b.h) return i;
  }
  return -1;
}

cv.addEventListener('mousedown', e => {
  const m = mousePos(e);
  if (addMode){
    drag = { type: 'new', x0: m.bx, y0: m.by };
    return;
  }
  // 缩放手柄
  if (sel >= 0){
    const b = boxes[sel], p = boxPx(b);
    if (Math.abs(m.px - p.x2) < 10 && Math.abs(m.py - p.y2) < 10){
      drag = { type: 'resize', i: sel };
      return;
    }
  }
  const i = hitTest(m.bx, m.by);
  sel = i;
  draw();
  if (i >= 0) drag = { type: 'move', i, dx: m.bx - boxes[i].x, dy: m.by - boxes[i].y };
});

cv.addEventListener('mousemove', e => {
  if (!drag) return;
  const m = mousePos(e);
  if (drag.type === 'new'){
    const n = norm(m.bx, m.by);
    const x = Math.min(drag.x0, n.x), y = Math.min(drag.y0, n.y);
    boxes.push({ x, y, w: Math.abs(n.x - drag.x0), h: Math.abs(n.y - drag.y0), c: 0 });
    drag.type = 'sizing';
    drag.i = boxes.length - 1;
  } else if (drag.type === 'sizing'){
    const b = boxes[drag.i];
    const x0 = drag.x0, y0 = drag.y0;
    const x1 = Math.max(0, Math.min(1, m.bx)), y1 = Math.max(0, Math.min(1, m.by));
    b.x = Math.min(x0, x1); b.y = Math.min(y0, y1);
    b.w = Math.abs(x1 - x0); b.h = Math.abs(y1 - y0);
    b.c = 0;
  } else if (drag.type === 'move'){
    const b = boxes[drag.i];
    b.x = Math.max(0, Math.min(1 - b.w, m.bx - drag.dx));
    b.y = Math.max(0, Math.min(1 - b.h, m.by - drag.dy));
  } else if (drag.type === 'resize'){
    const b = boxes[drag.i];
    b.w = Math.max(0.005, Math.min(1 - b.x, m.bx - b.x));
    b.h = Math.max(0.005, Math.min(1 - b.y, m.by - b.y));
  }
  markDirty();
  draw();
});

window.addEventListener('mouseup', () => { drag = null; });

function removeBox(i){
  boxes.splice(i, 1);
  if (sel === i) sel = -1;
  markDirty(); draw();
}

function markDirty(){
  dirty = true;
  el('status').textContent = '有改动，保存中…';
  clearTimeout(markDirty._t);
  markDirty._t = setTimeout(save, 600);
}

async function save(force){
  if (!force && !dirty) return;
  dirty = false;
  try{
    await api('/api/save/' + idx, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ boxes, reviewed: true })
    });
    el('status').textContent = '✓ 已保存';
  }catch(err){ el('status').textContent = '保存失败: ' + err.message; }
}

function drawStatus(){
  el('counter').textContent = `${idx + 1} / ${files.length}`;
  el('filename').textContent = files[idx];
  el('reviewedTag').textContent = '';
}

async function goto(ni){
  await save(true);
  idx = Math.max(0, Math.min(files.length - 1, ni));
  await loadImage();
}

el('prev').onclick = () => goto(idx - 1);
el('next').onclick = () => goto(idx + 1);
el('btnDel').onclick = () => { if (sel >= 0) removeBox(sel); };
el('btnReset').onclick = async () => { await loadImage(); };
el('btnAdd').onclick = () => {
  addMode = !addMode;
  el('btnAdd').classList.toggle('on', addMode);
  el('btnAdd').textContent = addMode ? '加框模式: 开' : '加框模式: 关';
};
document.addEventListener('keydown', e => {
  if (e.key === 'Delete' && sel >= 0){ removeBox(sel); }
  if (e.key === 'ArrowRight') goto(idx + 1);
  if (e.key === 'ArrowLeft') goto(idx - 1);
});

function resizeCanvas(){
  const st = el('stage');
  cv.width = Math.max(100, st.clientWidth);
  cv.height = Math.max(100, st.clientHeight);
  draw();
}
window.addEventListener('resize', resizeCanvas);

loadList().then(() => setTimeout(resizeCanvas, 60));
</script>
</body>
</html>
"""


def build_app(images_dir: Path, labels_dir: Path):
    app = Flask(__name__)

    def image_files():
        return sorted(p for p in images_dir.iterdir()
                      if p.suffix.lower() in IMG_EXTS)

    def safe_index(i):
        files = image_files()
        if not files:
            return None, None
        i = max(0, min(int(i), len(files) - 1))
        return files[i], i

    def read_boxes(path: Path):
        boxes = []
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                sp = line.split()
                if len(sp) == 5:
                    try:
                        c, cx, cy, w, h = (float(v) for v in sp)
                    except ValueError:
                        continue
                    if w > 0 and h > 0:
                        boxes.append({"c": int(c), "x": round(cx - w / 2, 4),
                                      "y": round(cy - h / 2, 4),
                                      "w": round(w, 4), "h": round(h, 4)})
        return boxes

    @app.get("/")
    def home():
        return Response(HTML, mimetype="text/html")

    @app.get("/api/images")
    def api_images():
        files = image_files()
        return jsonify({"files": [p.name for p in files]})

    @app.get("/api/img/<int:i>")
    def api_img(i):
        f, real = safe_index(i)
        if f is None:
            return jsonify({"error": "no images"}), 404
        lab = labels_dir / (f.stem + ".txt")
        return jsonify({"name": f.name, "boxes": read_boxes(lab)})

    @app.get("/api/imgfile/<int:i>")
    def api_imgfile(i):
        f, real = safe_index(i)
        if f is None:
            return "not found", 404
        return send_file(str(f), max_age=0, etag=False)

    @app.post("/api/save/<int:i>")
    def api_save(i):
        f, real = safe_index(i)
        if f is None:
            return jsonify({"error": "no images"}), 404
        data = request.get_json(force=True)
        raw_boxes = data.get("boxes", [])
        lines = []
        for b in raw_boxes:
            try:
                c = int(b.get("c", 0))
                cx = float(b["x"]) + float(b["w"]) / 2
                cy = float(b["y"]) + float(b["h"]) / 2
                w = float(b["w"]); h = float(b["h"])
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{c} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}")
        labels_dir.mkdir(parents=True, exist_ok=True)
        (labels_dir / (f.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        # 复核进度标记
        if data.get("reviewed"):
            mark = labels_dir / ".reviewed"
            mark.parent.mkdir(exist_ok=True)
            marks = set()
            if mark.exists():
                marks = {x for x in mark.read_text(encoding="utf-8").splitlines()}
            marks.add(f.stem)
            mark.write_text("\n".join(sorted(marks)), encoding="utf-8")
        return jsonify({"ok": True})

    @app.get("/api/stats")
    def api_stats():
        files = image_files()
        mark = labels_dir / ".reviewed"
        done = set()
        if mark.exists():
            done = {x for x in mark.read_text(encoding="utf-8").splitlines()}
        return jsonify({"total": len(files), "reviewed": len(done)})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "defect_annotate" / "images"))
    ap.add_argument("--labels", default=str(ROOT / "auto_label_output_qwen_eval" / "labels"))
    ap.add_argument("--port", type=int, default=5060)
    ap.add_argument("--host", default="0.0.0.0",
                    help="监听地址；0.0.0.0=本机+局域网都能访问，127.0.0.1=仅本机")
    args = ap.parse_args()
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    if not images_dir.is_dir():
        print(f"图片目录不存在: {images_dir}")
        sys.exit(2)
    labels_dir.mkdir(parents=True, exist_ok=True)
    app = build_app(images_dir, labels_dir)
    url = f"http://127.0.0.1:{args.port}"
    print(f"复核器已启动: {url}")
    print(f"图片: {images_dir}")
    print(f"标签: {labels_dir}")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
