# -*- coding: utf-8 -*-
"""Build new base.html - single sidebar nav, simplified topbar, unified components."""
import io, os

BASE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>智果 · AI品控系统</title>
<style>@font-face{font-family:'Inter';src:url('/static/vendor/fonts/InterVariable.woff2') format('woff2');font-weight:100 900;font-display:swap;}</style>
<link rel="stylesheet" href="/static/vendor/fontawesome/css/all.min.css">
<link href="/static/css/bootstrap.min.css" rel="stylesheet">
<link rel="manifest" href="/manifest.json">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<script src="/static/js/chart.umd.min.js"></script>
<style>
:root {
  --bg-dark: #0a0e1a;
  --bg-card: rgba(18, 28, 50, 0.85);
  --bg-sidebar: rgba(10, 16, 30, 0.95);
  --primary: #00d4ff;
  --primary-dim: rgba(0, 212, 255, 0.15);
  --accent: #00ff88;
  --accent-dim: rgba(0, 255, 136, 0.12);
  --warning: #ffc107;
  --danger: #ff4757;
  --text: #e8edf5;
  --text-dim: #8899bb;
  --border: rgba(0, 212, 255, 0.18);
  --glow: 0 0 20px rgba(0, 212, 255, 0.15);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Inter', -apple-system, 'Source Han Sans SC', 'Microsoft YaHei', sans-serif;
  background: var(--bg-dark);
  color: var(--text);
  overflow: hidden; height: 100vh;
}
.bg-grid {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: linear-gradient(rgba(0,212,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,255,0.04) 1px, transparent 1px);
  background-size: 40px 40px;
}
.bg-grid::after {
  content: ''; position: fixed; inset: 0;
  background: radial-gradient(ellipse at 20% 50%, rgba(0,212,255,0.06) 0%, transparent 60%),
              radial-gradient(ellipse at 80% 20%, rgba(0,255,136,0.04) 0%, transparent 50%);
  pointer-events: none;
}
.particles { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
.particles i {
  position: absolute; bottom: -10px; width: 3px; height: 3px; border-radius: 50%;
  background: var(--primary); opacity: 0.35;
  animation: rise 14s linear infinite;
}
.particles i:nth-child(1) { left: 8%; animation-duration: 14s; animation-delay: 0s; }
.particles i:nth-child(2) { left: 22%; width: 2px; height: 2px; animation-duration: 18s; animation-delay: 3s; opacity: 0.25; }
.particles i:nth-child(3) { left: 38%; animation-duration: 12s; animation-delay: 6s; }
.particles i:nth-child(4) { left: 55%; width: 2px; height: 2px; animation-duration: 20s; animation-delay: 2s; opacity: 0.2; }
.particles i:nth-child(5) { left: 71%; animation-duration: 16s; animation-delay: 8s; }
.particles i:nth-child(6) { left: 87%; width: 2px; height: 2px; animation-duration: 22s; animation-delay: 5s; opacity: 0.3; }
@keyframes rise {
  0% { transform: translateY(0); opacity: 0; }
  10% { opacity: 0.35; }
  90% { opacity: 0.2; }
  100% { transform: translateY(-105vh); opacity: 0; }
}
.content-wrapper { position: relative; z-index: 1; height: 100vh; display: flex; }
/* ===== 侧边栏（唯一主导航） ===== */
.sidebar {
  width: 224px; min-width: 224px;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border);
  backdrop-filter: blur(20px);
  display: flex; flex-direction: column;
}
.sidebar-logo {
  padding: 18px 20px 14px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 10px;
}
.sidebar-logo i { font-size: 30px; color: var(--primary); text-shadow: var(--glow); }
.sidebar-logo .logo-text {
  font-size: 16px; font-weight: 700; line-height: 1.35;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sidebar-nav { flex: 1; padding: 10px 12px 8px; display: flex; flex-direction: column; gap: 2px; overflow-y: auto; }
.nav-group { margin-top: 8px; }
.nav-group-title {
  padding: 3px 14px 5px; font-size: 10px; letter-spacing: 2px;
  color: rgba(136,153,187,0.55); font-weight: 600;
}
.sidebar-nav a {
  display: flex; align-items: center; gap: 10px;
  padding: 9px 14px; border-radius: 9px;
  color: var(--text-dim); text-decoration: none; font-size: 13px; font-weight: 500;
  transition: all 0.25s; position: relative;
}
.sidebar-nav a i { width: 18px; text-align: center; font-size: 14px; }
.sidebar-nav a:hover, .sidebar-nav a.active { background: var(--primary-dim); color: var(--primary); }
.sidebar-nav a.active::before {
  content: ''; position: absolute; left: -12px; top: 50%; transform: translateY(-50%);
  width: 3px; height: 18px; background: var(--primary); border-radius: 0 3px 3px 0;
  box-shadow: 0 0 10px var(--primary);
}
.sidebar-footer {
  padding: 12px 16px; border-top: 1px solid var(--border);
  font-size: 11px; color: var(--text-dim);
  display: flex; flex-direction: column; gap: 4px;
}
/* ===== 顶栏 ===== */
.main-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 24px; border-bottom: 1px solid var(--border);
  backdrop-filter: blur(12px); background: rgba(10, 14, 26, 0.7);
  gap: 10px; flex-wrap: wrap;
}
.category-switch { display: flex; gap: 6px; }
.cat-btn {
  padding: 5px 12px; border-radius: 8px;
  border: 1px solid var(--border); background: rgba(0,212,255,0.05);
  color: var(--text-dim); font-size: 12px; cursor: pointer;
  transition: all 0.25s;
}
.cat-btn:hover { border-color: var(--primary); color: var(--primary); }
.cat-btn.active { border-color: var(--primary); color: var(--primary); background: var(--primary-dim); box-shadow: 0 0 12px rgba(0,212,255,0.25); }
.topbar-right { display: flex; align-items: center; gap: 16px; font-size: 13px; color: var(--text-dim); }
.topbar-right .time { font-weight: 600; color: var(--text); font-family: 'Inter', monospace; }
.topbar-right .user-name { font-weight: 600; color: var(--text); max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.topbar-right .avatar {
  width: 30px; height: 30px; border-radius: 50%;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; color: var(--bg-dark); font-weight: 700;
}
.topbar-right a.logout { color: var(--text-dim); text-decoration: none; font-size: 14px; transition: all 0.3s; }
.topbar-right a.logout:hover { color: var(--danger); }
#menuBtn { display: none; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 8px; border: 1px solid var(--border); background: rgba(0,212,255,0.06); color: var(--text-dim); font-size: 15px; cursor: pointer; }
.page-content { flex: 1; overflow-y: auto; padding: 20px 24px; }
/* ===== 统一组件 ===== */
.glass-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 14px; backdrop-filter: blur(16px);
  box-shadow: var(--glow);
}
.card-title { display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--text-dim); font-weight: 500; margin-bottom: 10px; }
.card-title i { color: var(--primary); font-size: 13px; }
.btn-g, .btn-p, .btn-a, .btn-w, .btn-d {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 6px 14px; border-radius: 9px; font-size: 12px; font-weight: 500;
  border: 1px solid var(--border); background: rgba(0,212,255,0.05);
  color: var(--text-dim); cursor: pointer; transition: all 0.25s;
  white-space: nowrap; text-decoration: none; line-height: 1.4;
}
.btn-p { border-color: var(--primary); color: var(--primary); background: rgba(0,212,255,0.10); }
.btn-a { border-color: var(--accent); color: var(--accent); background: rgba(0,255,136,0.08); }
.btn-w { border-color: var(--warning); color: var(--warning); background: rgba(255,193,7,0.08); }
.btn-d { border-color: var(--danger); color: #ff6b81; background: rgba(255,71,87,0.08); }
.btn-g:hover, .btn-p:hover, .btn-a:hover, .btn-w:hover, .btn-d:hover { filter: brightness(1.25); box-shadow: 0 0 12px rgba(0,212,255,0.2); }
.btn-sm2 { padding: 4px 10px; font-size: 11px; border-radius: 7px; }
.dark-input, .dark-select {
  background: rgba(10,16,30,0.9); color: var(--text);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 10px; font-size: 12px; outline: none;
}
.dark-input:focus, .dark-select:focus { border-color: var(--primary); }
.hint-text { font-size: 11px; color: var(--text-dim); line-height: 1.6; }
.mono { font-family: 'Inter', Consolas, monospace; font-weight: 700; }
@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 8px rgba(0,212,255,0.3); }
  50% { box-shadow: 0 0 25px rgba(0,212,255,0.6); }
}
@keyframes flash-red {
  0%, 100% { border-color: var(--danger); background: rgba(255,71,87,0.1); }
  50% { border-color: transparent; background: transparent; }
}
.flash-defect { animation: flash-red 0.8s ease-in-out 3; }
.model-btn {
  width: 46px; height: 46px; border-radius: 50%; border: 2px solid var(--border);
  background: rgba(10,14,26,0.8); backdrop-filter: blur(8px);
  color: var(--text-dim); display: flex; align-items: center; justify-content: center;
  cursor: pointer; transition: all 0.4s; font-size: 10px; font-weight: 600;
}
.model-btn:hover { border-color: var(--primary); color: var(--primary); }
.model-btn.active { border-color: var(--primary); color: var(--primary); box-shadow: 0 0 20px rgba(0,212,255,0.4); animation: pulse-glow 2s infinite; }
.log-container { height: 180px; overflow-y: auto; font-size: 12px; }
.log-item { padding: 5px 8px; border-radius: 4px; margin-bottom: 2px; display: flex; justify-content: space-between; border-left: 3px solid var(--accent); background: rgba(0,255,136,0.04); }
.log-item.defect { border-left-color: var(--danger); background: rgba(255,71,87,0.06); }
.chart-box { position: relative; height: 140px; width: 100%; }
.chart-box-lg { position: relative; height: 180px; width: 100%; }
.metric-card { padding: 12px; border-radius: 10px; background: rgba(0,212,255,0.04); border: 1px solid rgba(0,212,255,0.1); text-align: center; }
.metric-card .value { font-size: 20px; font-weight: 700; font-family: 'Inter', monospace; color: var(--primary); }
.metric-card .label { font-size: 11px; color: var(--text-dim); margin-top: 4px; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.25); border-radius: 3px; }
::-webkit-scrollbar-track { background: transparent; }
/* ===== 移动端 ===== */
.mobile-bottom-nav {
  display: none; position: fixed; bottom: 0; left: 0; right: 0; z-index: 1000;
  background: var(--bg-sidebar); border-top: 1px solid var(--border);
  backdrop-filter: blur(20px); padding: 6px 8px;
  overflow-x: auto; -webkit-overflow-scrolling: touch; align-items: center;
}
.mobile-bottom-nav a {
  flex: 0 0 auto; text-align: center; padding: 5px 13px;
  color: var(--text-dim); text-decoration: none; font-size: 10px;
}
.mobile-bottom-nav a i { display: block; font-size: 17px; margin-bottom: 2px; }
.mobile-bottom-nav a.active { color: var(--primary); }
.mobile-bottom-nav a.active i { text-shadow: 0 0 8px var(--primary); }
.mobile-bottom-nav::-webkit-scrollbar { display: none; }
.drawer-overlay { display: none; position: fixed; inset: 0; z-index: 1190; background: rgba(0,0,0,0.55); }
@media (max-width: 768px) {
  .sidebar {
    position: fixed; top: 0; bottom: 0; left: 0; z-index: 1200;
    transform: translateX(-105%); transition: transform 0.3s ease; width: 240px; min-width: 240px;
  }
  .sidebar.open { transform: translateX(0); box-shadow: 20px 0 60px rgba(0,0,0,0.5); }
  .drawer-overlay.show { display: block; }
  #menuBtn { display: inline-flex; }
  .main-area { margin-left: 0; }
  .topbar { padding: 8px 12px; }
  .topbar-right .time { font-size: 11px; }
  .topbar-right .avatar { width: 28px; height: 28px; font-size: 12px; }
  .topbar-right .user-name { display: none; }
  .category-switch { order: 2; width: 100%; justify-content: center; }
  .mobile-bottom-nav { display: flex; }
  .content-wrapper { padding-bottom: 52px; }
  .model-btn { width: 40px; height: 40px; font-size: 9px; }
  .page-content { padding: 14px 12px; }
}
@media (max-width: 576px) {
  .topbar-right .weather { display: none; }
  .stream-area { min-height: 280px !important; }
}
</style>
</head>
<body>
<div class="bg-grid"></div>
<div class="particles"><i></i><i></i><i></i><i></i><i></i><i></i></div>
<div class="content-wrapper">
  <div class="sidebar" id="drawer">
    <div class="sidebar-logo"><i class="fas fa-apple-alt"></i><span class="logo-text">智果 · AI<br>品控系统</span></div>
    <div class="sidebar-nav">
      <a href="/" class="{{"active" if active_tab=="monitor" else ""}}"><i class="fas fa-camera"></i>实时监测</a>
      <div class="nav-group">
        <div class="nav-group-title">数据</div>
        <a href="/history" class="{{"active" if active_tab=="history" else ""}}"><i class="fas fa-history"></i>历史数据</a>
        <a href="/stats" class="{{"active" if active_tab=="stats" else ""}}"><i class="fas fa-chart-bar"></i>数据统计</a>
        <a href="/dataset" class="{{"active" if active_tab=="dataset" else ""}}"><i class="fas fa-database"></i>数据集</a>
        <a href="/annotate" class="{{"active" if active_tab=="annotate" else ""}}"><i class="fas fa-tags"></i>批量标注</a>
      </div>
      <div class="nav-group">
        <div class="nav-group-title">分析</div>
        <a href="/haccp" class="{{"active" if active_tab=="haccp" else ""}}"><i class="fas fa-clipboard-check"></i>HACCP品控</a>
        <a href="/sensory" class="{{"active" if active_tab=="sensory" else ""}}"><i class="fas fa-utensils"></i>感官数据</a>
        <a href="/enose" class="{{"active" if active_tab=="enose" else ""}}"><i class="fas fa-wind"></i>电子鼻</a>
        <a href="/significance" class="{{"active" if active_tab=="significance" else ""}}"><i class="fas fa-flask"></i>显著性分析</a>
      </div>
      <div class="nav-group">
        <div class="nav-group-title">系统</div>
        <a href="#"><i class="fas fa-cube"></i>模型管理</a>
        <a href="#"><i class="fas fa-cog"></i>系统设置</a>
      </div>
    </div>
    <div class="sidebar-footer">
      <div>v2.0 · AI品控系统</div>
      <div><span style="color:var(--accent);">●</span> 系统运行中</div>
    </div>
  </div>
  <div class="drawer-overlay" id="drawerOverlay" onclick="toggleDrawer()"></div>
  <div class="main-area">
    <div class="topbar">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <button id="menuBtn" onclick="toggleDrawer()"><i class="fas fa-bars"></i></button>
        <div class="category-switch">
          <button class="cat-btn active" data-cat="apple" onclick="switchCategory('apple')">🍎 苹果</button>
          <button class="cat-btn" data-cat="bread" onclick="switchCategory('bread')">🍞 面包</button>
          <button class="cat-btn" data-cat="beverage" onclick="switchCategory('beverage')">🥤 饮料</button>
          <button class="cat-btn" data-cat="auto" onclick="switchCategory('auto')">✨ 自动</button>
        </div>
      </div>
      <div class="topbar-right">
        <span class="weather"><i class="fas fa-cloud-sun"></i> 26&deg;C</span>
        <span class="time" id="headerTime">--:--:--</span>
        <span class="user-name" title="{{ current_user.email or '本地用户' }}"><i class="fas fa-user-circle"></i> {{ current_user.nickname or '本地用户' }}</span>
        <div class="avatar" title="{{ current_user.email or '本地用户' }}">{{ (current_user.nickname or '用')[0] }}</div>
        <a class="logout" href="/logout" title="退出登录"><i class="fas fa-sign-out-alt"></i></a>
      </div>
    </div>
    <div class="page-content">
      {% block content %}{% endblock %}
    </div>
  </div>
</div>
<script>
function switchCategory(cat) {
  localStorage.setItem('category', cat);
  document.querySelectorAll('.cat-btn').forEach(function(b) {
    var on = b.getAttribute('data-cat') === cat;
    b.classList.toggle('active', on);
    b.style.color = on ? 'var(--primary)' : 'var(--text-dim)';
    b.style.borderColor = on ? 'var(--primary)' : 'var(--border)';
  });
  var p = window.location.pathname;
  if (p.indexOf('/history') === 0) {
    window.location.href = '/history?category=' + cat;
  } else if (p.indexOf('/stats') === 0) {
    window.location.reload();
  } else if (p.indexOf('/dataset') === 0 && window.loadDataset) {
    window.loadDataset();
  }
}
(function() {
  var saved = localStorage.getItem('category') || 'apple';
  document.querySelectorAll('.cat-btn').forEach(function(b) {
    var on = b.getAttribute('data-cat') === saved;
    b.classList.toggle('active', on);
    b.style.color = on ? 'var(--primary)' : 'var(--text-dim)';
    b.style.borderColor = on ? 'var(--primary)' : 'var(--border)';
  });
})();
function toggleDrawer() {
  var sb = document.getElementById('drawer');
  var ov = document.getElementById('drawerOverlay');
  if (!sb) return;
  var open = sb.classList.contains('open');
  if (open) { sb.classList.remove('open'); ov.classList.remove('show'); }
  else { sb.classList.add('open'); ov.classList.add('show'); }
}
function updateClock() {
  var d = new Date();
  document.getElementById("headerTime").textContent = d.toLocaleTimeString("zh-CN", {hour12:false});
}
setInterval(updateClock, 1000);
updateClock();
// 登录失效（401）时自动跳转登录页
(function() {
  var origFetch = window.fetch;
  window.fetch = function() {
    return origFetch.apply(this, arguments).then(function(r) {
      if (r.status === 401 && window.location.pathname.indexOf('/login') !== 0) {
        window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname);
      }
      return r;
    });
  };
})();
</script>

<!-- 移动端底部导航 -->
<div class="mobile-bottom-nav">
  <a href="/" class="{{"active" if active_tab=="monitor" else ""}}"><i class="fas fa-camera"></i>实时</a>
  <a href="/history" class="{{"active" if active_tab=="history" else ""}}"><i class="fas fa-history"></i>历史</a>
  <a href="/stats" class="{{"active" if active_tab=="stats" else ""}}"><i class="fas fa-chart-bar"></i>统计</a>
  <a href="/dataset" class="{{"active" if active_tab=="dataset" else ""}}"><i class="fas fa-database"></i>数据</a>
  <a href="/annotate" class="{{"active" if active_tab=="annotate" else ""}}"><i class="fas fa-tags"></i>标注</a>
  <a href="/haccp" class="{{"active" if active_tab=="haccp" else ""}}"><i class="fas fa-clipboard-check"></i>品控</a>
  <a href="/sensory" class="{{"active" if active_tab=="sensory" else ""}}"><i class="fas fa-utensils"></i>感官</a>
  <a href="/enose" class="{{"active" if active_tab=="enose" else ""}}"><i class="fas fa-wind"></i>电子鼻</a>
  <a href="/significance" class="{{"active" if active_tab=="significance" else ""}}"><i class="fas fa-flask"></i>显著</a>
  <a href="#"><i class="fas fa-cog"></i>设置</a>
  <a href="/api/export/report"><i class="fas fa-download"></i>导出</a>
</div>
</body>
</html>
'''

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates', 'base.html')
with io.open(out, 'w', encoding='utf-8', newline='\n') as f:
    f.write(BASE)
print('base.html written:', os.path.getsize(out), 'bytes')
