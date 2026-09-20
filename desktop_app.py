# -*- coding: utf-8 -*-
"""
ZhiGuo AI QC System - desktop launcher.
Run `desktop.bat` or `python desktop_app.py` to open the system
in a standalone desktop window (no browser tabs / address bar).
The web server still runs locally, so phone access over LAN keeps working.
If the desktop window engine fails to start, the launcher automatically
opens the default browser instead.
"""
import os
import sys
import time
import socket
import threading
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

import config  # noqa: E402

TITLE = "\u667a\u679c\u00b7AI \u54c1\u63a7\u7cfb\u7edf"
PORT = config.PORT
BASE_URL = "http://127.0.0.1:{}/".format(PORT)

_ready = threading.Event()
_managed_server = None


def _port_open(timeout=0.3):
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_port(timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        if _port_open():
            return True
        time.sleep(0.2)
    return False


SPLASH_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Loading</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e1a;color:#e8f6ff;font-family:'Segoe UI','Microsoft YaHei',sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;overflow:hidden}
.card{text-align:center}
.spinner{width:56px;height:56px;border:4px solid rgba(34,211,238,.2);border-top-color:#22d3ee;border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 24px}
@keyframes spin{to{transform:rotate(360deg)}}
h2{font-size:20px;font-weight:600;letter-spacing:1px}
p{margin-top:12px;color:#93a7bc;font-size:13px}
.dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:#22d3ee;margin:0 3px;animation:blink 1.2s infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,100%{opacity:.2}50%{opacity:1}}
</style>
</head>
<body>
<div class="card">
  <div class="spinner"></div>
  <h2>\u6b63\u5728\u52a0\u8f7dAI\u6a21\u578b<span class="dot"></span><span class="dot"></span><span class="dot"></span></h2>
  <p>\u9996\u6b21\u542f\u52a8\u7ea6 20~30 \u79d2\uff0c\u8bf7\u7a0d\u5019</p>
</div>
<script>
(function(){
  var tries=0;
  var timer=setInterval(function(){
    fetch('/__desktop_status',{cache:'no-store'}).then(function(r){return r.json();}).then(function(d){
      if(d.ready){clearInterval(timer);window.location.href='/';}
    }).catch(function(){});
    if(++tries>120){clearInterval(timer);window.location.href='/';}
  },500);
})();
</script>
</body>
</html>"""


def register_launcher_routes(flask_app):
    from flask import jsonify

    @flask_app.route("/__desktop_status")
    def _desktop_status():
        return jsonify(ready=_ready.is_set(), port=PORT)

    @flask_app.route("/__desktop_loading")
    def _desktop_loading():
        return SPLASH_HTML


def start_server(flask_app):
    from waitress import create_server
    server = create_server(flask_app, host=config.HOST, port=PORT, threads=8)
    thread = threading.Thread(target=server.run, daemon=True, name="waitress")
    thread.start()
    return server


def warmup_in_background(flask_app):
    def _run():
        try:
            from app import db, _migrate_db, warmup_models
            with flask_app.app_context():
                db.create_all()
                _migrate_db()
            print("[desktop] loading models, please wait (20-30s)...")
            warmup_models()
            print("[desktop] models ready")
        except Exception as exc:
            print("[desktop] warmup warning: {}".format(exc))
        finally:
            _ready.set()

    threading.Thread(target=_run, daemon=True, name="warmup").start()


def stop_server():
    global _managed_server
    if _managed_server is not None:
        try:
            _managed_server.close()
        except Exception:
            pass
        _managed_server = None
        print("[desktop] server stopped")


def _browser_fallback(url):
    import webbrowser
    webbrowser.open(url)
    print("[desktop] opened in default browser. Close this black window to stop the server.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        stop_server()


def main():
    from app import app as flask_app
    register_launcher_routes(flask_app)

    global _managed_server
    already_running = _port_open()
    if not already_running:
        _managed_server = start_server(flask_app)
        if not _wait_port():
            print("[desktop] ERROR: server did not start on port {}".format(PORT))
            sys.exit(1)
        print("[desktop] server ready at http://127.0.0.1:{}/".format(PORT))
        warmup_in_background(flask_app)
        url = BASE_URL + "__desktop_loading"
    else:
        print("[desktop] server already running, attaching to it...")
        url = BASE_URL

    try:
        import webview
    except Exception as exc:
        print("[desktop] desktop window lib unavailable ({}), using browser...".format(exc))
        _browser_fallback(url)
        return

    state = {"failed": False}
    window = webview.create_window(
        TITLE,
        url,
        width=1440,
        height=900,
        min_size=(1080, 680),
        background_color="#0a0e1a",
    )

    def _watch():
        def _wait_then_check():
            try:
                loaded = window.events.loaded.wait(30)
            except Exception:
                loaded = False
            if not loaded:
                state["failed"] = True
                print("[desktop] window engine failed to start, using browser instead...")
                try:
                    window.destroy()
                except Exception:
                    pass
        threading.Thread(target=_wait_then_check, daemon=True).start()

    webview.start(func=_watch)
    if state["failed"]:
        _browser_fallback(url)
        return

    print("[desktop] window closed")
    stop_server()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_server()
