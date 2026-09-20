# -*- coding: utf-8 -*-
"""产线模拟引擎：定时触发 → 图片池取图（多张=同一苹果多面）→ 检测融合 → 看板统计 + 分拣指令"""
import os
import random
import threading
import time

import config

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

GRADE_ORDER = {"Premium": 0, "特级": 0, "一级": 1, "Standard": 1,
               "二级": 2, "Reject": 3, "不合格": 3, "残次": 3}


def _is_reject(grade) -> bool:
    """等级串可能是组合串（如「不合格 Reject」），用子串匹配判定"""
    g = str(grade)
    return any(k in g for k in ("不合格", "Reject", "残次"))


class IndustrialEngine:
    def __init__(self, detection_service):
        self.svc = detection_service
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._stop_event = threading.Event()
        self._pool = []
        self._pool_loaded = False
        self._latencies = []
        self._consistent_count = 0
        self._logs = []
        self._stats = {
            "running": False,
            "speed": 2.0,
            "faces_per_apple": 3,
            "total": 0, "ok": 0, "reject": 0, "review": 0,
            "ok_ratio": 0.0,
            "avg_latency_ms": 0.0,
            "consistent_ratio": 0.0,
            "last_error": "",
            "session_seconds": 0,
            "warming": False,
        }

    # ---------- 图片池 ----------
    def _load_pool(self, limit_per_dir=200, max_total=400):
        """扫描图片池（限流，避免大数据集首次加载阻塞产线）"""
        dirs = [
            config.ROOT_DIR / "dataset_collected" / "apple",
            config.ROOT_DIR / "dataset_collected" / "unlabeled",
            config.ROOT_DIR / "yolo_detect_dataset" / "images" / "train",
            config.ROOT_DIR / "yolo_detect_dataset" / "images" / "val",
        ]
        pool = []
        for d in dirs:
            if not d.exists():
                continue
            try:
                with os.scandir(str(d)) as it:
                    entries = [e.path for e in it if e.is_file() and e.name.lower().endswith(_IMG_EXTS)]
            except Exception:
                continue
            random.shuffle(entries)
            pool.extend(entries[:limit_per_dir])
        pool = list(dict.fromkeys(pool))
        random.shuffle(pool)
        self._pool = pool[:max_total]
        self._pool_loaded = True
        return self._pool

    def _next_apple_images(self, n):
        if not self._pool_loaded or not self._pool:
            self._load_pool()
        if not self._pool:
            return []
        k = min(n, len(self._pool))
        return random.sample(self._pool, k)

    def _cleanup_core_files(self, core):
        """删除 _apple_detect_core 生成的可视化文件，避免模拟产线堆积垃圾"""
        try:
            name = core.get("unique_name", "")
            if name:
                for p in config.UPLOAD_DIR.glob(f"*_{name}.jpg"):
                    os.remove(str(p))
        except Exception:
            pass

    # ---------- 运行控制 ----------
    def start(self, speed=2.0, faces_per_apple=3):
        with self._lock:
            if self._running:
                return {"error": "产线已在运行"}
            self._running = True
            self._stop_event.clear()
            self._stats["speed"] = float(speed)
            self._stats["faces_per_apple"] = int(faces_per_apple)
            self._stats["session_start"] = time.time()
            self._stats["total"] = self._stats["ok"] = self._stats["reject"] = self._stats["review"] = 0
            self._stats["ok_ratio"] = 0.0
            self._stats["avg_latency_ms"] = 0.0
            self._stats["consistent_ratio"] = 0.0
            self._stats["last_error"] = ""
            self._stats["warming"] = True
            self._latencies = []
            self._consistent_count = 0
            self._logs = []
            # 同步预加载图片池，避免首轮检测阻塞
            if not self._pool:
                self._load_pool()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            return {"ok": True, "speed": self._stats["speed"]}

    def stop(self):
        with self._lock:
            was = self._running
            self._running = False
            self._stop_event.set()
            return {"ok": True, "was_running": was}

    def single_step(self, faces_per_apple=3):
        log = self._process_one(faces_per_apple)
        return log

    def _run_loop(self):
        while not self._stop_event.is_set():
            interval = 1.0 / max(0.5, self._stats["speed"])
            t0 = time.time()
            try:
                self._process_one(self._stats["faces_per_apple"])
            except Exception as e:  # noqa: BLE001
                self._stats["last_error"] = str(e)[:200]
            remain = interval - (time.time() - t0)
            if remain > 0:
                self._stop_event.wait(remain)

    # ---------- 单颗苹果处理 ----------
    def _process_one(self, faces):
        imgs = self._next_apple_images(faces)
        if not imgs:
            self._stats["last_error"] = "图片池为空，请先准备数据集"
            return None
        t0 = time.time()
        results = []
        grades = []
        ratios = []
        for path in imgs:
            uname = f"line_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            core = self.svc._apple_detect_core(path, uname, {})
            if "error" in core:
                continue
            results.append(core)
            grades.append(core["grade"])
            ratios.append(core["defect_ratio"])
            self._cleanup_core_files(core)
        if not results:
            self._stats["last_error"] = "所有面检测失败"
            return None
        worst = max(range(len(results)), key=lambda i: (GRADE_ORDER.get(str(grades[i]), 2), ratios[i]))
        fused_grade = grades[worst]
        fused_ratio = ratios[worst]
        is_reject = _is_reject(fused_grade)
        consistent = len(set(grades)) == 1
        latency = (time.time() - t0) * 1000

        with self._lock:
            self._stats["total"] += 1
            if is_reject:
                self._stats["reject"] += 1
            else:
                self._stats["ok"] += 1
            self._stats["ok_ratio"] = round(self._stats["ok"] / max(1, self._stats["total"]), 4)
            self._latencies.append(latency)
            if len(self._latencies) > 200:
                self._latencies.pop(0)
            self._stats["avg_latency_ms"] = round(sum(self._latencies) / len(self._latencies), 1)
            if consistent:
                self._consistent_count += 1
            self._stats["consistent_ratio"] = round(self._consistent_count / max(1, self._stats["total"]), 4)
            self._stats["session_seconds"] = int(time.time() - self._stats.get("session_start", time.time()))
            log = {
                "ts": time.strftime("%H:%M:%S"),
                "apple_id": f"A{self._stats['total']:04d}",
                "grade": str(fused_grade),
                "ratio": f"{fused_ratio * 100:.1f}%",
                "faces": len(results),
                "action": "REJECT" if is_reject else "OK",
                "latency_ms": round(latency, 1),
                "consistent": consistent,
            }
            self._logs.insert(0, log)
            if len(self._logs) > 40:
                self._logs.pop()
            self._stats["warming"] = False
        return log

    # ---------- 统计 ----------
    def stats(self):
        with self._lock:
            s = dict(self._stats)
            s["logs"] = list(self._logs[:30])
            s["running"] = bool(self._running)
            return s