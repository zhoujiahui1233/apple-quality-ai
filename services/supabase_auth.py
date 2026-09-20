"""Supabase 认证与用户数据存取（基于官方 REST API，仅依赖 requests）

功能：
- 邮箱密码注册 / 登录 / 登出 / 获取当前用户
- 用户资料表 profiles 的读取与写入（表不存在时自动忽略，不影响登录）
"""
import logging
import time

import requests

import config

logger = logging.getLogger("supabase")

TIMEOUT = 10

_ONLINE = None
_ONLINE_AT = 0.0


def is_configured() -> bool:
    """Supabase 是否已配置（配置后才会启用登录）"""
    return bool(config.SUPABASE_URL and config.SUPABASE_ANON_KEY)


def is_online() -> bool:
    """Supabase 是否真的连得上（探测 /auth/v1/health，15 秒缓存）。
    连不上（项目暂停/网络断）时返回 False，让系统可退回本地模式。"""
    global _ONLINE, _ONLINE_AT
    now = time.time()
    if _ONLINE is not None and now - _ONLINE_AT < 15:
        return _ONLINE
    _ONLINE_AT = now
    try:
        requests.get(f"{config.SUPABASE_URL}/auth/v1/health",
                     headers=_headers(), timeout=4)
        _ONLINE = True
    except Exception:
        _ONLINE = False
    return _ONLINE


def _headers(token=None):
    h = {
        "apikey": config.SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }
    if token:
        h["Authorization"] = "Bearer " + token
    return h


def _post(url, payload, token=None):
    try:
        resp = requests.post(url, json=payload, headers=_headers(token), timeout=TIMEOUT)
    except Exception as e:
        logger.warning("Supabase POST 失败 %s: %s", url, e)
        return 0, {"msg": "无法连接 Supabase，请检查网络或配置"}
    try:
        data = resp.json()
    except Exception:
        data = {"msg": resp.text[:200]}
    return resp.status_code, data


def _get(url, token=None):
    try:
        resp = requests.get(url, headers=_headers(token), timeout=TIMEOUT)
    except Exception as e:
        logger.warning("Supabase GET 失败 %s: %s", url, e)
        return 0, {}
    try:
        data = resp.json()
    except Exception:
        data = {}
    return resp.status_code, data


def _err_msg(data, fallback):
    """解析 Supabase / GoTrue 的错误信息"""
    if isinstance(data, dict):
        for k in ("msg", "message", "error_description", "error", "hint"):
            v = data.get(k)
            if v:
                return str(v)
    return fallback


def user_view(user):
    """把 Supabase user 对象转为前端可用的精简字典"""
    meta = user.get("user_metadata") or {}
    email = user.get("email") or ""
    return {
        "id": user.get("id"),
        "email": email,
        "nickname": meta.get("nickname") or email.split("@")[0] or "用户",
        "role": meta.get("role") or "user",
    }


def sign_up(email, password, nickname=""):
    """注册。ok=True 时 data 可能含 access_token（项目关闭邮箱验证时直接登录）"""
    payload = {"email": email, "password": password}
    if nickname:
        payload["data"] = {"nickname": nickname}
    code, data = _post(f"{config.SUPABASE_URL}/auth/v1/signup", payload)
    if code in (200, 201):
        return True, data
    return False, {"error": _err_msg(data, "注册失败")}


def sign_in(email, password):
    code, data = _post(
        f"{config.SUPABASE_URL}/auth/v1/token?grant_type=password",
        {"email": email, "password": password},
    )
    if code == 200 and data.get("access_token"):
        return True, data
    return False, {"error": _err_msg(data, "登录失败，请检查邮箱和密码")}


def get_user(token):
    code, data = _get(f"{config.SUPABASE_URL}/auth/v1/user", token)
    return (code == 200), data


def sign_out(token):
    if not token:
        return
    try:
        requests.post(
            f"{config.SUPABASE_URL}/auth/v1/logout",
            headers=_headers(token), timeout=TIMEOUT,
        )
    except Exception as e:
        logger.warning("登出 Supabase 失败: %s", e)


def get_profile(user_id):
    """读取 profiles 表（需先在 Supabase 执行建表 SQL）；表不存在时返回 (False, None)"""
    if not is_configured() or not user_id:
        return False, None
    try:
        r = requests.get(
            f"{config.SUPABASE_URL}/rest/v1/profiles",
            params={"id": "eq." + user_id, "select": "*", "limit": "1"},
            headers=_headers(), timeout=TIMEOUT,
        )
        if r.status_code == 200 and r.json():
            return True, r.json()[0]
    except Exception as e:
        logger.warning("读取用户资料失败: %s", e)
    return False, None


def upsert_profile(user_id, nickname, role="user"):
    """写入/更新 profiles 表（upsert）；表不存在时静默忽略"""
    if not is_configured() or not user_id:
        return False
    try:
        headers = _headers()
        headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
        r = requests.post(
            f"{config.SUPABASE_URL}/rest/v1/profiles",
            json={"id": user_id, "nickname": nickname, "role": role},
            headers=headers, timeout=TIMEOUT,
        )
        return r.status_code in (200, 201, 204)
    except Exception as e:
        logger.warning("写入用户资料失败: %s", e)
        return False
