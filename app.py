#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ski Pro AI — Streamlit Web App
运行方式：
    streamlit run app.py
    （可选）另开终端：cpolar http 8502  ← Z-Pay 回调端口
"""

import os
import sys
import uuid
import time
import json
import hashlib
import shutil
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs
from datetime import datetime
from pathlib import Path

_CACHE = os.path.join(os.path.dirname(__file__), ".matplotlib-cache")
os.makedirs(_CACHE, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _CACHE)

# ════════════════════════════════════════════════════════════════════════════════
# Modal 后端 API 基础地址（web_api，不含 /analyze、/status 等路径）
# 可通过环境变量 MODAL_API_URL 或 st.secrets["MODAL_API_URL"] 覆盖。
# ════════════════════════════════════════════════════════════════════════════════
MODAL_API_URL = os.environ.get(
    "MODAL_API_URL",
    "https://henryhed--ski-pro-ai-api-web-api.modal.run",
)

import numpy as np
import requests
import pandas as pd
import base64
import io
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))


# ═══════════════════════════════════════════════════════════════════════════════
# 内嵌 Z-Pay 回调 HTTP 服务（监听 8502，后台线程）
# ═══════════════════════════════════════════════════════════════════════════════
_NOTIFY_PORT      = 8502
_BASE_DIR         = Path(__file__).parent
_PAYMENT_STATUS_F = _BASE_DIR / "payment_status.json"
_ORDERS_F         = _BASE_DIR / "orders.json"
_notify_lock      = threading.Lock()


def _zpay_verify(data: dict, key: str) -> bool:
    """ASCII 排序签名校验"""
    remote = str(data.get("sign", "")).strip().lower()
    if not remote:
        return False
    filtered = {k: str(v) for k, v in data.items()
                if k not in ("sign", "sign_type") and str(v) != ""}
    check_str = "&".join(f"{k}={v}" for k, v in sorted(filtered.items())) + key
    return hashlib.md5(check_str.encode("utf-8")).hexdigest() == remote


def _write_paid(order_id: str, trade_no: str, money: str) -> None:
    with _notify_lock:
        try:
            data = {}
            if _PAYMENT_STATUS_F.exists():
                data = json.loads(_PAYMENT_STATUS_F.read_text("utf-8"))
        except Exception:
            data = {}
        if not data.get(order_id, {}).get("paid"):
            data[order_id] = {
                "paid": True, "trade_no": trade_no, "money": money,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "notify",
            }
            _PAYMENT_STATUS_F.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), "utf-8"
            )


class _NotifyHandler(BaseHTTPRequestHandler):
    KEY = "i9IoCmuGX8fDIXp57Ke7tgVgKEzVxzEv"

    def log_message(self, fmt, *args):  # 静默日志
        print(f"[notify-http] {fmt % args}")

    def _handle(self, params: dict):
        print(f"[notify-http] 收到回调: {params}")
        if not _zpay_verify(params, self.KEY):
            print("[notify-http] 签名校验失败")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"fail")
            return

        if params.get("trade_status") == "TRADE_SUCCESS":
            order_id  = params.get("out_trade_no", "").strip()
            trade_no  = params.get("trade_no", "").strip()
            money     = params.get("money", "").strip()
            # 金额校验
            expected = None
            try:
                if _ORDERS_F.exists():
                    orders = json.loads(_ORDERS_F.read_text("utf-8"))
                    expected = orders.get(order_id, {}).get("amount")
            except Exception:
                pass
            if expected and abs(float(money) - float(expected)) > 0.01:
                print(f"[notify-http] 金额不匹配 notify={money} expected={expected}")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"fail")
                return
            _write_paid(order_id, trade_no, money)
            print(f"[notify-http] 支付成功写入: {order_id}")

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"success")

    def do_GET(self):
        parsed = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        self._handle(params)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length).decode("utf-8") if length else ""
        params = {k: v[0] for k, v in parse_qs(body).items()}
        if not params:
            parsed = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        self._handle(params)


def _start_notify_server():
    """在后台线程中启动 HTTP 回调服务，仅启动一次。"""
    try:
        server = HTTPServer(("0.0.0.0", _NOTIFY_PORT), _NotifyHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"[notify-http] 回调服务已启动，监听端口 {_NOTIFY_PORT}")
    except OSError:
        print(f"[notify-http] 端口 {_NOTIFY_PORT} 已被占用，跳过启动（notify_server 可能已在运行）")


# 仅在主进程中启动一次（防止 Streamlit 热重载时重复绑定端口）
if not os.environ.get("_NOTIFY_STARTED"):
    os.environ["_NOTIFY_STARTED"] = "1"
    _start_notify_server()


def _ensure_h264(video_path: Path) -> bool:
    """
    检测视频文件是否存在；云端版本不再在本地转码，仅作存在性检查。
    """
    return video_path.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Z-Pay 易支付服务类
# ═══════════════════════════════════════════════════════════════════════════════
class ZPayService:
    def __init__(self):
        self.pid         = "2026030109230189"
        self.key         = "i9IoCmuGX8fDIXp57Ke7tgVgKEzVxzEv"
        self.api_url     = "https://zpayz.cn/submit.php"
        # notify_url 与 return_url 默认指向最新的 Modal Streamlit 公网地址，
        # 在生产环境中也可以通过环境变量 ZPAY_NOTIFY_URL / ZPAY_RETURN_URL 覆盖。
        default_cb       = "https://henryhed--ski-pro-ai-streamlit-app.modal.run/"
        self.notify_url  = os.environ.get("ZPAY_NOTIFY_URL", default_cb)
        self.return_url  = os.environ.get("ZPAY_RETURN_URL", self.notify_url)
        self.price_yuan  = "0.01"
        self.price_label = "¥0.01"
        self.pay_type    = "wxpay"
        self.site_name   = "Ski Pro AI"

    def _build_sign_str(self, money: str, name: str, notify_url: str,
                        out_trade_no: str, pay_type: str,
                        return_url: str, site_name: str) -> str:
        """
        严格按照官方 demo 的固定字段顺序拼接签名字符串：
        money & name & notify_url & out_trade_no & pid & return_url & sitename & type
        """
        return (
            f"money={money}"
            f"&name={name}"
            f"&notify_url={notify_url}"
            f"&out_trade_no={out_trade_no}"
            f"&pid={self.pid}"
            f"&return_url={return_url}"
            f"&sitename={site_name}"
            f"&type={pay_type}"
        )

    def generate_pay_url(self, order_id: str, amount: str = None,
                         name: str = "Ski Pro AI 深度诊断报告",
                         pay_type: str = None) -> str:
        if amount is None:
            amount = self.price_yuan
        if pay_type is None:
            pay_type = self.pay_type
        money_str = f"{float(amount):.2f}"
        sg = self._build_sign_str(
            money=money_str, name=name,
            notify_url=self.notify_url, out_trade_no=order_id,
            pay_type=pay_type, return_url=self.return_url,
            site_name=self.site_name,
        )
        sign = hashlib.md5((sg + self.key).encode("utf-8")).hexdigest()
        return f"{self.api_url}?{sg}&sign={sign}&sign_type=MD5"

    def verify_notify(self, data: dict) -> bool:
        """
        校验 Z-Pay 异步回调签名。
        按官方 demo：去除 sign/sign_type 后，将剩余参数按 ASCII 升序排列拼接，末尾追加 key。
        """
        remote_sign = str(data.get("sign", "")).strip().lower()
        if not remote_sign:
            return False
        check_params = {k: str(v) for k, v in data.items()
                        if k not in ("sign", "sign_type") and str(v) != ""}
        sorted_pairs = sorted(check_params.items(), key=lambda x: x[0])
        check_str    = "&".join(f"{k}={v}" for k, v in sorted_pairs) + self.key
        local_sign   = hashlib.md5(check_str.encode("utf-8")).hexdigest()
        return local_sign == remote_sign


_zpay = ZPayService()

INPUT_DIR      = Path(__file__).parent / "input_videos"
DATABASE_CSV   = Path(__file__).parent / "database.csv"
PAYMENT_STATUS = Path(__file__).parent / "payment_status.json"
ORDERS_FILE    = Path(__file__).parent / "orders.json"
MAX_VIDEO_DURATION_SEC = 15


def _get_modal_base_url() -> str:
    """获取 Modal web_api 的 base URL（不含 /analyze、/status）。优先 secrets，其次环境变量，最后常量。"""
    base = ""
    try:
        base = (st.secrets.get("MODAL_API_URL") or "").strip()
    except Exception:
        pass
    if not base:
        base = (os.environ.get("MODAL_API_URL") or MODAL_API_URL or "").strip()
    return base.rstrip("/")


def call_modal_submit(video_bytes: bytes, filename: str) -> dict:
    """
    POST base_url + "/analyze" 用 multipart/form-data 上传视频，提交分析任务。
    返回响应 JSON，其中应包含 job_id。
    """
    base = _get_modal_base_url()
    if not base:
        raise RuntimeError("未配置 MODAL_API_URL")
    url = f"{base}/analyze"
    files = {"video": (filename or "video.mp4", video_bytes, "video/mp4")}
    resp = requests.post(url, files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()


def get_modal_status_once(base_url: str, job_id: str) -> dict:
    """
    单次请求 GET base_url + "/status/" + job_id，返回状态 JSON。
    进行中时含 progress_pct、stage；完成时含 status "done" 及完整 files 等。
    """
    url = f"{base_url.rstrip('/')}/status/{job_id}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json() or {}


def poll_modal_status(base_url: str, job_id: str, interval_sec: float = 2.0, timeout_sec: float = 1200) -> dict:
    """
    轮询 GET base_url + "/status/" + job_id，直到 status 为 "done" 或 "error"。
    返回最后一次响应的完整 JSON（即 meta，含 files 等）。
    """
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        meta = get_modal_status_once(base_url, job_id)
        status = (meta or {}).get("status")
        if status in ("done", "completed"):
            return meta
        if status == "error":
            return meta
        time.sleep(interval_sec)
    raise TimeoutError(f"轮询超时（{timeout_sec} 秒）未得到完成状态")


def _call_modal_analyze(video_bytes: bytes, filename: str = "video.mp4") -> dict:
    """
    提交分析任务到 Modal web_api，仅返回提交响应（含 job_id）。
    不在此处轮询；调用方需自行轮询 GET /status/{job_id}。
    """
    return call_modal_submit(video_bytes, filename)


# ═══════════════════════════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ski Pro AI · 滑雪诊断系统",
    page_icon="⛷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 全局样式
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #f5f5f7 !important;
    color: #1d1d1f !important;
    font-family: -apple-system, "SF Pro Display", "PingFang SC",
                 "Helvetica Neue", sans-serif !important;
}
.stApp {
    background: radial-gradient(circle at 75% 8%,
                rgba(255,255,255,0.92) 0%, #f5f5f7 60%) !important;
}
[data-testid="stHeader"] {
    background: rgba(245,245,247,0.82) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-bottom: 1px solid rgba(0,0,0,0.06) !important;
}
[data-testid="stSidebar"] { background: rgba(245,245,247,0.95) !important; }
.block-container { padding-top: 0 !important; max-width: 1180px; }

/* ── 毛玻璃卡片 ── */
.apple-card {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.4rem;
    border: 1px solid rgba(255,255,255,0.45);
    box-shadow: 0 8px 32px rgba(31,38,135,0.07);
}

/* ── 解锁卡片（渐变边框 + 金色点缀）── */
.unlock-card {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-radius: 22px;
    padding: 2rem 2rem 1.6rem;
    border: 1.5px solid rgba(0,113,227,0.18);
    box-shadow: 0 12px 40px rgba(0,113,227,0.10),
                inset 0 1px 0 rgba(255,255,255,0.9);
    position: relative;
    overflow: hidden;
}
.unlock-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0071e3 0%, #34aadc 50%, #5ac8fa 100%);
    border-radius: 22px 22px 0 0;
}

/* ── 锁定内容遮罩 ── */
.locked-preview {
    position: relative;
    border-radius: 16px;
    overflow: hidden;
}
.locked-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        to bottom,
        rgba(245,245,247,0) 0%,
        rgba(245,245,247,0.55) 40%,
        rgba(245,245,247,0.92) 75%,
        rgba(245,245,247,1) 100%
    );
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    padding-bottom: 1.8rem;
    border-radius: 16px;
}

/* ── section 小标签 ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #aeaeb2;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    display: block;
}

/* ── 步骤指示器 ── */
.step-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin: 0 auto 2.4rem;
    max-width: 440px;
}
.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.3rem;
    flex: 1;
}
.step-dot {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: #e5e5ea;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 600; color: #aeaeb2;
    transition: all 0.3s;
}
.step-dot.active  { background: #0071e3; color: #fff; box-shadow: 0 2px 10px rgba(0,113,227,0.35); }
.step-dot.done    { background: #34c759; color: #fff; }
.step-line { flex: 1; height: 2px; background: #e5e5ea; margin-bottom: 1.4rem; }
.step-line.done { background: #34c759; }
.step-label { font-size: 0.68rem; color: #aeaeb2; font-weight: 500; letter-spacing: 0.04em; }
.step-label.active { color: #0071e3; font-weight: 600; }
.step-label.done   { color: #34c759; }

/* ── 动画 ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in { animation: fadeSlideUp 0.52s cubic-bezier(0.25,0.46,0.45,0.94) both; }

@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
.shimmer {
    background: linear-gradient(90deg,
        rgba(255,255,255,0) 0%, rgba(255,255,255,0.6) 50%, rgba(255,255,255,0) 100%);
    background-size: 400px 100%;
    animation: shimmer 1.6s infinite;
}

/* ── 主按钮 ── */
div.stButton > button {
    background: #0071e3 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 2px 8px rgba(0,113,227,0.28) !important;
    transition: background 0.2s, transform 0.15s, box-shadow 0.2s !important;
    width: 100% !important;
}
div.stButton > button:hover {
    background: #0077ed !important;
    transform: scale(1.02) !important;
    box-shadow: 0 8px 20px rgba(0,113,227,0.32) !important;
}
div.stButton > button:active { transform: scale(0.98) !important; }

/* ── 上传框 ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.5) !important;
    border: 2px dashed #d2d2d7 !important;
    border-radius: 18px !important;
    padding: 1rem !important;
}

/* ── 进度条 ── */
.stSpinner > div { border-top-color: #0071e3 !important; }
[data-testid="stProgressBar"] > div > div > div {
    background: #0071e3 !important;
    border-radius: 4px !important;
}

/* ── metric ── */
[data-testid="stMetricValue"] {
    color: #0071e3 !important;
    font-size: 1.9rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricLabel"] {
    color: #6e6e73 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.02em !important;
}

/* ── 下载按钮 ── */
div.stDownloadButton > button {
    background: #fff !important;
    color: #0071e3 !important;
    border: 1.5px solid #0071e3 !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    transition: background 0.2s, color 0.2s !important;
}
div.stDownloadButton > button:hover {
    background: #0071e3 !important;
    color: #fff !important;
}

/* ── 脉冲点 ── */
.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #0071e3;
    animation: pulse 1.4s ease-in-out infinite;
    margin-right: 6px;
    vertical-align: middle;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.3; transform:scale(0.6); }
}

/* ── pay-box ── */
.pay-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1.2rem;
    text-align: center;
}

/* ── 视频标签 ── */
.video-label {
    color: #0071e3;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.04em;
    margin-bottom: 0.5rem;
}

/* ── 输入框 ── */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.8) !important;
    border: 1px solid #d2d2d7 !important;
    border-radius: 10px !important;
    color: #1d1d1f !important;
    font-size: 0.95rem !important;
}

/* ── radio ── */
[data-testid="stRadio"] label { color: #1d1d1f !important; }

/* ── code ── */
code {
    background: rgba(0,0,0,0.05) !important;
    color: #0071e3 !important;
    border-radius: 5px;
    padding: 1px 6px;
    font-size: 0.88em;
}

/* ── alert ── */
[data-testid="stAlert"] { border-radius: 14px !important; }

hr {
    border: none !important;
    border-top: 1px solid rgba(0,0,0,0.08) !important;
    margin: 1.5rem 0 !important;
}

/* ── 功能亮点行 ── */
.feature-row {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(0,0,0,0.05);
    font-size: 0.9rem;
    color: #3a3a3c;
}
.feature-row:last-child { border-bottom: none; }
.feature-icon {
    font-size: 1.1rem;
    min-width: 24px;
    margin-top: 1px;
}

/* ── 荣誉勋章 ── */
.badge-wrap {
    display: flex;
    align-items: center;
    gap: 1.4rem;
    background: linear-gradient(135deg,
        rgba(0,113,227,0.08) 0%, rgba(52,170,220,0.06) 100%);
    border: 1.5px solid rgba(0,113,227,0.2);
    border-radius: 20px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.badge-wrap::after {
    content: "";
    position: absolute;
    top: -30px; right: -30px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: radial-gradient(circle,
        rgba(0,113,227,0.12) 0%, rgba(0,113,227,0) 70%);
}
.badge-seal {
    width: 72px; height: 72px;
    border-radius: 50%;
    background: linear-gradient(145deg, #0071e3 0%, #34aadc 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem;
    box-shadow: 0 4px 16px rgba(0,113,227,0.35);
    flex-shrink: 0;
}
.badge-text-main {
    font-size: 1.12rem;
    font-weight: 700;
    color: #1d1d1f;
    letter-spacing: -0.01em;
    line-height: 1.3;
}
.badge-text-sub {
    font-size: 0.8rem;
    color: #6e6e73;
    margin-top: 0.2rem;
    letter-spacing: 0.02em;
}
.badge-no {
    font-size: 0.72rem;
    font-family: "SF Mono", "Fira Code", monospace;
    color: #0071e3;
    background: rgba(0,113,227,0.08);
    border-radius: 6px;
    padding: 2px 8px;
    margin-top: 0.4rem;
    display: inline-block;
    letter-spacing: 0.06em;
}

/* ── 高斯模糊锁定预览 ── */
.blurred-preview {
    position: relative;
    border-radius: 16px;
    overflow: hidden;
    cursor: pointer;
}
.blurred-preview img {
    width: 100%;
    filter: blur(8px) brightness(0.85);
    transform: scale(1.04);
    transition: filter 0.3s;
    border-radius: 16px;
}
.blurred-lock {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    background: rgba(0,0,0,0.08);
    border-radius: 16px;
}
.lock-icon {
    font-size: 2.4rem;
    filter: drop-shadow(0 2px 6px rgba(0,0,0,0.25));
}
.lock-tip {
    font-size: 0.85rem;
    font-weight: 600;
    color: #fff;
    background: rgba(0,0,0,0.45);
    padding: 4px 14px;
    border-radius: 20px;
    backdrop-filter: blur(8px);
    letter-spacing: 0.02em;
}


}

/* ── AI 教练寄语卡片 ── */
.coach-quote {
    background: linear-gradient(135deg,
        rgba(0,113,227,0.07) 0%, rgba(90,200,250,0.06) 100%);
    border-left: 4px solid #0071e3;
    border-radius: 0 16px 16px 0;
    padding: 1.4rem 1.8rem;
    margin-top: 0.6rem;
    font-size: 1.05rem;
    color: #1d1d1f;
    line-height: 1.7;
    font-style: italic;
    letter-spacing: 0.01em;
}
.coach-quote-author {
    font-size: 0.78rem;
    color: #aeaeb2;
    margin-top: 0.8rem;
    font-style: normal;
    letter-spacing: 0.04em;
}

/* ── 双按钮行 ── */
.btn-row {
    display: flex;
    gap: 0.8rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        # pay_first → (pay) → upload → generating_preview → preview → final
        "stage":            "pay_first",
        "order_id":         None,
        "user_name":        "",
        "video_filename":   "",
        "video_bytes":      None,
        "pay_url":          None,
        "pay_type":         "wxpay",
        "preview_done":     False,   # 基础骨骼视频已生成
        "analysis_done":    False,   # 深度分析已完成
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════
def _make_order_id() -> str:
    ts   = datetime.now().strftime("%Y%m%d%H%M%S")
    rand = str(uuid.uuid4()).replace("-", "")[:6].upper()
    return f"SKI{ts}{rand}"


def _mark_paid_local(order_id: str, trade_no: str = "", money: str = "") -> None:
    """将支付结果写入本地 payment_status.json（主动查单成功后调用）。"""
    try:
        data = {}
        if PAYMENT_STATUS.exists():
            with open(PAYMENT_STATUS, "r", encoding="utf-8") as f:
                data = json.load(f)
        if not data.get(order_id, {}).get("paid"):
            data[order_id] = {
                "paid":     True,
                "trade_no": trade_no,
                "money":    money,
                "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source":   "api_query",
            }
            with open(PAYMENT_STATUS, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[pay] 写入本地状态失败: {e}")


def _query_order_from_zpay(order_id: str) -> bool:
    """
    调用 Z-Pay API 主动查询订单状态（不依赖 cpolar 回调的兜底方案）。
    返回 True 表示已支付成功。
    """
    try:
        import urllib.request
        url = (
            f"https://zpayz.cn/api.php?act=order"
            f"&pid={_zpay.pid}"
            f"&key={_zpay.key}"
            f"&out_trade_no={order_id}"
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": "SkiProAI/1.0"}
        )
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read().decode("utf-8"))
        print(f"[pay] Z-Pay 查单结果: {result}")
        if str(result.get("code", "")) in ("1", "success"):
            if str(result.get("status", "0")) == "1":
                _mark_paid_local(
                    order_id,
                    trade_no=str(result.get("trade_no", "")),
                    money=str(result.get("money", "")),
                )
                return True
        return False
    except Exception as e:
        print(f"[pay] Z-Pay API 查单异常: {e}")
        return False


def _check_payment_status(order_id: str) -> bool:
    """
    检查订单是否已支付。
    优先读本地缓存（由回调写入或上次主动查单写入），
    若本地无记录则主动调用 Z-Pay API 查询。
    """
    # 先查本地缓存
    if PAYMENT_STATUS.exists():
        try:
            with open(PAYMENT_STATUS, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get(order_id, {}).get("paid"):
                return True
        except Exception:
            pass
    # 本地没有则主动查单
    return _query_order_from_zpay(order_id)


def _save_order(order_id: str, amount: str) -> None:
    """将订单金额持久化，供 notify_server.py 做金额二次校验。"""
    try:
        data = {}
        if ORDERS_FILE.exists():
            with open(ORDERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception:
        data = {}
    data[order_id] = {"amount": amount, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(ORDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _save_to_database(record: dict):
    df_new = pd.DataFrame([record])
    if DATABASE_CSV.exists():
        df_new.to_csv(DATABASE_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df_new.to_csv(DATABASE_CSV, mode="w", header=True,  index=False, encoding="utf-8-sig")


def _get_analysis_stats() -> dict:
    """从 Modal 返回的 analysis_csv 中解码统计指标。"""
    try:
        files = st.session_state.get("modal_result", {}).get("files", {})
    except Exception:
        files = {}
    csv_b64 = files.get("analysis_csv")
    if not csv_b64:
        return {}
    try:
        csv_bytes = base64.b64decode(csv_b64)
        df = pd.read_csv(io.BytesIO(csv_bytes))
        return {
            "avg_similarity_score": round(float(df["similarity_score"].mean()), 2),
            "max_edge_angle":       round(float(df["edge_angle_deg"].max()), 1),
            "avg_knee_angle":       round(float(df["knee_angle_avg_deg"].mean()), 1),
            "avg_lean_angle":       round(float(df["lean_angle_deg"].mean()), 1),
        }
    except Exception:
        return {}


def _load_video_bytes_from_value(val: str | bytes | None):
    """
    从后端返回的字段中安全获取视频二进制：
    - 若为 URL（http/https），使用 requests.get 拉取二进制流
    - 若为 Base64 字符串，则尝试解码
    - 其他情况返回 None
    """
    if not val:
        return None
    # 已经是二进制，直接返回
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)
    if not isinstance(val, str):
        return None

    v = val.strip()
    try:
        if v.lower().startswith(("http://", "https://")):
            try:
                resp = requests.get(v, timeout=30)
                resp.raise_for_status()
                return resp.content
            except Exception as e:
                print(f"[video] 通过 URL 获取视频失败: {e}")
                return None
        # 默认按 Base64 解码
        return base64.b64decode(v)
    except Exception as e:
        print(f"[video] 解码视频数据失败: {e}")
        return None


def _get_video_duration_seconds(video_bytes: bytes) -> float | None:
    """
    通过临时文件 + OpenCV 估算视频时长（秒），失败时返回 None。
    """
    try:
        import tempfile
        import cv2
        import os as _os

        tmp_path = None
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        if tmp_path:
            try:
                _os.remove(tmp_path)
            except Exception:
                pass
        if fps > 0 and frame_count > 0:
            return float(frame_count / fps)
    except Exception:
        return None
    return None


def _save_uploaded_video(video_bytes: bytes, filename: str) -> Path:
    """
    旧版本地调试时用于把上传视频保存到 input_videos/ 的工具函数，
    目前云端版流程不再调用，仅作为本地离线开发时的备用工具保留。
    """
    INPUT_DIR.mkdir(exist_ok=True)
    for old in INPUT_DIR.glob("*"):
        if old.suffix.lower() in (".mp4", ".mov"):
            old.unlink(missing_ok=True)
    dest = INPUT_DIR / filename
    with open(dest, "wb") as f:
        f.write(video_bytes)
    return dest


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤指示器
# ═══════════════════════════════════════════════════════════════════════════════
_STAGE_ORDER = ["pay_first", "upload", "generating_preview", "preview", "final"]
_STAGE_LABEL = {
    "pay_first":          "支付",
    "upload":             "上传",
    "generating_preview": "提取",
    "preview":            "预览",
    "final":              "报告",
}

def _render_steps():
    current = st.session_state.stage
    try:
        cur_idx = _STAGE_ORDER.index(current)
    except ValueError:
        cur_idx = 0

    dots, lines = [], []
    for i, s in enumerate(_STAGE_ORDER):
        if i < cur_idx:
            dots.append(f'<div class="step-dot done">✓</div>'
                        f'<div class="step-label done">{_STAGE_LABEL[s]}</div>')
            if i < len(_STAGE_ORDER) - 1:
                lines.append('<div class="step-line done"></div>')
        elif i == cur_idx:
            dots.append(f'<div class="step-dot active">{i+1}</div>'
                        f'<div class="step-label active">{_STAGE_LABEL[s]}</div>')
            if i < len(_STAGE_ORDER) - 1:
                lines.append('<div class="step-line"></div>')
        else:
            dots.append(f'<div class="step-dot">{i+1}</div>'
                        f'<div class="step-label">{_STAGE_LABEL[s]}</div>')
            if i < len(_STAGE_ORDER) - 1:
                lines.append('<div class="step-line"></div>')

    items_html = ""
    for i, d in enumerate(dots):
        items_html += f'<div class="step-item">{d}</div>'
        if i < len(lines):
            items_html += lines[i]

    st.markdown(
        f'<div class="step-bar animate-in">{items_html}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Hero 顶部品牌区（所有阶段共用）
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="animate-in" style="text-align:center;padding:3rem 0 1.6rem">
  <div style="font-size:3rem;font-weight:700;letter-spacing:-0.045em;
              color:#1d1d1f;line-height:1.05">
    Ski Pro AI
  </div>
  <div style="font-size:1rem;color:#6e6e73;margin-top:0.5rem;
              font-weight:300;letter-spacing:0.01em">
    专业级滑雪姿态数字分析系统
  </div>
  <div style="font-size:0.82rem;color:#aeaeb2;margin-top:0.25rem;
              letter-spacing:0.02em">
    每一个转弯，都是一次数据的升华。
  </div>
</div>
""", unsafe_allow_html=True)

_render_steps()

# 首页 Hero 中央示例图卡片（展示骨骼视频 + 报告范例）
_hero_demo_candidates = [
    Path(__file__).parent / "assets:hero_demo_example.jpg",
    Path(__file__).parent / "hero_demo_example.jpg",
]
_hero_demo_path = next((p for p in _hero_demo_candidates if p.exists()), None)
if _hero_demo_path is not None:
    st.markdown('<div class="apple-card animate-in" '
                'style="max-width:960px;margin:0 auto 1.8rem;padding:0;overflow:hidden">',
                unsafe_allow_html=True)
    st.image(str(_hero_demo_path), use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Z-Pay GET 回调捕获（notify_url 与 return_url 均指向本页）
# Z-Pay 发送的是 GET 请求，参数附在 URL query string 上
# ═══════════════════════════════════════════════════════════════════════════════
_qp = st.query_params
if _qp.get("trade_status") == "TRADE_SUCCESS":
    _cb_data = dict(_qp)
    _cb_order = str(_cb_data.get("out_trade_no", "")).strip()
    if _cb_order and _zpay.verify_notify(_cb_data):
        _cb_money   = str(_cb_data.get("money", "")).strip()
        _cb_tradeno = str(_cb_data.get("trade_no", "")).strip()
        # 金额校验
        _cb_ok = True
        try:
            if ORDERS_FILE.exists():
                _cb_expected = json.loads(ORDERS_FILE.read_text("utf-8")).get(_cb_order, {}).get("amount")
                if _cb_expected and abs(float(_cb_money) - float(_cb_expected)) > 0.01:
                    _cb_ok = False
                    print(f"[zpay-cb] 金额不匹配 notify={_cb_money} expected={_cb_expected}")
        except Exception:
            pass
        if _cb_ok:
            _write_paid(_cb_order, _cb_tradeno, _cb_money)
            print(f"[zpay-cb] GET 回调写入成功: {_cb_order}")
            # 清空 query params，防止刷新重复处理
            st.query_params.clear()
            if st.session_state.get("order_id") == _cb_order:
                # 若已生成预览/分析完成，跳转报告页；否则为「先付后传」流程，跳转上传页
                if st.session_state.get("preview_done") or st.session_state.get("analysis_done"):
                    st.session_state.stage = "final"
                else:
                    st.session_state.stage = "upload"
                st.rerun()
    else:
        print(f"[zpay-cb] 签名校验失败: {dict(_qp)}")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — 先支付后上传（感谢为中国算力支持）
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "pay_first":
    _, center_col, _ = st.columns([1, 1.8, 1])
    with center_col:
        st.markdown('<div class="apple-card animate-in">', unsafe_allow_html=True)
        # 标题 + 说明
        st.markdown(
            '<div style="font-size:1.5rem;font-weight:600;color:#1d1d1f;'
            'letter-spacing:-0.02em;margin-bottom:0.6rem">感谢为算力算法支持</div>'
            '<p style="color:#6e6e73;font-size:0.95rem;line-height:1.7;margin-bottom:1.2rem">'
            '您的支持将用于 GPU 算力与滑雪姿态分析算法迭代，支付完成后即可上传滑雪视频并获取完整专业诊断服务。</p>',
            unsafe_allow_html=True,
        )
        # 支付后将获得的内容预览
        st.markdown(
            """
<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0.9rem;margin-bottom:1.1rem">
  <div style="display:flex;gap:0.6rem;align-items:flex-start">
    <div style="width:28px;height:28px;border-radius:9px;background:rgba(0,113,227,0.08);
                display:flex;align-items:center;justify-content:center;font-size:1.1rem">🦴</div>
    <div>
      <div style="font-size:0.9rem;font-weight:600;color:#1d1d1f">骨骼数据</div>
      <div style="font-size:0.8rem;color:#6e6e73">逐帧姿态关键点与立刃角等核心曲线</div>
    </div>
  </div>
  <div style="display:flex;gap:0.6rem;align-items:flex-start">
    <div style="width:28px;height:28px;border-radius:9px;background:rgba(52,199,89,0.10);
                display:flex;align-items:center;justify-content:center;font-size:1.1rem">📄</div>
    <div>
      <div style="font-size:0.9rem;font-weight:600;color:#1d1d1f">专业报告</div>
      <div style="font-size:0.8rem;color:#6e6e73">一页式可视化战力雷达与关键指标解读</div>
    </div>
  </div>
  <div style="display:flex;gap:0.6rem;align-items:flex-start">
    <div style="width:28px;height:28px;border-radius:9px;background:rgba(255,159,10,0.10);
                display:flex;align-items:center;justify-content:center;font-size:1.1rem">🤖</div>
    <div>
      <div style="font-size:0.9rem;font-weight:600;color:#1d1d1f">AI 指导</div>
      <div style="font-size:0.8rem;color:#6e6e73">针对发力、立刃、重心等给出训练建议</div>
    </div>
  </div>
  <div style="display:flex;gap:0.6rem;align-items:flex-start">
    <div style="width:28px;height:28px;border-radius:9px;background:rgba(88,86,214,0.10);
                display:flex;align-items:center;justify-content:center;font-size:1.1rem">🎬</div>
    <div>
      <div style="font-size:0.9rem;font-weight:600;color:#1d1d1f">算法视频</div>
      <div style="font-size:0.8rem;color:#6e6e73">原片 vs 骨骼分屏对比 + 关键帧回放</div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        # 价格与按钮
        st.markdown(
            f'<div style="text-align:center;margin:0.6rem 0 1.0rem">'
            f'<span style="color:#0071e3;font-size:2rem;font-weight:700">{_zpay.price_label}</span>'
            f'<span style="color:#6e6e73;font-size:0.9rem;margin-left:0.3rem"> 开始使用</span></div>',
            unsafe_allow_html=True,
        )
        pay_start_btn = st.button(
            f"支持算力算法 {_zpay.price_label} 开始使用  💚",
            use_container_width=True,
            type="primary",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if pay_start_btn:
        order_id = _make_order_id()
        st.session_state.order_id = order_id
        st.session_state.pay_type = "wxpay"
        st.session_state.pay_url = _zpay.generate_pay_url(order_id=order_id, pay_type="wxpay")
        _save_order(order_id, _zpay.price_yuan)
        st.session_state.stage = "paying"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — 上传视频（支付完成后可见）
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "upload":

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # 标题单独渲染（纯文字，不产生空白块）
        st.markdown(
            '<span class="section-label">上传视频</span>'
            '<div style="font-size:1.5rem;font-weight:600;color:#1d1d1f;margin-bottom:0.25rem">'
            '上传您的滑雪视频</div>'
            '<p style="color:#6e6e73;font-size:0.88rem;margin-bottom:1rem">'
            '支持 MP4 / MOV 格式 · 建议时长 15 秒以内（越短分析越快）</p>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            label="拖拽或点击上传视频",
            type=["mp4", "MP4", "mov", "MOV"],
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        user_name = st.text_input(
            "昵称（用于报告署名）",
            placeholder="例如：张教练",
            value=st.session_state.user_name,
        )

    with col_right:
        st.markdown("""
<div class="apple-card animate-in">
  <span class="section-label">免费可见</span>
  <div style="font-size:1.4rem;font-weight:600;color:#1d1d1f;margin-bottom:0.8rem">
    AI 骨骼提取预览
  </div>
  <div style="color:#3a3a3c;font-size:0.9rem;line-height:1.9;margin-bottom:1rem">
    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem">
      <span>🦴</span>
      <span>AI滑行数据报告，免费生成预览视频</span>
    </div>
    <div style="display:flex;align-items:center;gap:0.5rem">
      <span>🔒</span>
      <span> AI专业滑行数据分析报告</span>
    </div>
  </div>
  <div style="background:rgba(0,113,227,0.06);border-radius:14px;
              padding:0.9rem 1.1rem;border:1px solid rgba(0,113,227,0.12)">
    <div style="font-size:0.82rem;font-weight:600;color:#0071e3;
                margin-bottom:0.5rem;letter-spacing:0.02em">解锁后获得</div>
    <div style="font-size:0.82rem;color:#3a3a3c;line-height:2">
      📡 &nbsp;5 维战力雷达图·综合评分可视化<br>
      🤖 &nbsp;AI 教练专业建议报告<br>
      🎬 &nbsp;原片 vs 骨骼标注左右对比视频<br>
      📐 &nbsp;立刃角·膝盖角·重心·相似度全指标
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 2])

    if "start_clicked" not in st.session_state:
        st.session_state.start_clicked = False

    # 未点击时显示按钮；点击且校验通过后直接跳转分析页，按钮因离开本页而消失
    with btn_col:
        if not st.session_state.start_clicked:
            start_btn = st.button(
                "开始免费检测  →",
                use_container_width=True,
            )
        else:
            start_btn = False
            st.caption("正在提交…")

    if start_btn and not st.session_state.start_clicked:
        if not uploaded:
            st.warning("请先上传滑雪视频！")
        elif not user_name.strip():
            st.warning("请填写您的昵称！")
        else:
            video_bytes = uploaded.read()
            duration_sec = _get_video_duration_seconds(video_bytes)
            if duration_sec is not None and duration_sec > MAX_VIDEO_DURATION_SEC:
                st.error(
                    f"视频时长约为 {duration_sec:.1f} 秒，已超过 {MAX_VIDEO_DURATION_SEC} 秒上限。"
                    "请截取 15 秒以内的精彩片段重新上传，以保证分析速度和稳定性。"
                )
            else:
                st.session_state.user_name      = user_name.strip()
                st.session_state.video_filename = uploaded.name
                st.session_state.video_bytes    = video_bytes
                st.session_state.start_clicked  = True
                st.session_state.stage          = "generating_preview"
                st.rerun()  # 一次 rerun 即进入分析页，上传页不再渲染，按钮自然消失


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — 生成骨骼预览（后台运行 process_videos）
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "generating_preview":

    _, mid_col, _ = st.columns([1, 4, 1])
    with mid_col:
        st.markdown('<div class="apple-card animate-in">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:1.6rem;font-weight:600;color:#1d1d1f;'
            'letter-spacing:-0.02em;margin-bottom:0.25rem">AI 正在提取骨骼框架…</div>'
            '<p style="color:#6e6e73;font-size:0.88rem;margin-bottom:1.2rem">'
            '骨骼识别 · 关键点标注 · 生成预览视频，请稍候（点击右上角三个点，选择「浮窗」即可保持当前页面）</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with mid_col:
        progress_bar = st.progress(0, text="准备中…")
        status_text  = st.empty()

    if not st.session_state.preview_done:
        progress_bar.progress(15, text="正在连接 Modal 云端…")
        status_text.markdown(
            '<p style="color:#6e6e73;font-size:0.88rem">GPU 正在全力分析中，请勿关闭页面…</p>',
            unsafe_allow_html=True,
        )

        # 第一步：POST /analyze 上传视频，获取 job_id
        try:
            submit_resp = _call_modal_analyze(
                st.session_state.video_bytes,
                st.session_state.get("video_filename") or "video.mp4",
            )
        except Exception as e:
            st.error(f"云端分析失败：{e}")
            st.info("请稍后重试，或联系客服。")
            if st.button("重新上传"):
                for k in ["stage", "preview_done", "analysis_done", "start_clicked",
                          "video_bytes", "video_filename", "modal_result", "job_id"]:
                    st.session_state.pop(k, None)
                st.rerun()
            st.stop()

        job_id = (submit_resp or {}).get("job_id")
        if not job_id:
            st.error(f"云端未返回 job_id，响应内容：{submit_resp}")
            if st.button("重新上传"):
                for k in ["stage", "preview_done", "analysis_done", "start_clicked",
                          "video_bytes", "video_filename", "modal_result", "job_id"]:
                    st.session_state.pop(k, None)
                st.rerun()
            st.stop()

        st.session_state["job_id"] = job_id

        # 第二步：轮询 /status/{job_id}，用返回的百分比更新进度条，完成后再取最终结果
        base_url = _get_modal_base_url()
        result = None
        deadline = time.time() + 1200
        try:
            while time.time() < deadline:
                meta = get_modal_status_once(base_url, job_id)
                status = (meta or {}).get("status")
                pct = (meta or {}).get("progress_pct", 0)
                stage = (meta or {}).get("stage", "分析中…")
                progress_bar.progress(min(100, max(0, pct)) / 100.0, text=stage)
                status_text.markdown(
                    f'<p style="color:#6e6e73;font-size:0.88rem">{stage}</p>',
                    unsafe_allow_html=True,
                )
                if status in ("done", "completed"):
                    result = meta
                    break
                if status == "error":
                    st.error(meta.get("error", "分析失败"))
                    if st.button("重新上传"):
                        for k in ["stage", "preview_done", "analysis_done", "start_clicked",
                                  "video_bytes", "video_filename", "modal_result", "job_id"]:
                            st.session_state.pop(k, None)
                        st.rerun()
                    st.stop()
                time.sleep(1.5)
            if result is None:
                raise TimeoutError("轮询超时（20 分钟）未得到完成状态")
        except Exception as e:
            st.error(f"云端分析轮询失败：{e}")
            st.info("请稍后重试，或联系客服。")
            if st.button("重新上传"):
                for k in ["stage", "preview_done", "analysis_done", "start_clicked",
                          "video_bytes", "video_filename", "modal_result", "job_id"]:
                    st.session_state.pop(k, None)
                st.rerun()
            st.stop()

        # 调试：在任何 Base64/URL 解码前先打印后端返回的文件键名
        try:
            res_files = (result or {}).get("files", {}) or {}
        except Exception:
            res_files = {}
        print(f"[modal] res_files.keys(): {list(res_files.keys())}")

        st.session_state["modal_result"] = result or {}
        if result.get("status") == "error":
            st.error(result.get("error", "分析失败"))
            if st.button("重新上传"):
                for k in ["stage", "preview_done", "analysis_done", "start_clicked",
                          "video_bytes", "video_filename", "modal_result", "job_id"]:
                    st.session_state.pop(k, None)
                st.rerun()
            st.stop()
        if result.get("status") not in ("done", "completed"):
            st.error("分析未完成，请稍后重试。")
            st.stop()

        # 若云端结果中未包含任何骨骼视频相关 key，则直接在生成阶段报错，避免后续空白预览。
        if "skeleton_video_mp4" not in res_files and "comparison_video_mp4" not in res_files:
            st.error(
                "云端分析已完成，但结果中未包含骨骼视频文件。"
                "请稍后重试，如多次出现请联系开发者排查后端。"
            )
            st.write(f"后端返回的文件键：{list(res_files.keys())}")
            st.stop()

        progress_bar.progress(100, text="云端分析完成")
        status_text.markdown(
            '<p style="color:#34c759;font-size:0.95rem;font-weight:600">'
            '云端分析完成，正在跳转预览…</p>',
            unsafe_allow_html=True,
        )
        st.session_state.preview_done  = True
        st.session_state.analysis_done = True
        time.sleep(0.8)
        st.session_state.stage = "preview"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — 预览与支付
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "preview":

    files = st.session_state.get("modal_result", {}).get("files", {}) or {}
    # 再次打印一次 keys，便于前端调试不同后端返回结构
    try:
        print(f"[preview] res_files.keys(): {list(files.keys())}")
    except Exception:
        pass

    # 若缺少骨骼视频相关 key，则在预览阶段直接提示错误，避免误以为分析成功。
    if "skeleton_video_mp4" not in files and "comparison_video_mp4" not in files:
        st.error(
            "云端结果中未包含骨骼视频，请返回上一页重新上传更清晰、时长更短的片段后重试。"
        )
        st.write(f"后端返回的文件键：{list(files.keys())}")
        st.stop()

    ski_b64        = files.get("ski_report_jpg")
    skel_video_val = files.get("skeleton_video_mp4")

    col_vid, col_pay = st.columns([3, 2])

    # ── 左侧：骨骼预览视频 + 模糊雷达图诱导
    with col_vid:
        st.markdown(
            '<span class="section-label">免费预览</span>'
            '<div style="font-size:1.3rem;font-weight:600;color:#1d1d1f;margin-bottom:0.6rem">'
            'AI 骨骼提取预览</div>',
            unsafe_allow_html=True,
        )
        # 优先播放云端生成的骨骼视频（支持 Base64 或 URL），其次回退到原始上传视频
        video_bytes = _load_video_bytes_from_value(skel_video_val)
        if not video_bytes:
            video_bytes = st.session_state.get("video_bytes")

        if video_bytes:
            st.video(video_bytes)
        else:
            st.info("暂时无法播放视频，请返回上一页重新上传。")

        st.markdown(
            '<p style="color:#aeaeb2;font-size:0.8rem;margin-top:0.8rem">'
            '以上为免费骨骼标注预览 · 解锁后可获得完整深度报告</p>',
            unsafe_allow_html=True,
        )

        # 权益卡片（进入 preview 阶段后才显示，带淡入动画）
        st.markdown("""
<div class="animate-in" style="margin-top:1rem;background:rgba(0,113,227,0.06);
     border-radius:16px;padding:1rem 1.2rem;border:1px solid rgba(0,113,227,0.14)">
  <div style="font-size:0.95rem;font-weight:700;color:#0071e3;margin-bottom:0.7rem;
              letter-spacing:-0.01em">
    🎁 预览已生成！支付解锁更多维度诊断
  </div>
  <div class="feature-row"><span class="feature-icon">📡</span>
    <div><strong style="color:#1d1d1f">5 维战力雷达图</strong>
      <span style="color:#6e6e73;font-size:0.8rem"> · 综合评分可视化</span></div>
  </div>
  <div class="feature-row"><span class="feature-icon">🤖</span>
    <div><strong style="color:#1d1d1f">AI 教练专业建议</strong>
      <span style="color:#6e6e73;font-size:0.8rem"> · 针对你的问题给出改进方案</span></div>
  </div>
  <div class="feature-row"><span class="feature-icon">🎬</span>
    <div><strong style="color:#1d1d1f">原片 vs 骨骼对比视频</strong>
      <span style="color:#6e6e73;font-size:0.8rem"> · 左右分屏直观对比</span></div>
  </div>
  <div class="feature-row"><span class="feature-icon">📐</span>
    <div><strong style="color:#1d1d1f">全维度指标数据</strong>
      <span style="color:#6e6e73;font-size:0.8rem"> · 立刃角·膝盖角·重心·相似度</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

        # 模糊预览：战力雷达图（激发好奇心）
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<span class="section-label">深度报告预览（已锁定）</span>',
                    unsafe_allow_html=True)
        if ski_b64:
            st.markdown(
                f'<div class="blurred-preview">'
                f'<img src="data:image/jpeg;base64,{ski_b64}" />'
                f'<div class="blurred-lock">'
                f'<div class="lock-icon">🔒</div>'
                f'<div class="lock-tip">解锁后查看完整战力报告</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="blurred-preview" style="height:180px;'
                'background:linear-gradient(135deg,#e5e5ea,#d2d2d7);'
                'display:flex;align-items:center;justify-content:center;'
                'flex-direction:column;gap:0.5rem;border-radius:16px">'
                '<div style="font-size:2.4rem">🔒</div>'
                '<div style="font-size:0.85rem;color:#6e6e73;font-weight:600">'
                '战力雷达图 · 解锁后可见</div>'
                '</div>',
                unsafe_allow_html=True,
            )
    # ── 右侧：已支付，进入完整报告
    with col_pay:
        st.markdown('<div class="unlock-card animate-in">', unsafe_allow_html=True)
        st.markdown(
            '<span class="section-label">深度诊断报告</span>'
            '<div style="font-size:1.5rem;font-weight:700;color:#1d1d1f;'
            'letter-spacing:-0.025em;margin-bottom:0.3rem">您已支持中国算力</div>'
            '<p style="color:#6e6e73;font-size:0.88rem;margin-bottom:1.2rem">'
            '分析已完成，点击下方查看完整诊断报告</p>',
            unsafe_allow_html=True,
        )
        st.markdown("""
<div style="display:flex;flex-direction:column;gap:0">
  <div class="feature-row">
    <span class="feature-icon">📡</span>
    <div><strong style="color:#1d1d1f">5 维战力雷达图</strong>
      <br><span style="color:#6e6e73;font-size:0.8rem">稳定性·立刃角·爆发力·重心·流畅度</span>
    </div>
  </div>
  <div class="feature-row">
    <span class="feature-icon">🤖</span>
    <div><strong style="color:#1d1d1f">AI 教练专业建议</strong>
      <br><span style="color:#6e6e73;font-size:0.8rem">针对你的滑行问题给出改进方案</span>
    </div>
  </div>
  <div class="feature-row">
    <span class="feature-icon">🎬</span>
    <div><strong style="color:#1d1d1f">原片 vs 骨骼对比视频</strong>
      <br><span style="color:#6e6e73;font-size:0.8rem">左右分屏，直观对比姿态偏差</span>
    </div>
  </div>
  <div class="feature-row">
    <span class="feature-icon">📐</span>
    <div><strong style="color:#1d1d1f">全维度指标数据</strong>
      <br><span style="color:#6e6e73;font-size:0.8rem">立刃角·膝盖角·重心·相似度</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        view_report_btn = st.button(
            "查看完整报告  →",
            use_container_width=True,
            type="primary",
        )
        if view_report_btn:
            st.session_state.stage = "final"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    _, back_col = st.columns([4, 1])
    with back_col:
        if st.button("← 重新上传", use_container_width=True):
            for k in ["preview_done", "analysis_done", "start_clicked",
                      "video_bytes", "video_filename", "modal_result", "job_id"]:
                st.session_state.pop(k, None)
            st.session_state.stage = "upload"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — 支付等待页
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "paying":
    order_id = st.session_state.order_id
    pay_url  = st.session_state.pay_url

    _, center_col, _ = st.columns([1, 1.8, 1])
    with center_col:
        st.markdown('<div class="apple-card animate-in pay-box">', unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:1.6rem;font-weight:600;color:#1d1d1f;'
            'letter-spacing:-0.02em">请完成支付</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="color:#6e6e73;font-size:0.88rem;margin:0.3rem 0 1.2rem">'
            f'订单号：<code>{order_id}</code>&nbsp;·&nbsp;'
            f'<span style="color:#0071e3;font-weight:600">{_zpay.price_label}</span></p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<a href="{pay_url}" target="_blank" style="'
            'display:inline-block;background:#0071e3;'
            'color:#fff;font-weight:500;font-size:1rem;padding:0.7rem 2.4rem;'
            'border-radius:12px;text-decoration:none;'
            'box-shadow:0 4px 14px rgba(0,113,227,0.3);'
            'letter-spacing:-0.01em;transition:background 0.2s">'
            '前往支付页面</a>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#aeaeb2;font-size:0.82rem;margin-top:0.6rem">'
            '在新标签页打开 · 支持微信 / 支付宝</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="margin-top:0.8rem;color:#6e6e73;font-size:0.88rem">'
            '<span class="pulse-dot"></span>等待支付确认，成功后自动跳转…</div>',
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    back_col, manual_col, _ = st.columns([1, 1, 3])
    with back_col:
        if st.session_state.get("preview_done") or st.session_state.get("analysis_done"):
            if st.button("← 返回预览"):
                st.session_state.stage = "preview"
                st.rerun()
        else:
            if st.button("← 返回"):
                st.session_state.stage = "pay_first"
                st.rerun()
    with manual_col:
        if st.button("✅ 我已完成支付"):
            with st.spinner("正在向 Z-Pay 查询订单状态…"):
                paid = _check_payment_status(order_id)
            if paid:
                if st.session_state.get("preview_done") or st.session_state.get("analysis_done"):
                    st.session_state.stage = "final"
                else:
                    st.session_state.stage = "upload"
                st.rerun()
            else:
                st.warning("暂未收到支付确认，请稍等几秒后再试。")

    # ── Python 端主动轮询（不用 JS reload，不丢失 session_state）
    # 每次 rerun 都向 Z-Pay API 查一次，查到已付款立即跳转
    _pay_checked = _check_payment_status(order_id)
    if _pay_checked:
        if st.session_state.get("preview_done") or st.session_state.get("analysis_done"):
            st.session_state.stage = "final"
        else:
            st.session_state.stage = "upload"
        st.rerun()
    else:
        # 等待 4 秒后自动 rerun，持续轮询，无需用户手动刷新
        time.sleep(4)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — 全量结果页（深度报告 · 仪式感满级）
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "final":

    stats = _get_analysis_stats()
    files = st.session_state.get("modal_result", {}).get("files", {}) or {}
    try:
        print(f"[final] res_files.keys(): {list(files.keys())}")
    except Exception:
        pass

    ski_b64       = files.get("ski_report_jpg")
    coach_b64     = files.get("coach_report_png")
    csv_b64       = files.get("analysis_csv")
    skel_val      = files.get("skeleton_video_mp4")
    cmp_val       = files.get("comparison_video_mp4")

    # 数据埋点（仅首次进入时写入）
    if not st.session_state.get("_db_saved"):
        record = {
            "upload_time":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_name":            st.session_state.user_name,
            "video_filename":       st.session_state.video_filename,
            "order_id":             st.session_state.get("order_id", "PAID"),
            "avg_similarity_score": stats.get("avg_similarity_score", ""),
            "max_edge_angle":       stats.get("max_edge_angle", ""),
        }
        _save_to_database(record)
        st.session_state["_db_saved"] = True

    user_name   = st.session_state.user_name
    report_no   = f"SKILAB-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"
    report_date = datetime.now().strftime("%Y 年 %m 月 %d 日")

    # ══════════════════════════════════════════════════
    # 荣誉勋章
    # ══════════════════════════════════════════════════
    st.markdown(
        f'<div class="badge-wrap animate-in">'
        f'<div class="badge-seal">⛷</div>'
        f'<div>'
        f'<div class="badge-text-main">Ski Pro AI 实验室特约测试员</div>'
        f'<div class="badge-text-sub">{user_name} · 专业滑雪姿态深度诊断报告 · {report_date}</div>'
        f'<div class="badge-no">报告编号：{report_no}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════
    # 第一行：Plotly 雷达图 + AI 教练图片报告
    # ══════════════════════════════════════════════════
    radar_col, coach_col = st.columns([1, 1])

    with radar_col:
        st.markdown(
            '<span class="section-label">5 维战力雷达图</span>'
            '<div style="font-size:1.1rem;font-weight:600;color:#1d1d1f;margin-bottom:0.8rem">'
            '综合战力评估</div>',
            unsafe_allow_html=True,
        )

        # 从数据推导雷达图各维度得分
        sim   = min(stats.get("avg_similarity_score", 70), 100) if stats else 70
        edge  = min(stats.get("max_edge_angle", 35) / 45 * 100, 100) if stats else 65
        knee  = min(100 - abs(stats.get("avg_knee_angle", 145) - 145) / 1.2, 100) if stats else 72
        lean  = min(100 - abs(stats.get("avg_lean_angle", 12) - 12) / 0.8, 100) if stats else 68
        smooth = round((sim * 0.4 + edge * 0.3 + knee * 0.3), 1)

        categories  = ["稳定性", "立刃角度", "爆发力", "重心控制", "动作流畅度"]
        values_user = [round(sim, 1), round(edge, 1), round(knee, 1),
                       round(lean, 1), round(smooth, 1)]
        values_pro  = [92, 88, 85, 90, 87]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_user + [values_user[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(0,113,227,0.18)",
            line=dict(color="#0071e3", width=2.5),
            name=user_name,
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
        ))
        fig.add_trace(go.Scatterpolar(
            r=values_pro + [values_pro[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(52,199,89,0.08)",
            line=dict(color="#34c759", width=1.5, dash="dot"),
            name="冠军参考",
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    tickfont=dict(size=10, color="#aeaeb2"),
                    gridcolor="rgba(0,0,0,0.08)",
                    linecolor="rgba(0,0,0,0.08)",
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color="#1d1d1f", family="PingFang SC"),
                    gridcolor="rgba(0,0,0,0.06)",
                    linecolor="rgba(0,0,0,0.1)",
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            margin=dict(t=30, b=30, l=60, r=60),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.15,
                xanchor="center", x=0.5,
                font=dict(size=11, color="#6e6e73"),
            ),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with coach_col:
        st.markdown('<span class="section-label">AI 教练建议</span>', unsafe_allow_html=True)
        if coach_b64:
            st.image("data:image/png;base64," + coach_b64, use_container_width=True)
        else:
            st.info("AI 教练报告生成中…")

    # ══════════════════════════════════════════════════
    # 第二行：核心指标卡
    # ══════════════════════════════════════════════════
    if stats:
        st.markdown('<span class="section-label">核心指标数据</span>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("相似度得分",   f"{stats.get('avg_similarity_score', 0):.1f}")
        m2.metric("最大立刃角",   f"{stats.get('max_edge_angle', 0):.1f}°")
        m3.metric("平均膝盖角",   f"{stats.get('avg_knee_angle', 0):.1f}°")
        m4.metric("平均身体倾斜", f"{stats.get('avg_lean_angle', 0):.1f}°")
        st.divider()

    # ══════════════════════════════════════════════════
    # 第三行：左右分屏对比视频
    # ══════════════════════════════════════════════════
    orig_bytes = st.session_state.get("video_bytes")
    skel_bytes = _load_video_bytes_from_value(skel_val)
    cmp_bytes  = _load_video_bytes_from_value(cmp_val)

    st.markdown(
        '<span class="section-label">左右分屏对比</span>'
        '<div style="font-size:1.1rem;font-weight:600;color:#1d1d1f;margin-bottom:1rem">'
        '原片 vs AI 骨骼标注</div>',
        unsafe_allow_html=True,
    )
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.markdown('<div class="video-label">原片回放</div>', unsafe_allow_html=True)
        if orig_bytes:
            st.video(orig_bytes)
        else:
            st.info("原始视频不可用，请返回首页重新上传。")
    with v_col2:
        st.markdown('<div class="video-label">AI 骨骼纠偏</div>', unsafe_allow_html=True)
        # 优先播放云端生成的对比视频，其次回退到骨骼视频；不再回退到原片
        display_bytes = cmp_bytes or skel_bytes
        if display_bytes:
            st.video(display_bytes)
        else:
            st.info("当前暂未生成骨骼对比视频，稍后可以重新尝试上传更清晰、时长更短的片段。")

    st.divider()

    # ══════════════════════════════════════════════════
    # 第四行：战力诊断报告图
    # ══════════════════════════════════════════════════
    if ski_b64:
        st.markdown('<span class="section-label">战力诊断报告</span>', unsafe_allow_html=True)
        st.image("data:image/jpeg;base64," + ski_b64, use_container_width=True)
        st.divider()

    # ══════════════════════════════════════════════════
    # 第五行：AI 教练寄语
    # ══════════════════════════════════════════════════
    def _gen_coach_quote(s: dict) -> str:
        if not s:
            return (
                "每一帧数据背后，都是你对滑雪的热爱与坚持。"
                "继续练习，你的姿态会越来越接近冠军水准。"
            )
        sim   = s.get("avg_similarity_score", 0)
        edge  = s.get("max_edge_angle", 0)
        knee  = s.get("avg_knee_angle", 145)

        if sim >= 85:
            opening = f"你的整体姿态相似度达到 {sim:.1f}，已超越 {min(int(sim), 97)}% 的雪友，"
        elif sim >= 70:
            opening = f"你的整体姿态相似度为 {sim:.1f}，正处于快速进步的黄金区间，"
        else:
            opening = f"你的姿态基础扎实，相似度得分 {sim:.1f} 有较大提升空间，"

        if edge >= 40:
            mid = f"入弯立刃角高达 {edge:.1f}°，展现出出色的激进风格。"
        elif edge >= 25:
            mid = f"入弯立刃角为 {edge:.1f}°，说明你已掌握弧线转弯的核心要领。"
        else:
            mid = f"入弯立刃角为 {edge:.1f}°，建议专项训练压刃技术以提升弯道速度。"

        if abs(knee - 145) <= 10:
            tail = "膝盖弯曲角度控制到位，重心稳定是你最大的优势，继续保持！"
        else:
            tail = "建议重点强化后程压板稳定性与膝盖缓冲训练，潜力巨大。"

        return opening + mid + tail

    quote = _gen_coach_quote(stats)

    st.markdown(
        f'<span class="section-label">AI 教练寄语</span>'
        f'<div class="coach-quote">'
        f'"{quote}"'
        f'<div class="coach-quote-author">— Ski Pro AI 教练系统 · {report_date}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ══════════════════════════════════════════════════
    # 下载区
    # ══════════════════════════════════════════════════
    st.markdown('<span class="section-label">下载报告</span>', unsafe_allow_html=True)
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        if ski_b64:
            st.download_button(
                label="战力报告 JPG",
                data=base64.b64decode(ski_b64),
                file_name=f"Ski_Report_{user_name}.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )
    with dl_col2:
        if coach_b64:
            st.download_button(
                label="教练报告 PNG",
                data=base64.b64decode(coach_b64),
                file_name=f"Ski_Coach_{user_name}.png",
                mime="image/png",
                use_container_width=True,
            )
    with dl_col3:
        if csv_b64:
            st.download_button(
                label="骨骼数据 CSV",
                data=base64.b64decode(csv_b64),
                file_name=f"Ski_Analysis_{user_name}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)

    # 重新分析按钮
    _, btn_center, _ = st.columns([1, 1, 1])
    with btn_center:
        if st.button("重新分析", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # 页脚
    st.markdown(
        '<div style="text-align:center;color:#aeaeb2;font-size:0.78rem;'
        'margin-top:3rem;padding-bottom:2rem;letter-spacing:0.03em">'
        'Ski Pro AI · Powered by MediaPipe &amp; Streamlit · 数据已加密保存'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
