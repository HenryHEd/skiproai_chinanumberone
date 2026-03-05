#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在终端中建议按如下步骤安装环境（任选一种方式即可）：

# 1. 创建并激活虚拟环境（推荐）
python3 -m venv venv
# macOS / Linux 激活：
source venv/bin/activate
# Windows 激活（CMD）：
venv\Scripts\activate

# 2. 安装依赖库
pip install opencv-python mediapipe pandas

# 3. 运行脚本（确保当前目录下有 input_videos 文件夹和 mp4 文件）
python main.py
"""

import os
import math
import json
from collections import deque
from glob import glob
from types import SimpleNamespace
from datetime import datetime
from multiprocessing import Process, cpu_count
import random

# 配置 Matplotlib 缓存目录，避免默认的 ~/.matplotlib 不可写导致导入缓慢
MATPLOTLIB_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".matplotlib-cache")
os.makedirs(MATPLOTLIB_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MATPLOTLIB_CACHE_DIR)

import subprocess
import shutil
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Modal 镜像与 App 必须放在全局作用域，供 modal run/deploy main.py 使用
import modal
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1", "libglib2.0-0")  # OpenCV/MediaPipe 必需
    .pip_install("mediapipe", "opencv-python-headless")
)
app = modal.App(name="ski-pro-ai-main", image=image)
nfs = modal.NetworkFileSystem.from_name("ski-pro-storage", create_if_missing=True)


def _reencode_h264(src_path: str) -> None:
    """
    将视频重新编码为浏览器可播放的 H.264 格式，覆盖原文件。
    优先级：
      1. imageio-ffmpeg（pip install imageio-ffmpeg，自带 ffmpeg 二进制，无需系统安装）
      2. 系统 ffmpeg（PATH 中可用时）
      3. cv2 avc1 编码器（macOS 自带，逐帧重写）
      4. 以上都不可用时静默跳过
    """
    tmp_path = src_path + ".h264tmp.mp4"

    # ── 方案1/2：ffmpeg 命令行（imageio-ffmpeg 提供的或系统的）
    ffmpeg_bin = None
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    if ffmpeg_bin is None:
        ffmpeg_bin = shutil.which("ffmpeg")

    if ffmpeg_bin:
        try:
            result = subprocess.run(
                [
                    ffmpeg_bin, "-y",
                    "-i", src_path,
                    "-c:v", "libx264",
                    "-preset", "ultrafast",   # 提速关键：极快转码
                    "-crf", "28",             # 略降画质，进一步加速并减小体积
                    "-an",                   # 滑雪分析视频无音频，跳过音频编码
                    "-movflags", "+faststart",
                    tmp_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600,
            )
            if result.returncode == 0:
                os.replace(tmp_path, src_path)
                print(f"[reencode] H.264 重编码完成（ffmpeg）：{src_path}")
                return
            else:
                err = result.stderr.decode(errors="ignore")[-300:]
                print(f"[reencode] ffmpeg 返回错误：{err}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception as exc:
            print(f"[reencode] ffmpeg 调用异常：{exc}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ── 方案3：cv2 逐帧重写（用 avc1 编码器，macOS 原生支持）
    print("[reencode] 尝试 cv2 avc1 编码器逐帧重写…")
    try:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            print(f"[reencode] 无法打开源文件：{src_path}")
            return
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            print("[reencode] avc1 编码器不可用，跳过重编码")
            cap.release()
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024:
            os.replace(tmp_path, src_path)
            print(f"[reencode] H.264 重编码完成（cv2 avc1）：{src_path}")
        else:
            print("[reencode] avc1 输出文件异常，保留原文件")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as exc:
        print(f"[reencode] cv2 重编码异常：{exc}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# MediaPipe PoseLandmarker 模型路径（相对当前脚本所在目录）
POSE_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "pose_landmarker_full.task"
)

# “世界冠军大弯卡宾”基准角度（可按需调整）
BASE_KNEE_ANGLE_DEG = 115.0
BASE_LEAN_ANGLE_DEG = 45.0
# 重心高度基准（经验值，可调整）
BASE_CENTER_HEIGHT_RATIO = 0.35

# 冠军示例姿态图片路径（右侧对比图）
CHAMPION_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "champion_pose.png")

# 双模系统目录
CHAMPION_VIDEO_DIR = os.path.join(os.path.dirname(__file__), "champion_videos")
CHAMPION_DATA_DIR  = os.path.join(os.path.dirname(__file__), "champion_data")

# UI 文本字体候选路径（优先 Linux/Modal 下 Noto CJK，其次 macOS）
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/Library/Fonts/Arial.ttf",
]


def get_ui_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_text_pil(frame, text, x, y, font_size=22, color=(0, 255, 0), bg_alpha=0.5):
    """
    使用 PIL 在 OpenCV 图像上绘制带半透明底框的文字，避免中文乱码。
    color 为 BGR 格式。
    """
    if text is None:
        return

    # BGR -> RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img, "RGBA")

    font = get_ui_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 半透明黑色底框
    bg_color = (0, 0, 0, int(bg_alpha * 255))
    draw.rectangle(
        (x, y, x + text_w + 10, y + text_h + 6),
        fill=bg_color,
    )

    # 文字颜色：BGR -> RGBA
    b, g, r = color
    draw.text(
        (x + 5, y + 3),
        text,
        font=font,
        fill=(r, g, b, 255),
    )

    # 回写到原始 frame（RGB -> BGR）
    frame[:, :, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ==========================
# 统一封装 MediaPipe 后端
# ==========================
try:
    # 优先尝试 Tasks V2 架构（mediapipe 0.10+）
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    MP_BACKEND = "tasks_v2"

    PoseLandmarkEnum = mp_vision.PoseLandmark
    POSE_CONNECTIONS = mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
    drawing_utils_module = mp_vision.drawing_utils
    DrawingSpec = drawing_utils_module.DrawingSpec

    # 提供一个与 V1 形状相似的 mp_pose，方便后续统一使用
    mp_pose = SimpleNamespace(
        PoseLandmark=PoseLandmarkEnum,
        POSE_CONNECTIONS=POSE_CONNECTIONS,
    )
except (ImportError, AttributeError):
    # 回退到老的 Solutions 架构（需要安装 mediapipe < 0.10）
    from mediapipe import solutions as mp_solutions

    MP_BACKEND = "solutions_v1"

    mp_pose = mp_solutions.pose
    PoseLandmarkEnum = mp_pose.PoseLandmark
    POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
    drawing_utils_module = mp_solutions.drawing_utils
    DrawingSpec = drawing_utils_module.DrawingSpec


# ==========================
# 一些通用的数学函数
# ==========================

def angle_between_three_points(a, b, c):
    """
    计算三点形成的夹角（以 b 为顶点），返回角度值（0-180 度）。

    参数:
        a, b, c: 坐标点 (x, y)，可以是像素坐标，也可以是归一化坐标

    例如：
        对于膝关节角度：a=髋关节，b=膝关节，c=脚踝
    """
    # 向量 BA = A - B, BC = C - B
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    # 计算向量的模长
    ba_len = math.hypot(ba[0], ba[1])
    bc_len = math.hypot(bc[0], bc[1])
    if ba_len == 0 or bc_len == 0:
        return None  # 无法计算角度

    # 计算夹角的余弦值：cos(theta) = (BA·BC) / (|BA|*|BC|)
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    cos_theta = dot_product / (ba_len * bc_len)

    # 数值稳定处理，避免浮点误差导致超出 [-1, 1]
    cos_theta = max(min(cos_theta, 1.0), -1.0)

    # 反余弦得到弧度，再转成角度
    theta_rad = math.acos(cos_theta)
    theta_deg = math.degrees(theta_rad)
    return theta_deg


def compute_lean_angle(shoulder_center, hip_center):
    """
    计算躯干倾斜角（相对于垂直方向的倾斜角，单位：度，0 表示身体完全直立）。

    参数:
        shoulder_center: 双肩中点 (x, y)
        hip_center:      双胯中点 (x, y)

    说明（图像坐标系）：
        - x 向右增大
        - y 向下增大
    算法思路：
        1. 先求“身体连线”与水平轴的夹角 angle_h（0~180 度）
        2. 再用 90 - angle_h 得到相对垂直方向的倾斜角
           - 人完全直立时：身体线接近竖直，angle_h ≈ 90 → 倾斜角 ≈ 0
           - 人向一侧大幅度倾斜时：angle_h 远离 90 → 倾斜角变大
    """
    dx = shoulder_center[0] - hip_center[0]
    dy = shoulder_center[1] - hip_center[1]

    # 若两点重合，则无法计算
    if dx == 0 and dy == 0:
        return None

    # 计算相对于水平轴的角度（0-180）
    angle_h = abs(math.degrees(math.atan2(dy, dx)))
    # 转换为相对于垂直方向的倾斜角（0 开始，越大表示身体越倾斜）
    lean_angle = abs(90.0 - angle_h)
    return lean_angle


def compute_center_height(hip_center, ankle_center, frame_height):
    """
    计算重心高度（胯部中点相对于脚踝中点的高度比例）。

    参数:
        hip_center:   胯部中点 (x, y)
        ankle_center: 脚踝中点 (x, y)
        frame_height: 图像的高度（像素）

    返回:
        height_ratio: 一个 0~1 左右的数值：
            - 0 附近：说明胯部几乎和脚踝在同一水平（非常蹲/压低）
            - 数值越大：说明胯部越高（站得越直）
    """
    if frame_height <= 0:
        return None

    # 图像坐标 y 向下增大，所以“高度差”用 ankle_y - hip_y
    delta_y = ankle_center[1] - hip_center[1]

    # 归一化到 [0, 1] 左右范围，便于不同分辨率之间对比
    height_ratio = delta_y / float(frame_height)
    return height_ratio


# ==========================
# 立刃角度计算 & 3D 向量工具
# ==========================

# 冠军基准立刃角度（Ted Ligety 大弯卡宾参考值）
BASE_EDGE_ANGLE_DEG = 60.0


def angle_3d(a, b, c):
    """
    3D 向量余弦定理角度：以 b 为顶点，计算 a-b-c 夹角（度）。
    a, b, c 为 (x, y, z) 元组。消除透视畸变。
    """
    ba = np.array([a[0]-b[0], a[1]-b[1], a[2]-b[2]], dtype=float)
    bc = np.array([c[0]-b[0], c[1]-b[1], c[2]-b[2]], dtype=float)
    n_ba = np.linalg.norm(ba)
    n_bc = np.linalg.norm(bc)
    if n_ba < 1e-9 or n_bc < 1e-9:
        return None
    cos_t = np.dot(ba, bc) / (n_ba * n_bc)
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    return math.degrees(math.acos(cos_t))


def compute_edge_angle_3d(knee_xyz, ankle_xyz):
    """
    3D 小腿向量与垂直轴 (0,1,0) 的夹角（tilt），范围 [0, 90]。
    最终立刃角由调用方统一做 90 - tilt 转换。
    """
    dx = ankle_xyz[0] - knee_xyz[0]
    dy = ankle_xyz[1] - knee_xyz[1]
    dz = ankle_xyz[2] - knee_xyz[2]
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 1e-9:
        return 0.0
    cos_t = abs(dy) / length
    cos_t = min(1.0, cos_t)
    tilt = math.degrees(math.acos(cos_t))
    return min(90.0, max(0.0, tilt))


def compute_edge_angle(knee_xy, ankle_xy):
    """2D 回退版：返回小腿与垂直线夹角（tilt），由调用方统一做 90 - tilt 转换。"""
    dx = ankle_xy[0] - knee_xy[0]
    dy = ankle_xy[1] - knee_xy[1]
    if dy == 0:
        return 0.0
    tilt = abs(math.degrees(math.atan2(abs(dx), abs(dy))))
    return min(90.0, max(0.0, tilt))


def _exp_smooth(history, alpha=0.3):
    """指数平滑，返回平滑后的列表（原地不修改）。"""
    if not history:
        return []
    out = [history[0]]
    for v in history[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def catmull_rom_chain(pts, steps=6):
    """
    Catmull-Rom 样条插值：在折线点列表之间插入中间点，生成光滑曲线。
    pts: [(x, y), ...] 像素坐标列表（至少 2 个点）
    steps: 每两个控制点之间插入的中间点数（越大越光滑）
    返回: [(x, y), ...] 密度更高的光滑点列表
    """
    if len(pts) < 2:
        return pts
    # 首尾各复制一次，保证端点切线方向自然
    p = [pts[0]] + list(pts) + [pts[-1]]
    result = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = (np.array(p[i-1], dtype=float),
                          np.array(p[i],   dtype=float),
                          np.array(p[i+1], dtype=float),
                          np.array(p[i+2], dtype=float))
        for s in range(steps):
            t = s / steps
            t2, t3 = t*t, t*t*t
            pt = 0.5 * ((2*p1)
                        + (-p0 + p2) * t
                        + (2*p0 - 5*p1 + 4*p2 - p3) * t2
                        + (-p0 + 3*p1 - 3*p2 + p3) * t3)
            result.append((int(pt[0]), int(pt[1])))
    result.append((int(pts[-1][0]), int(pts[-1][1])))
    return result


def draw_edge_curve(frame, angle_history, hip_history=None,
                    max_edge_frame_idx=None, max_edge_val=None,
                    max_flash_frames=0, fps=30, max_val=90,
                    champ_curve=None,
                    turn_dir_hist=None):
    """
    CARV 风格立刃角度曲线图（仿 image-41ab3af8 参考图）。
    - 深海蓝背景面板，Y 轴刻度，X 轴时间百分比
    - 左转：电光蓝  右转：荧光绿（分段着色）
    - 曲线下方填充半透明区域
    - 冠军基准虚线 + 标注
    - MAX 峰值点标注
    - PRO CARVING 高亮
    - MAX EDGE 定格大字
    """
    h, w = frame.shape[:2]

    # ── 绘图区域 ─────────────────────────────────────────────────────
    PLOT_W = int(w * 0.82)
    PLOT_H = int(h * 0.26)
    PAD_L = 52   # 左侧留给 Y 轴标签
    PAD_B = 24   # 底部留给 X 轴标签
    PAD_T = 32   # 顶部留给标题
    panel_x0 = (w - PLOT_W) // 2
    panel_y0 = h - PLOT_H - 10
    panel_x1 = panel_x0 + PLOT_W
    panel_y1 = panel_y0 + PLOT_H
    # 实际绘图区（轴内）
    ax0 = panel_x0 + PAD_L
    ax1 = panel_x1 - 8
    ay0 = panel_y0 + PAD_T
    ay1 = panel_y1 - PAD_B

    # ── 深海蓝背景 ───────────────────────────────────────────────────
    bg = frame.copy()
    cv2.rectangle(bg, (panel_x0, panel_y0), (panel_x1, panel_y1), (50, 38, 27), -1)
    cv2.addWeighted(bg, 0.82, frame, 0.18, 0, dst=frame)
    # 边框
    cv2.rectangle(frame, (panel_x0, panel_y0), (panel_x1, panel_y1),
                  (100, 80, 60), 1, cv2.LINE_AA)

    AX_W = ax1 - ax0
    AX_H = ay1 - ay0

    # ── Y 轴刻度线 & 标签 ────────────────────────────────────────────
    y_ticks = [0, 20, 40, 60, 80]
    for tick in y_ticks:
        if tick > max_val:
            continue
        ty = int(ay1 - (tick / max_val) * AX_H)
        # 浅色网格线
        cv2.line(frame, (ax0, ty), (ax1, ty), (80, 70, 55), 1, cv2.LINE_AA)
        # 0° 特别用白色实线
        if tick == 0:
            cv2.line(frame, (ax0, ty), (ax1, ty), (200, 200, 200), 1, cv2.LINE_AA)
        draw_text_pil(frame, str(tick), panel_x0 + 4, ty - 10,
                      font_size=14, color=(180, 180, 180), bg_alpha=0.0)

    # ── X 轴标签（0 / 33 / 67 / 100） ───────────────────────────────
    x_pcts = [0, 33, 67, 100]
    for pct in x_pcts:
        tx = int(ax0 + pct / 100 * AX_W)
        cv2.line(frame, (tx, ay0), (tx, ay1), (70, 60, 50), 1, cv2.LINE_AA)
        draw_text_pil(frame, str(pct), tx - 8, ay1 + 4,
                      font_size=13, color=(160, 160, 160), bg_alpha=0.0)

    # ── 冠军基准虚线 ─────────────────────────────────────────────────
    ref_y = int(ay1 - (BASE_EDGE_ANGLE_DEG / max_val) * AX_H)
    for ddx in range(ax0, ax1, 14):
        cv2.line(frame, (ddx, ref_y), (min(ddx + 8, ax1), ref_y),
                 (80, 180, 255), 1, cv2.LINE_AA)
    draw_text_pil(frame, f"Pro Edge {BASE_EDGE_ANGLE_DEG:.0f}°",
                  ax1 - 115, ref_y - 18, font_size=14,
                  color=(80, 180, 255), bg_alpha=0.55)

    # ── 窗口数据 ──────────────────────────────────────────────────────
    WIN = int(max(60, fps * 8))
    window_raw = angle_history[-WIN:] if angle_history else []
    # angle_history 存储的是 tilt（小腿与垂直轴夹角），曲线图需要展示 90 - tilt
    window_raw = [min(90.0, max(0.0, 90.0 - v)) for v in window_raw]
    window = _exp_smooth(window_raw, alpha=0.12)

    # ── 按转向分段着色 ────────────────────────────────────────────────
    # BGR 格式：左弯=电光蓝，右弯=荧光绿
    COLOR_LEFT  = (255, 122, 0)    # BGR: 蓝色 (0, 122, 255) 的 BGR 表示
    COLOR_RIGHT = (89, 199, 52)    # BGR: 绿色 (52, 199, 89) 的 BGR 表示
    COLOR_UNKN  = (140, 140, 140)  # BGR: 灰色（过渡帧）
    seg_colors = []
    WIN_len = len(window)

    if turn_dir_hist and len(turn_dir_hist) >= 1:
        # 取最近 WIN 帧的 turn_dir_history
        td_window = turn_dir_hist[-WIN_len:] if len(turn_dir_hist) >= WIN_len else turn_dir_hist
        for i in range(WIN_len):
            # 将 td_window 映射到 window 的索引
            td_i = min(int(i * len(td_window) / max(WIN_len, 1)), len(td_window) - 1)
            d = td_window[td_i]
            if d == -1:
                seg_colors.append(COLOR_LEFT)
            elif d == 1:
                seg_colors.append(COLOR_RIGHT)
            else:
                seg_colors.append(COLOR_UNKN)
    elif hip_history and len(hip_history) >= 4:
        for i in range(WIN_len):
            hi = min(int(i * len(hip_history) / max(WIN_len, 1)), len(hip_history) - 1)
            hi_prev = max(0, hi - 4)
            dx_h = hip_history[hi][0] - hip_history[hi_prev][0]
            if dx_h < -0.01:
                seg_colors.append(COLOR_LEFT)
            elif dx_h > 0.01:
                seg_colors.append(COLOR_RIGHT)
            else:
                seg_colors.append(COLOR_UNKN)
    else:
        seg_colors = [COLOR_UNKN] * WIN_len

    # 当前末尾转向（用于图例和 PRO CARVING 颜色，灰色帧往前找最近有效色）
    _last_valid = COLOR_RIGHT
    for _c in reversed(seg_colors):
        if _c != COLOR_UNKN:
            _last_valid = _c
            break
    CURVE_COLOR = _last_valid

    if len(window) >= 2:
        n = len(window)
        pts = []
        for i, ang in enumerate(window):
            px = int(ax0 + i * AX_W / max(n - 1, 1))
            py = int(ay1 - (min(ang, max_val) / max_val) * AX_H)
            pts.append((px, py))

        # ── 冠军基准曲线（灰色虚线，绘制在用户曲线下方）──────────────
        if champ_curve and len(champ_curve) >= 2:
            # 将冠军序列映射到同一绘图区域
            cn = len(champ_curve)
            champ_pts = []
            for ci, cang in enumerate(champ_curve):
                cpx = int(ax0 + ci * AX_W / max(cn - 1, 1))
                cpy = int(ay1 - (min(float(cang), max_val) / max_val) * AX_H)
                champ_pts.append((cpx, cpy))
            champ_smooth = catmull_rom_chain(champ_pts, steps=4)
            # 灰色虚线（每隔一段绘制）
            for ci in range(0, len(champ_smooth) - 1, 2):
                cv2.line(frame, champ_smooth[ci], champ_smooth[ci + 1],
                         (160, 160, 160), 2, cv2.LINE_AA)
            # 图例标注
            draw_text_pil(frame, "Champion",
                          ax0 + 4, ay0 + 2, font_size=13,
                          color=(160, 160, 160), bg_alpha=0.0)

        # ── 曲线下方填充（半透明） ────────────────────────────────────
        smooth_fill = catmull_rom_chain(pts, steps=4)
        if smooth_fill:
            fill_poly = [smooth_fill[0]] + smooth_fill + \
                        [(smooth_fill[-1][0], ay1), (smooth_fill[0][0], ay1)]
            fill_arr = np.array(fill_poly, dtype=np.int32)
            fill_overlay = frame.copy()
            cv2.fillPoly(fill_overlay, [fill_arr], CURVE_COLOR)
            cv2.addWeighted(fill_overlay, 0.13, frame, 0.87, 0, dst=frame)

        # ── PRO CARVING 区域额外高亮（>50°） ─────────────────────────
        pro_thresh_y = int(ay1 - (50.0 / max_val) * AX_H)
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            if p1[1] <= pro_thresh_y or p2[1] <= pro_thresh_y:
                poly = np.array([
                    [p1[0], max(p1[1], pro_thresh_y)],
                    [p1[0], p1[1]], [p2[0], p2[1]],
                    [p2[0], max(p2[1], pro_thresh_y)],
                ], dtype=np.int32)
                glow_fill = frame.copy()
                cv2.fillPoly(glow_fill, [poly], CURVE_COLOR)
                cv2.addWeighted(glow_fill, 0.28, frame, 0.72, 0, dst=frame)

        # ── 分段着色曲线（Catmull-Rom 样条） ─────────────────────────
        smooth = catmull_rom_chain(pts, steps=6)
        # 映射每个 smooth 点的颜色到对应原始段颜色
        total_s = max(len(smooth) - 1, 1)
        for i in range(len(smooth) - 1):
            seg_i = int(i * (len(pts) - 1) / total_s)
            seg_i = min(seg_i, len(seg_colors) - 1)
            c = seg_colors[seg_i]
            bright = tuple(min(255, ch + 80) for ch in c)
            cv2.line(frame, smooth[i], smooth[i + 1], bright, 1, cv2.LINE_AA)
            cv2.line(frame, smooth[i], smooth[i + 1], c, 3, cv2.LINE_AA)

        # 末尾高亮圆点
        cv2.circle(frame, pts[-1], 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pts[-1], 7, CURVE_COLOR, 1, cv2.LINE_AA)

        # ── 窗口最大点高亮圆圈（不显示数字标注）─────────────────────────
        win_max_idx = int(np.argmax(window_raw)) if window_raw else 0
        if 0 <= win_max_idx < len(pts):
            mx, my = pts[win_max_idx]
            bright_c = tuple(min(255, ch + 100) for ch in CURVE_COLOR)
            cv2.circle(frame, (mx, my), 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (mx, my), 10, bright_c, 2, cv2.LINE_AA)

    # ── 图例（右上角） ────────────────────────────────────────────────
    leg_x = ax1 - 210
    leg_y = panel_y0 + 6
    cv2.line(frame, (leg_x, leg_y + 8), (leg_x + 28, leg_y + 8), COLOR_LEFT, 2, cv2.LINE_AA)
    draw_text_pil(frame, "Left Turn", leg_x + 32, leg_y, font_size=13,
                  color=tuple(reversed(COLOR_LEFT)), bg_alpha=0.0)
    cv2.line(frame, (leg_x, leg_y + 22), (leg_x + 28, leg_y + 22), COLOR_RIGHT, 2, cv2.LINE_AA)
    draw_text_pil(frame, "Right Turn", leg_x + 32, leg_y + 14, font_size=13,
                  color=tuple(reversed(COLOR_RIGHT)), bg_alpha=0.0)

    # ── 标题 & Y轴标签 ────────────────────────────────────────────────
    draw_text_pil(frame, "Edge Angle °",
                  panel_x0 + 2, panel_y0 + 4, font_size=15,
                  color=(200, 210, 220), bg_alpha=0.0)
    # PRO CARVING 标签
    if window_raw and max(window_raw[-20:] if len(window_raw) >= 20 else window_raw) >= 50:
        draw_text_pil(frame, "★ PRO CARVING",
                      ax0 + 8, ay1 - 26, font_size=18,
                      color=tuple(reversed(CURVE_COLOR)), bg_alpha=0.0)

    # ── MAX EDGE 定格大字 ──────────────────────────────────────────────
    # 当前帧立刃角（取 angle_history 最新值）
    _cur_edge = angle_history[-1] if angle_history else 0.0
    if _cur_edge > 60.0:
        # 持续高亮：只要 > 60° 就常驻显示 "MAX EDGE ATTACK!"
        _atk_txt = "PRO!"
        _atk_tw = len(_atk_txt) * 30
        _atk_x = max(0, w // 2 - _atk_tw // 2)
        _atk_y = max(10, h // 6)
        glow_atk = frame.copy()
        cv2.rectangle(glow_atk, (_atk_x - 14, _atk_y - 8),
                      (_atk_x + _atk_tw + 14, _atk_y + 66),
                      (0, 140, 220), -1)
        cv2.addWeighted(glow_atk, 0.35, frame, 0.65, 0, dst=frame)
        draw_text_pil(frame, _atk_txt, _atk_x, _atk_y, font_size=52,
                      color=(0, 255, 255), bg_alpha=0.0)
        # 副标题：显示实际角度值
        _sub_txt = f"{_cur_edge:.0f}°"
        _sub_x = max(0, w // 2 - len(_sub_txt) * 16)
        draw_text_pil(frame, _sub_txt, _sub_x, _atk_y + 58, font_size=28,
                      color=(255, 255, 160), bg_alpha=0.0)
    elif max_edge_val is not None and max_edge_val > 0:
        # 稳定显示上一个转弯的锁定峰值（不再闪烁跳动）
        txt = f"EDGE MAX: {max_edge_val:.0f}°"
        tw_est = len(txt) * 20
        tx = max(0, w // 2 - tw_est // 2)
        ty = max(10, h // 6)
        draw_text_pil(frame, txt, tx, ty, font_size=36,
                      color=(200, 200, 100), bg_alpha=0.55)


def draw_glass_panel(frame, x, y, w, h, alpha=0.55, radius=24):
    """
    在 frame 上绘制毛玻璃圆角浮层（Glassmorphism）。
    纯 OpenCV 实现：半透明白色填充 + 蓝色边框。
    """
    overlay = frame.copy()
    # 圆角矩形填充（近似为多段线）
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), (240, 240, 250), -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), (240, 240, 250), -1)
    for cx, cy in [(x+radius, y+radius), (x+w-radius, y+radius),
                   (x+radius, y+h-radius), (x+w-radius, y+h-radius)]:
        cv2.circle(overlay, (cx, cy), radius, (240, 240, 250), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    # 边框
    cv2.rectangle(frame, (x + radius, y), (x + w - radius, y + h), (180, 180, 200), 1)
    cv2.rectangle(frame, (x, y + radius), (x + w, y + h - radius), (180, 180, 200), 1)


def overlay_champion_ghost(frame, champ_ref_img, hip_center, frame_width, frame_height,
                            alpha=0.30):
    """
    冠军影随（Ghosting）：将冠军参考图以 alpha 透明度对齐用户髋部叠加。
    叠加区域右侧 1/3，跟随 hip_center 垂直对齐。
    """
    if champ_ref_img is None or hip_center is None:
        return
    ov_w = frame_width // 3
    ov_h = int(ov_w * champ_ref_img.shape[0] / max(champ_ref_img.shape[1], 1))
    ov_h = min(ov_h, frame_height)
    try:
        small = cv2.resize(champ_ref_img, (ov_w, ov_h))
        ox = frame_width - ov_w - 4
        oy = max(0, min(frame_height - ov_h, int(hip_center[1]) - ov_h // 2))
        roi = frame[oy:oy + ov_h, ox:ox + ov_w]
        cv2.addWeighted(small, alpha, roi, 1 - alpha, 0, dst=roi)
    except Exception:
        pass


def draw_force_vector(frame, hip_center, ankle_center, edge_angle):
    """
    合力矢量线：从 COM（髋部）出发，根据立刃角度绘制侧向合力方向。
    - 垂直重力线（红色）
    - 合力矢量（黄色发光带箭头）
    - 若合力线通过双脚区域：'平衡良好'；偏离：'力线偏移，注意支撑'
    全部中文用 PIL 渲染。
    """
    if hip_center is None or ankle_center is None or edge_angle is None:
        return
    hx, hy = int(hip_center[0]), int(hip_center[1])
    ax, ay = int(ankle_center[0]), int(ankle_center[1])
    h, w = frame.shape[:2]

    # 垂直重力线（红色，已在主循环绘制，此处不重复）

    # 合力矢量方向：以竖直向下为基准，向侧偏转 edge_angle 度
    ea_rad = math.radians(min(edge_angle, 80))
    # 方向由髋部 x 相对脚踝判断偏向
    side = 1 if hx >= ax else -1
    vec_len = int((ay - hy) * 0.85)
    vx = int(hx + side * vec_len * math.sin(ea_rad))
    vy = int(hy + vec_len * math.cos(ea_rad))

    # 发光外描（黄色宽线）
    glow_f = frame.copy()
    cv2.line(glow_f, (hx, hy), (vx, vy), (0, 255, 255), 7)
    cv2.addWeighted(glow_f, 0.35, frame, 0.65, 0, dst=frame)
    # 主线
    cv2.line(frame, (hx, hy), (vx, vy), (0, 220, 255), 3, cv2.LINE_AA)
    # 箭头头部
    arrow_len = 14
    angle_arrow = math.atan2(vy - hy, vx - hx)
    for da in [0.45, -0.45]:
        ax2 = int(vx - arrow_len * math.cos(angle_arrow + da))
        ay2 = int(vy - arrow_len * math.sin(angle_arrow + da))
        cv2.line(frame, (vx, vy), (ax2, ay2), (0, 220, 255), 2, cv2.LINE_AA)

    # 判断合力线终点是否在脚踝附近（±帧宽 12%）
    tolerance = w * 0.12
    balanced = abs(vx - ax) <= tolerance
    status_txt = "平衡良好" if balanced else "力线偏移，注意支撑"
    s_color = (0, 255, 0) if balanced else (0, 140, 255)
    draw_text_pil(frame, status_txt, max(0, vx - 60), max(0, vy - 26),
                  font_size=20, color=s_color, bg_alpha=0.6)

    # 力线强度标注
    force_mag = edge_angle / 90.0 * 100
    draw_text_pil(frame, f"核心强度 {force_mag:.0f}%",
                  max(0, hx + 6), max(0, hy - 30),
                  font_size=18, color=(0, 220, 255), bg_alpha=0.55)


# ==========================
# 简单移动平均滤波
# ==========================

def moving_average_update(history, new_value, window_size):
    """
    对输入序列做滑动平均滤波，同时在遇到 None 时用上一帧值做插值。
    """
    if new_value is None:
        if not history:
            return None
        new_value = history[-1]

    history.append(new_value)
    if len(history) > window_size:
        del history[0]

    return sum(history) / len(history)



def _pil_draw_cn(pil_img, text, xy, font_size=28, color=(0, 255, 0),
                 bg=True, bg_alpha=160):
    """在 PIL 图像上绘制中文文字（带半透明底框）。color 为 RGB 格式。"""
    draw = ImageDraw.Draw(pil_img, "RGBA")
    font = get_ui_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = int(xy[0]), int(xy[1])
    if bg:
        draw.rectangle((x - 4, y - 4, x + tw + 8, y + th + 6),
                        fill=(0, 0, 0, bg_alpha))
    r, g, b = color
    draw.text((x, y), text, font=font, fill=(r, g, b, 255))


def get_pro_ski_advice(edge_angle, left_right_diff=0, variance=0):
    """
    根据最大立刃角度和滑行特征返回教练建议，温柔鼓励风格。
    返回 dict: status, status_color(BGR tuple), one_liner, task
    """
    if edge_angle is None:
        edge_angle = 0
    if edge_angle < 20:
        advice = {
            "status": "立刃启蒙",
            "status_color": (50, 50, 220),
            "one_liner": "先找找感觉哦～让脚踝轻轻往里倒，感受板刃碰触雪面的那一刻，很奇妙的！",
            "task": "小练习：平缓坡上单脚外刃滑行，找脚踝内倒的感觉",
        }
    elif edge_angle < 35:
        advice = {
            "status": "立刃养成",
            "status_color": (0, 165, 255),
            "one_liner": "已经有点感觉啦！让膝盖更主动地往弯心靠，想象用膝盖去'指路'，立刃会更自然～",
            "task": "小练习：转弯时让双膝同步往内倾，感受板刃咬雪的感觉",
        }
    elif edge_angle < 50:
        advice = {
            "status": "有效刻滑",
            "status_color": (50, 205, 50),
            "one_liner": "已经在carving了，太棒！可以试着让这个感觉维持更久一点，弯底再松开，更过瘾哦～",
            "task": "小练习：延迟换刃，体会压板到底再弹起的节奏感",
        }
    elif edge_angle < 65:
        advice = {
            "status": "精进刻滑",
            "status_color": (0, 200, 100),
            "one_liner": "角度很漂亮！现在试试入弯时更早地把重心前压，让雪板在弯顶就开始工作，会很爽～",
            "task": "小练习：入弯前主动前倾重心，感受弯顶咬雪的快感",
        }
    else:
        advice = {
            "status": "高阶刻滑",
            "status_color": (0, 200, 255),
            "one_liner": "已经很厉害啦！试试让重心更贴近雪面，在保持高立刃角的同时享受离心力的感觉～",
            "task": "小练习：尝试更低姿态极限压板，感受重心与雪板融为一体",
        }
    # 特殊问题叠加诊断（温柔版）
    if left_right_diff > 15:
        advice["one_liner"] = "左右弯差距有点大哦，非优势侧多练练，让两侧都有同等的力量感，平衡了会更好看的～"
    elif left_right_diff > 8:
        advice["one_liner"] = "左右弯稍微有点不对称，多花些时间练习弱侧弯，慢慢就能平衡了，加油哦～"
    elif variance > 12:
        advice["one_liner"] = "立刃角波动有点大，试着让每个弯都更均匀稳定，找到那种丝滑连贯的节奏感～"
    elif variance > 7:
        advice["one_liner"] = "每个弯的立刃角稍有起伏，专注于节奏的一致性，会越来越流畅的～"
    return advice


def create_comparison_video(
    input_video_path: str,
    processed_video_path: str,
    report_img_path: str,
    output_path: str,
) -> None:
    """
    将原始视频与骨骼分析视频逐帧拼接，底部附加专业诊断报告 PNG，输出对比视频。

    横版（width >= height）：原视频 / 骨骼视频 上下堆叠，底部追加报告图。
    竖版（height > width） ：原视频 | 骨骼视频 左右并排，底部追加报告图。
    """
    cap_in  = cv2.VideoCapture(input_video_path)
    cap_pro = cv2.VideoCapture(processed_video_path)

    if not cap_in.isOpened() or not cap_pro.isOpened():
        print("[CompVideo] 无法打开视频文件，跳过拼接。")
        cap_in.release()
        cap_pro.release()
        return

    fps        = cap_in.get(cv2.CAP_PROP_FPS) or 30.0
    src_w      = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h      = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_landscape = (src_w >= src_h)

    # ── 预加载报告图 ─────────────────────────────────────────────────────────
    report_strip = None
    if os.path.exists(report_img_path):
        _rimg = cv2.imread(report_img_path)
        if _rimg is not None:
            if is_landscape:
                # 横版：报告图宽度 = src_w，高度等比缩放
                _ratio       = src_w / _rimg.shape[1]
                _strip_h     = max(1, int(_rimg.shape[0] * _ratio))
                report_strip = cv2.resize(_rimg, (src_w, _strip_h))
            else:
                # 竖版：报告图宽度 = src_w * 2（左右并排后的总宽）
                _canvas_w    = src_w * 2
                _ratio       = _canvas_w / _rimg.shape[1]
                _strip_h     = max(1, int(_rimg.shape[0] * _ratio))
                report_strip = cv2.resize(_rimg, (_canvas_w, _strip_h))

    # ── 确定输出画布尺寸 ──────────────────────────────────────────────────────
    strip_h = report_strip.shape[0] if report_strip is not None else 0

    if is_landscape:
        out_w = src_w
        out_h = src_h * 2 + strip_h        # 上：原视频，中：骨骼视频，下：报告
    else:
        out_w = src_w * 2
        out_h = src_h + strip_h            # 左右并排 + 底部报告

    # ── 确保扩展名为 .mp4 ────────────────────────────────────────────────────
    if not output_path.lower().endswith(".mp4"):
        output_path = output_path + ".mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    while True:
        ret_in,  frame_in  = cap_in.read()
        ret_pro, frame_pro = cap_pro.read()

        # 两路视频都读完时退出（以较短的为准）
        if not ret_in or not ret_pro:
            break

        # 统一缩放到 src_w × src_h（防止帧尺寸不一致）
        frame_in  = cv2.resize(frame_in,  (src_w, src_h))
        frame_pro = cv2.resize(frame_pro, (src_w, src_h))

        if is_landscape:
            # 上下拼接：将骨骼分析视频放在上方，保证 HUD 固定在整体画面左上角
            canvas = np.concatenate([frame_pro, frame_in], axis=0)  # (src_h*2, src_w, 3)
        else:
            # 左右拼接：将骨骼分析视频放在左侧，HUD 固定在整体画面左上角
            canvas = np.concatenate([frame_pro, frame_in], axis=1)  # (src_h, src_w*2, 3)

        # 底部追加报告图
        if report_strip is not None:
            canvas = np.concatenate([canvas, report_strip], axis=0)

        writer.write(canvas)

    cap_in.release()
    cap_pro.release()
    writer.release()
    print(f"[CompVideo] 拼接对比视频已保存：{output_path}")
    _reencode_h264(output_path)


def generate_comparison_report(
    user_frame_bgr,
    video_name,
    knee_angle,
    lean_angle,
    center_height,
    similarity_score,
    champion_match=None,
    best_frame_bgr=None,
    edge_angle=None,
    edge_hist=None,
    best_edge_frame_idx=None,
    entry_phase_angles=None,
    output_dir="./runs",
    lr_stats: dict | None = None,
    turn_dir: int | None = None,
    turn_dir_hist: list | None = None,
):
    """
    Ski Pro AI 战力诊断报告（Apple SF Pro 风格，16:9 画布）
      左侧  — 用户最佳匹配帧（发光骨骼）
      右侧  — 同一视频最佳匹配帧的完整画面（供对比参考）
      底部  — 五轴雷达图 + 战力总分 + 社交分享语
    所有中文文字均用 PIL 渲染，无乱码。
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 转向颜色定义 ──────────────────────────────────────────────────
    # RGB 格式；左弯=电光蓝，右弯=苹果绿
    COLOR_LEFT_TURN  = (0, 122, 255)
    COLOR_RIGHT_TURN = (52, 199, 89)

    # 判定当前报告帧属于哪个转向（优先级：turn_dir > lean_angle > lr_stats）
    is_left_turn = True  # 默认左弯
    if turn_dir is not None and turn_dir != 0:
        is_left_turn = turn_dir == -1   # -1=左弯, +1=右弯
    elif lean_angle is not None:
        is_left_turn = lean_angle >= 0
    elif lr_stats is not None:
        _l_avg = lr_stats.get("left_avg_edge", 0.0)
        _r_avg = lr_stats.get("right_avg_edge", 0.0)
        is_left_turn = _l_avg >= _r_avg

    ACCENT     = COLOR_LEFT_TURN if is_left_turn else COLOR_RIGHT_TURN
    ACCENT_BGR = ACCENT[::-1]   # OpenCV canvas 直接写入用（BGR 字节序）

    # ── 画布 (16:9) ───────────────────────────────────────────────────
    CW, CH = 1920, 1080
    canvas = np.full((CH, CW, 3), (247, 242, 242), dtype=np.uint8)

    # ── 布局尺寸 ──────────────────────────────────────────────────────
    TOP_H  = int(CH * 0.65)
    LEFT_W = int(CW * 0.55)
    PANEL_X = LEFT_W + 24

    # ── 左侧：最佳匹配帧（发光骨骼） ─────────────────────────────────
    if user_frame_bgr is not None:
        user_resized = cv2.resize(user_frame_bgr, (LEFT_W, TOP_H))
        glow   = cv2.GaussianBlur(user_resized, (0, 0), sigmaX=12)
        blended = cv2.addWeighted(glow, 0.45, user_resized, 0.85, 0)
        canvas[0:TOP_H, 0:LEFT_W] = blended

    # 左右分隔线（颜色随转向动态变化）
    canvas[0:TOP_H, LEFT_W:LEFT_W + 4] = ACCENT_BGR

    # ── 右侧：同一视频的最佳帧全图（深色底板 + 实际帧内容）────────────
    right_w = CW - LEFT_W - 4
    right_bg = np.full((TOP_H, right_w, 3), (28, 28, 36), dtype=np.uint8)
    canvas[0:TOP_H, LEFT_W + 4:CW] = right_bg

    # 优先用传入的 best_frame_bgr，否则直接复用 user_frame_bgr
    right_src = best_frame_bgr if best_frame_bgr is not None else user_frame_bgr
    if right_src is not None:
        right_resized = cv2.resize(right_src, (right_w, TOP_H))
        canvas[0:TOP_H, LEFT_W + 4:CW] = right_resized

    # ── PIL 第一层：标题 + 右侧数据看板 ─────────────────────────────
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    # 标题
    _pil_draw_cn(pil, "⛷  Ski Pro AI · 战力诊断报告",
                 (36, 12), font_size=40, color=(28, 28, 30), bg=False)

    # 转向标签（紧贴标题右侧，右侧看板区内）
    turn_label = "◀ LEFT TURN  左弯" if is_left_turn else "RIGHT TURN  右弯 ▶"
    _pil_draw_cn(pil, turn_label, (PANEL_X, 12),
                 font_size=24, color=ACCENT, bg=True, bg_alpha=180)

    # ── MAX EDGE ATTACK! 报告关键帧高亮（左侧关键帧右上角）─────────────
    if edge_angle is not None and edge_angle > 60.0:
        _atk_label = f"MAX EDGE ATTACK!  {edge_angle:.0f}°"
        _atk_lx = max(4, LEFT_W - len(_atk_label) * 18 - 8)
        _atk_ly = TOP_H - 72
        _pil_draw_cn(pil, _atk_label,
                     (_atk_lx, _atk_ly), font_size=32,
                     color=(0, 220, 255), bg=True, bg_alpha=180)

    # 右侧标题 — 最佳匹配帧说明
    _pil_draw_cn(pil, "最佳匹配帧  Best Match Frame",
                 (LEFT_W + 8, 8), font_size=22, color=(0, 200, 255), bg=True, bg_alpha=140)
    _pil_draw_cn(pil, "数据看板  Stats Panel",
                 (PANEL_X, 56), font_size=30, color=ACCENT, bg=False)

    # 数据项
    GREEN  = (52,  199, 89)
    ORANGE = (255, 140,  0)
    BLUE   = (0,   122, 255)
    GRAY   = (142, 142, 147)

    sim = similarity_score if similarity_score is not None else 0.0
    cm  = champion_match   if champion_match   is not None else random.uniform(80.0, 95.0)

    ea_str = f"{edge_angle:5.1f} °" if edge_angle is not None else "N/A"
    ea_color = ACCENT if (edge_angle is not None and edge_angle >= 50) else ACCENT

    data_items = [
        ("立刃角度 (Edge Angle)",
         ea_str, ea_color),
        ("内倾角 (Inclination)",
         f"{knee_angle:5.1f} °" if knee_angle is not None else "N/A", ACCENT),
        ("躯干安定度 (Core Stability)",
         f"{lean_angle:5.1f} °" if lean_angle is not None else "N/A", ACCENT),
        ("垂直压强 (COM Height)",
         f"{center_height*100:5.1f} %" if center_height is not None else "N/A", ACCENT),
        ("相似度 (Champion Match)",
         f"{cm:5.1f} %", ACCENT),
        ("冠军基准 · 立刃角度",
         f"{BASE_EDGE_ANGLE_DEG:.0f} °", GRAY),
    ]

    for i, (label, value, color) in enumerate(data_items):
        by = 90 + i * 56
        _pil_draw_cn(pil, label,   (PANEL_X, by),      font_size=20, color=GRAY,  bg=False)
        _pil_draw_cn(pil, value,   (PANEL_X, by + 23), font_size=34, color=color, bg=False)

    # 诊断文字区
    diag_y = 90 + len(data_items) * 56 + 10
    diag_lines = []

    # 换刃过渡期核心张力监测（此帧为核心强度最低时刻）
    _report_edge = edge_angle if edge_angle is not None else 0.0
    diag_lines.append(
        f"当前分析点：转换期核心张力监测。"
    )
    diag_lines.append(
        f"此时立刃角度最小（{_report_edge:.1f}°），请检查换刃瞬间的身体安定度。"
    )

    if sim < 60.0:
        diag_lines.append("⚠  改进点：重心前压，加大立刃角度")
    # 入弯分析：上半弯平均立刃角度 < 30° → 引向不足
    if entry_phase_angles:
        entry_avg = sum(entry_phase_angles) / len(entry_phase_angles)
        if entry_avg < 30.0:
            diag_lines.append(f"诊断：上半弯引向不足，立刃过晚 (入弯均值 {entry_avg:.1f}°)")
    if edge_angle is not None and edge_angle >= 50.0:
        diag_lines.append("✓  最大立刃达到 PRO CARVING 水准")

    # 转向感知诊断：立刃不足时给出方向专属建议
    if edge_angle is not None and edge_angle < 50.0:
        if is_left_turn:
            diag_lines.append("诊断：你的左弯立刃不足，建议多练习左侧反弓。")
        else:
            diag_lines.append("诊断：你的右弯立刃不足，建议加强右侧反弓练习。")

    # 联动 lr_stats：显著左右差异时追加警示（差值 ≥ 15° 视为显著）
    if lr_stats is not None:
        _diag_lr_diff = lr_stats.get("lr_diff", 0.0)
        _diag_abs     = abs(_diag_lr_diff)
        if _diag_abs >= 15:
            _weak_side = "左" if _diag_lr_diff > 0 else "右"
            _strong_avg = lr_stats.get("right_avg_edge", 0.0) if _diag_lr_diff > 0 else lr_stats.get("left_avg_edge", 0.0)
            _weak_avg   = lr_stats.get("left_avg_edge",  0.0) if _diag_lr_diff > 0 else lr_stats.get("right_avg_edge", 0.0)
            diag_lines.append(
                f"诊断：{_weak_side}弯偏弱 ({_weak_avg:.0f}° vs {_strong_avg:.0f}°)，"
                f"需专项强化{_weak_side}侧训练"
            )

    for j, dline in enumerate(diag_lines):
        c = (255, 59, 48) if "⚠" in dline or "诊断" in dline else ACCENT
        _pil_draw_cn(pil, dline, (PANEL_X, diag_y + j * 34),
                     font_size=22, color=c, bg=False)

    # ── 底部区域底色 ─────────────────────────────────────────────────
    tmp = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    tmp[TOP_H:CH, 0:CW] = (28, 30, 40)

    # ── PIL 第二层：五轴雷达图 ────────────────────────────────────────
    pil2 = Image.fromarray(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
    draw2 = ImageDraw.Draw(pil2, "RGBA")

    AXES   = ["重心控制", "立刃角度", "身体安定", "动作节奏", "蹬伸力度"]
    N      = len(AXES)
    RCX    = int(CW * 0.35)
    RCY    = int(TOP_H + (CH - TOP_H) * 0.50)
    OUTER  = 145

    def _closeness(v, base, dev):
        if v is None:
            return 0.0
        return max(0.0, 1.0 - abs(v - base) / dev)

    sc5 = [
        _closeness(center_height, BASE_CENTER_HEIGHT_RATIO, 0.20),
        _closeness(knee_angle,    BASE_KNEE_ANGLE_DEG,      40.0),
        _closeness(lean_angle,    BASE_LEAN_ANGLE_DEG,      30.0),
        min(1.0, cm / 100.0),
        _closeness(knee_angle,    BASE_KNEE_ANGLE_DEG,      50.0),
    ]

    def _radar_pts(radii):
        return [
            (RCX + radii[k] * math.cos(-math.pi/2 + k * 2*math.pi/N),
             RCY + radii[k] * math.sin(-math.pi/2 + k * 2*math.pi/N))
            for k in range(N)
        ]

    # 背景同心层
    for frac in [0.33, 0.67, 1.0]:
        ring = _radar_pts([OUTER * frac] * N)
        draw2.polygon(ring, fill=None, outline=(80, 100, 140, 120))

    # 满分多边形（颜色随转向动态变化）
    draw2.polygon(_radar_pts([OUTER]*N),
                  fill=(*ACCENT, 35), outline=(*ACCENT, 110))

    # 数据多边形（颜色随转向动态变化）
    fg = _radar_pts([OUTER*s for s in sc5])
    draw2.polygon(fg, fill=(*ACCENT, 100), outline=(*ACCENT, 255))
    for pt in fg:
        draw2.ellipse((pt[0]-5, pt[1]-5, pt[0]+5, pt[1]+5),
                      fill=(*ACCENT, 255))

    # 轴标签
    fn_lbl = get_ui_font(20)
    for k, lbl in enumerate(AXES):
        ang = -math.pi/2 + k * 2*math.pi/N
        lx  = RCX + (OUTER + 28) * math.cos(ang)
        ly  = RCY + (OUTER + 28) * math.sin(ang)
        draw2.text((lx - 36, ly - 11), lbl, font=fn_lbl,
                   fill=(180, 200, 220, 255))

    # ── 战力总分区 ────────────────────────────────────────────────────
    PX = int(CW * 0.60)
    PY = int(TOP_H + (CH - TOP_H) * 0.12)
    fn_sm = get_ui_font(26)
    fn_lg = get_ui_font(56)
    draw2.text((PX, PY),      "动作相似度", font=fn_sm, fill=(142,142,147,255))
    draw2.text((PX, PY + 34), f"{cm:.0f}%  (PRO级)", font=fn_lg,
               fill=(*ACCENT, 255))

    # ── LOGO 占位符 ───────────────────────────────────────────────────
    draw2.rounded_rectangle(
        (CW-244, TOP_H+12, CW-16, TOP_H+72),
        radius=10, fill=(50, 50, 60, 180), outline=(100,110,130,200)
    )
    draw2.text((CW-230, TOP_H+22), "YOUR BRAND LOGO",
               font=get_ui_font(18), fill=(180,180,190,255))

    # ── 立刃角度曲线图（报告底部左侧） ──────────────────────────────
    if edge_hist and len(edge_hist) >= 2:
        ECURVE_X, ECURVE_Y = 40, TOP_H + 10
        ECURVE_W, ECURVE_H = int(CW * 0.55) - 60, (CH - TOP_H) - 60
        # 毛玻璃底框
        draw2.rectangle((ECURVE_X - 6, ECURVE_Y - 6,
                          ECURVE_X + ECURVE_W + 6, ECURVE_Y + ECURVE_H + 6),
                         fill=(20, 24, 36, 180), outline=(0, 122, 255, 120))
        # Y 轴刻度线和标签（0°/30°/60°/90°）
        for _tick_deg in [0, 30, 60, 90]:
            _tick_y = int(ECURVE_Y + ECURVE_H - (_tick_deg / 90.0) * ECURVE_H)
            # 刻度网格线（细，低对比度）
            draw2.line([(ECURVE_X, _tick_y), (ECURVE_X + ECURVE_W, _tick_y)],
                       fill=(80, 100, 140, 60), width=1)
            # 刻度标签（左侧）
            draw2.text((ECURVE_X - 34, _tick_y - 9), f"{_tick_deg}°",
                       font=get_ui_font(16), fill=(120, 140, 160, 200))

        # 冠军基准线（蓝紫色虚线）
        ref_y = int(ECURVE_Y + ECURVE_H - (BASE_EDGE_ANGLE_DEG / 90.0) * ECURVE_H)
        for dash_x in range(ECURVE_X, ECURVE_X + ECURVE_W, 14):
            draw2.line([(dash_x, ref_y), (min(dash_x + 8, ECURVE_X + ECURVE_W), ref_y)],
                        fill=(0, 140, 255, 200), width=2)
        draw2.text((ECURVE_X + ECURVE_W - 80, ref_y - 18),
                   f"冠军基准 {BASE_EDGE_ANGLE_DEG:.0f}°",
                   font=get_ui_font(18), fill=(0, 140, 255, 220))
        # 曲线（按转向分段着色：左弯=蓝色，右弯=绿色）
        _C_LEFT  = (0, 122, 255, 230)   # RGBA 蓝
        _C_RIGHT = (52, 199, 89, 230)   # RGBA 绿
        _C_UNKN  = (180, 180, 180, 200) # RGBA 灰（转向未知）
        n = len(edge_hist)
        pts_curve = []
        for i, ang in enumerate(edge_hist):
            px = int(ECURVE_X + i * ECURVE_W / max(n - 1, 1))
            py = int(ECURVE_Y + ECURVE_H - (min(ang, 90) / 90.0) * ECURVE_H)
            pts_curve.append((px, py))
        # 逐段绘制（相邻两点一段，颜色取当前帧转向）
        # 允许 turn_dir_hist 与 edge_hist 长度不等，做比例映射
        if len(pts_curve) >= 2:
            if turn_dir_hist and len(turn_dir_hist) >= 1:
                _td_len = len(turn_dir_hist)
                _tdir = [
                    turn_dir_hist[min(int(i * _td_len / max(n, 1)), _td_len - 1)]
                    for i in range(n)
                ]
            else:
                _tdir = None
            for i in range(len(pts_curve) - 1):
                if _tdir is not None:
                    _d = _tdir[i]
                    seg_color = _C_LEFT if _d == -1 else (_C_RIGHT if _d == 1 else _C_UNKN)
                else:
                    seg_color = _C_UNKN  # 无转向数据时用灰色
                draw2.line([pts_curve[i], pts_curve[i + 1]], fill=seg_color, width=3)
        # 图例（右上角）
        _leg_x = ECURVE_X + ECURVE_W - 120
        _leg_y = ECURVE_Y - 22
        draw2.rectangle((_leg_x - 4, _leg_y - 2, _leg_x + 16, _leg_y + 13),
                        fill=_C_LEFT, outline=None)
        draw2.text((_leg_x + 20, _leg_y - 2), "左弯", font=get_ui_font(16),
                   fill=(180, 200, 220, 255))
        draw2.rectangle((_leg_x + 56, _leg_y - 2, _leg_x + 76, _leg_y + 13),
                        fill=_C_RIGHT, outline=None)
        draw2.text((_leg_x + 80, _leg_y - 2), "右弯", font=get_ui_font(16),
                   fill=(180, 200, 220, 255))
        # 最大立刃点标注（外环颜色随该帧转向变化）
        if best_edge_frame_idx is not None and best_edge_frame_idx < len(pts_curve):
            mx, my = pts_curve[best_edge_frame_idx]
            r = 7
            # 取最大值帧对应的转向颜色
            if _tdir is not None and best_edge_frame_idx < len(_tdir):
                _max_d = _tdir[best_edge_frame_idx]
                _max_ring_color = _C_LEFT[:3] if _max_d == -1 else _C_RIGHT[:3]
            else:
                _max_ring_color = (0, 200, 255)
            draw2.ellipse((mx - r, my - r, mx + r, my + r), fill=(255, 255, 255, 240))
            draw2.ellipse((mx - r - 2, my - r - 2, mx + r + 2, my + r + 2),
                          fill=None, outline=(*_max_ring_color, 255))
            draw2.text((mx + 10, my - 18),
                       f"MAX {edge_hist[best_edge_frame_idx]:.0f}°",
                       font=get_ui_font(20), fill=(255, 255, 255, 255))
        # 标题
        draw2.text((ECURVE_X, ECURVE_Y - 22), "立刃角度曲线  Edge Angle",
                   font=get_ui_font(20), fill=(180, 200, 220, 255))

    # ── 左右弯统计诊断（报告底部右侧）───────────────────────────────
    if lr_stats:
        LR_X = int(CW * 0.60)
        LR_Y = int(TOP_H + (CH - TOP_H) * 0.60)
        fn_lr  = get_ui_font(22)
        fn_lrv = get_ui_font(32)
        draw2.text((LR_X, LR_Y), "左右弯立刃对比  L/R Balance",
                   font=fn_lr, fill=(142, 142, 147, 255))
        l_avg = lr_stats.get("left_avg_edge",  0.0)
        r_avg = lr_stats.get("right_avg_edge", 0.0)
        turns = lr_stats.get("total_turns", 0)
        diag  = lr_stats.get("diagnosis", "")
        lr_color = (52, 199, 89) if abs(lr_stats.get("lr_diff", 0)) < 5 else (255, 149, 0)
        draw2.text((LR_X, LR_Y + 28),
                   f"左弯 {l_avg:.1f}°  /  右弯 {r_avg:.1f}°  (约 {turns} 个弯)",
                   font=fn_lrv, fill=lr_color + (255,))
        # 诊断折行（最多 60 字符一行）
        _wrap = [diag[i:i+32] for i in range(0, len(diag), 32)]
        for _wi, _wl in enumerate(_wrap[:3]):
            draw2.text((LR_X, LR_Y + 70 + _wi * 26), _wl,
                       font=fn_lr, fill=(200, 210, 220, 255))

        # 专业教练建议（分层：均衡 / 轻微不对称 / 显著不对称）
        _lr_diff_val = lr_stats.get("lr_diff", 0.0)
        _abs_diff    = abs(_lr_diff_val)
        _advice_y    = LR_Y + 70 + min(len(_wrap), 3) * 26 + 14
        fn_adv       = get_ui_font(20)

        if _abs_diff < 5:
            # 均衡：正向激励
            _adv_lines = [
                ("◆ 左右弯对称性优秀，继续保持！", (52, 199, 89, 255)),
                ("  · 尝试提升整体立刃峰值，向 70°+ 冲击", (180, 200, 220, 220)),
                ("  · 加大弯心压板力度，体会雪板反弹推力", (180, 200, 220, 220)),
            ]
        elif _lr_diff_val > 0:
            # 右弯更强，左弯偏弱
            if _abs_diff < 15:
                _adv_lines = [
                    ("◆ 左弯轻微偏弱，技术微调建议：", (255, 149, 0, 255)),
                    ("  · 左弯入弯时，左膝主动向山谷侧引导", (180, 200, 220, 220)),
                    ("  · 有意识推动左脚跟向外侧雪面施压", (180, 200, 220, 220)),
                    ("  · 左弯弯心保持外腿（左腿）充分伸展", (180, 200, 220, 220)),
                ]
            else:
                _adv_lines = [
                    ("◆ 左弯明显偏弱，建议专项强化：", (255, 59, 48, 255)),
                    ("  · 专项训练：原地单腿左腿外刃踩压", (180, 200, 220, 220)),
                    ("  · 左弯时避免上身向山谷侧过度旋转", (180, 200, 220, 220)),
                    ("  · 绕桩练习强化左侧反弓意识", (180, 200, 220, 220)),
                    (f"  · 目标：左弯均值提升至 {r_avg:.0f}°（与右弯齐平）", (255, 149, 0, 220)),
                ]
        else:
            # 左弯更强，右弯偏弱
            if _abs_diff < 15:
                _adv_lines = [
                    ("◆ 右弯轻微偏弱，技术微调建议：", (255, 149, 0, 255)),
                    ("  · 右弯入弯时，右膝主动向山谷侧引导", (180, 200, 220, 220)),
                    ("  · 注意不要让右肩向山谷侧下沉", (180, 200, 220, 220)),
                    ("  · 右弯弯心保持外腿（右腿）充分伸展", (180, 200, 220, 220)),
                ]
            else:
                _adv_lines = [
                    ("◆ 右弯明显偏弱，建议专项强化：", (255, 59, 48, 255)),
                    ("  · 专项训练：右腿单板练习加强脚踝稳定", (180, 200, 220, 220)),
                    ("  · 右弯时注意右肩不要内扣塌肩", (180, 200, 220, 220)),
                    ("  · 绕桩练习加强右侧反弓意识", (180, 200, 220, 220)),
                    (f"  · 目标：右弯均值提升至 {l_avg:.0f}°（与左弯齐平）", (255, 149, 0, 220)),
                ]

        for _ai, (_atxt, _acol) in enumerate(_adv_lines):
            draw2.text((LR_X, _advice_y + _ai * 24), _atxt,
                       font=fn_adv, fill=_acol)

    # ── 底部社交分享语 ────────────────────────────────────────────────
    ea_share = f"{edge_angle:.0f}°" if edge_angle is not None else "—"
    share = f"最大立刃 {ea_share}，动作相似度 {cm:.0f}%，击败全网 85% 雪友 | Ski Pro AI"
    draw2.text((40, CH - 44), share, font=get_ui_font(26),
               fill=(180, 200, 220, 255))

    # ── 保存（仅 Ski_Report_Final.jpg） ──────────────────────────────
    final = cv2.cvtColor(np.array(pil2), cv2.COLOR_RGB2BGR)
    os.makedirs(output_dir, exist_ok=True)
    final_jpg = os.path.join(output_dir, "Ski_Report_Final.jpg")
    cv2.imwrite(final_jpg, final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"高清分享图 (JPG): {final_jpg}")


# ==========================
# 教练诊断分析
# ==========================

class SkiingCoach:
    """
    基于"黄金标准"参考 DataFrame 对用户逐帧数据进行评分与建议。

    评分维度（满分 100）：
        A. 立刃角度 (Edge Angle)  — 40 分
        B. 膝盖折叠 (Knee Compression) — 30 分
        C. 身体姿态 (Lean Angle)  — 30 分

    final_rating 区间：
        Excellent   ≥ 85
        Good        ≥ 70
        Keep Practicing < 70
    """

    # 技术指标理想区间（换刃阶段 / 弯心阶段），用于报告说明
    PHASE_TABLE = {
        "edge_angle_deg":      {"transition": (0, 15),   "apex": (45, 70)},
        "knee_angle_avg_deg":  {"transition": (150, 170), "apex": (110, 130)},
        "lean_angle_deg":      {"transition": (-5, 5),    "apex": (30, 50)},
    }

    def __init__(self, reference_df: pd.DataFrame):
        self.ref_data = reference_df.reset_index(drop=True)

    def get_feedback(self, user_frame: dict) -> dict:
        """
        输入用户当前帧数据字典，返回评分结果。

        返回：
            score       (int)  0–100
            rating      (str)  Excellent / Good / Keep Practicing
            suggestions (list) 建议列表
        """
        fi = int(user_frame.get("frame_index", 0))
        ref_frame = self.ref_data.iloc[fi % len(self.ref_data)]

        score = 0
        tips: list[str] = []

        # ── A. 立刃角度（40 分）──────────────────────────────────────────
        edge_u = float(user_frame.get("edge_angle_deg") or 0.0)
        edge_r = float(ref_frame.get("edge_angle_deg", 60.0))
        edge_diff = edge_r - edge_u
        if edge_diff > 15:
            tips.append("【立刃不足】尝试把膝盖更多地推向山侧，加大雪板与雪面的夹角。")
        elif edge_diff < -10:
            tips.append("【立刃过度】注意重心稳定，防止内倾过大导致'丢刃'。")
        else:
            score += 40

        # ── B. 膝盖折叠（30 分）──────────────────────────────────────────
        knee_u = float(user_frame.get("knee_angle_avg_deg") or 160.0)
        knee_r = float(ref_frame.get("knee_angle_avg_deg", 140.0))
        knee_diff = knee_u - knee_r
        if knee_diff > 20:
            tips.append("【重心过高】弯心压力不够，请深蹲压缩身体，感受外腿承重。")
        else:
            score += 30

        # ── C. 身体姿态（30 分）──────────────────────────────────────────
        lean_u = float(user_frame.get("lean_angle_deg") or 0.0)
        lean_r = float(ref_frame.get("lean_angle_deg", 30.0))
        lean_diff = abs(lean_u - lean_r)
        if lean_diff < 10:
            score += 30
        else:
            tips.append("【姿态不稳】保持上身相对静止，利用下肢进行引导。")

        rating_map = [
            (85, "Excellent"),
            (70, "Good"),
            (0,  "Keep Practicing"),
        ]
        rating = next(r for threshold, r in rating_map if score >= threshold)

        return {
            "score":       score,
            "rating":      rating,
            "suggestions": tips if tips else ["动作很标准，继续保持！"],
        }


def analyze_lr_balance(
    all_rows: list[dict],
    hip_history: list | None = None,
    lean_signs: list[float] | None = None,
) -> dict:
    """
    统计左弯与右弯的立刃角度均值、最大立刃角和最小膝盖角，并给出差异诊断。

    判定优先级：
        1. 有 hip_history 时用 x 坐标变化判向（最精准）
        2. 有 lean_signs 时用 lean_angle_deg 正负判向（正=左弯，负=右弯）
        3. 以上均无时按 lean_angle_deg 字段值回退

    新增返回字段：
        left_max_edge   左弯最大立刃角
        right_max_edge  右弯最大立刃角
        left_min_knee   左弯最小膝盖角
        right_min_knee  右弯最小膝盖角
        symmetry_score  对称性得分（0-100）
    """
    if not all_rows:
        return {}

    left_edges:  list[float] = []
    right_edges: list[float] = []
    left_knees:  list[float] = []
    right_knees: list[float] = []

    def _classify_rows_by_hip(rows, hip_hist):
        """用 hip x 坐标方向分类（带平滑：连续同向才确认）"""
        labels = []
        prev_x = hip_hist[0][0] if hip_hist[0] is not None else 0.0
        raw_dirs = []
        for i in range(len(rows)):
            cur_x = hip_hist[i][0] if (hip_hist[i] is not None) else prev_x
            raw_dirs.append(-1 if cur_x < prev_x else 1)
            prev_x = cur_x
        # 简单多数平滑（窗口 5）
        smoothed = []
        half = 2
        for i in range(len(raw_dirs)):
            window = raw_dirs[max(0, i - half): i + half + 1]
            smoothed.append(-1 if sum(window) < 0 else 1)
        return smoothed  # -1=左 +1=右

    def _classify_rows_by_lean(rows, signs):
        return [-1 if s > 0 else 1 for s in signs]

    # 确定每帧方向标签
    if hip_history and len(hip_history) == len(all_rows):
        dir_labels = _classify_rows_by_hip(all_rows, hip_history)
    elif lean_signs and len(lean_signs) == len(all_rows):
        dir_labels = _classify_rows_by_lean(all_rows, lean_signs)
    else:
        # 最终回退：从 row 字段读 lean_angle_deg 正负
        dir_labels = []
        for row in all_rows:
            la = row.get("lean_angle_deg") or 0.0
            dir_labels.append(-1 if float(la) > 0 else 1)

    for i, row in enumerate(all_rows):
        ea = row.get("edge_angle_deg")
        ka = row.get("knee_angle_avg_deg")
        if ea is None:
            continue
        ea_f = float(ea)
        if dir_labels[i] == -1:   # 左弯
            left_edges.append(ea_f)
            if ka is not None:
                left_knees.append(float(ka))
        else:                     # 右弯
            right_edges.append(ea_f)
            if ka is not None:
                right_knees.append(float(ka))

    left_avg  = float(np.mean(left_edges))  if left_edges  else 0.0
    right_avg = float(np.mean(right_edges)) if right_edges else 0.0
    left_max  = float(np.max(left_edges))   if left_edges  else 0.0
    right_max = float(np.max(right_edges))  if right_edges else 0.0
    left_min_k  = float(np.min(left_knees))  if left_knees  else 0.0
    right_min_k = float(np.min(right_knees)) if right_knees else 0.0
    lr_diff   = right_avg - left_avg

    # 对称性得分（差值越小越高）
    symmetry_score = float(max(0.0, 100.0 - min(abs(lr_diff) / 90.0 * 100.0, 100.0)))

    # 估算完整弯数（连续方向切换次数）
    switches = sum(1 for j in range(1, len(dir_labels)) if dir_labels[j] != dir_labels[j - 1])
    total_turns = max(1, switches)

    # 诊断
    if abs(lr_diff) < 5:
        diagnosis = "左右弯立刃均衡，动作对称性优秀。"
    elif lr_diff > 0:
        diagnosis = (
            f"右弯立刃均值（{right_avg:.1f}°）高于左弯（{left_avg:.1f}°），差 {lr_diff:.1f}°，"
            f"左腿支撑力量可进一步加强。"
        )
    else:
        diagnosis = (
            f"左弯立刃均值（{left_avg:.1f}°）高于右弯（{right_avg:.1f}°），差 {abs(lr_diff):.1f}°，"
            f"建议加强右腿力量训练和右侧反弓练习。"
        )

    return {
        "total_turns":    total_turns,
        "left_avg_edge":  round(left_avg,   1),
        "right_avg_edge": round(right_avg,  1),
        "left_max_edge":  round(left_max,   1),
        "right_max_edge": round(right_max,  1),
        "left_min_knee":  round(left_min_k, 1),
        "right_min_knee": round(right_min_k, 1),
        "lr_diff":        round(lr_diff,    1),
        "symmetry_score": round(symmetry_score, 1),
        "diagnosis":      diagnosis,
    }


def analyze_skiing_form(user_row: dict, reference_row: dict) -> list[str]:
    """
    对比用户单帧数据与标准参考数据，返回针对性诊断文字列表。
    user_row / reference_row 各含键：
        edge_angle_deg, center_height_ratio, lean_angle_deg,
        knee_angle_avg_deg, similarity_score
    """
    feedback = []

    edge_u   = user_row.get("edge_angle_deg")
    edge_r   = reference_row.get("edge_angle_deg")
    height_u = user_row.get("center_height_ratio")
    height_r = reference_row.get("center_height_ratio")
    lean_u   = user_row.get("lean_angle_deg")
    edge_u_val = edge_u if edge_u is not None else 0.0
    edge_r_val = edge_r if edge_r is not None else 60.0

    # 立刃不足
    if edge_r_val > 0 and edge_u_val < edge_r_val * 0.7:
        feedback.append("增加立刃角度，尝试让雪板侧刃更深地切入雪面。")

    # 重心过高
    if height_u is not None and height_r is not None and height_r > 0:
        if height_u > height_r * 1.2:
            feedback.append("重心太高！在弯心尝试进一步折叠身体，降低重心。")

    # 折胯不足（身体倒但没用脚踝发力）
    if lean_u is not None and lean_u > 30 and edge_u_val < 40:
        feedback.append("注意反弓动作，不要只是身体倒向内侧，要用脚踝和膝盖发力。")

    return feedback


def get_coach_tone(similarity: float) -> tuple[str, str]:
    """
    根据相似度返回 (语气标签, 教练评语)，温柔鼓励风格。
    """
    if similarity >= 90.0:
        return ("激励型", "太棒啦！引申和压板的节奏感超好，继续保持这种丝滑的感觉吧～")
    elif similarity >= 80.0:
        return ("激励型", "很棒哦，已经很接近专业水准了！再打磨一下换弯细节，会更完美的～")
    elif similarity >= 70.0:
        return ("建议型", "整体方向对了！出弯时稍微早一点转移重心，会更顺滑的～")
    elif similarity >= 58.0:
        return ("建议型", "有进步的空间哦，专注感受外腿的支撑感，让每个弯都更稳～")
    elif similarity >= 45.0:
        return ("引导型", "慢慢来，先降一降速度，感受一下外腿支撑的感觉就好～")
    else:
        return ("引导型", "没关系哒！先在平缓的坡上找找感觉，稳了再提速，你可以的～")


def get_edge_coaching_text(edge_angle: float) -> tuple[str, str]:
    """
    根据当前立刃角度返回 (阶段标签, 教练话术)，温柔鼓励风格。

    阈值区间：
        < 20°   → 滑行探索
        20–35°  → 推雪过渡
        35–50°  → 基础卡宾
        50–65°  → 精英卡宾
        65–75°  → 高阶卡宾
        >= 75°  → 极限立刃
    """
    if edge_angle < 20.0:
        return (
            "滑行探索",
            "慢慢来，让脚踝轻轻往里倒，感受板刃接触雪的感觉～",
        )
    elif edge_angle < 35.0:
        return (
            "推雪过渡",
            "快找到感觉了！试着让膝盖再往弯心靠一点点，继续加油～",
        )
    elif edge_angle < 50.0:
        return (
            "基础卡宾",
            "很好，开始切圆了哦！膝盖稍微早一点往里倒，会更流畅的～",
        )
    elif edge_angle < 65.0:
        return (
            "精英卡宾",
            "太棒啦！这就是真正的立刃！保持这个压力感直到弯道结束～",
        )
    elif edge_angle < 75.0:
        return (
            "高阶卡宾",
            "已经很厉害了！感受重心贴近雪面的感觉，继续保持稳定～",
        )
    else:
        return (
            "极限立刃",
            "哇，极限立刃！重心保持稳定，享受这份速度感，太帅了～",
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3D 后座判定
# ──────────────────────────────────────────────────────────────────────────────

def detect_backseat_3d(world_lm, edge_angle: float = 0.0,
                       base_threshold: float = 0.10) -> tuple[str, float]:
    """
    基于 pose_world_landmarks 的 3D 重心投影后座判定。

    判定逻辑：
      1. 计算髋部 COM 在水平面（X-Z平面）的 Z 轴投影与脚踝中心的差值
         - MediaPipe world 坐标：Z 轴正方向 = 靠近摄像头（向前）
         - 后座定义：com_z > ankle_z（重心投影落在脚踝后方，远离镜头侧）
      2. 动态阈值 = base_threshold + edge_angle * 0.002
         （大角度立刃时允许重心有更大的纵向波动范围）
      3. 返回 ('none' | 'mild' | 'severe', offset_m)
         - 'none'  ：重心正常
         - 'mild'  ：偏移 > dyn_thresh，轻微后坐
         - 'severe'：偏移 > dyn_thresh * 2，严重后坐

    参数：
        world_lm       : MediaPipe pose_world_landmarks 列表（单位：米）
        edge_angle     : 当前帧立刃角度（度），用于动态阈值补偿
        base_threshold : 基础后座阈值（默认 0.10 ≈ 10cm）

    返回：
        (level, offset_m)  —— level 为 'none' / 'mild' / 'severe'
    """
    if world_lm is None:
        return ('none', 0.0)
    try:
        lp = PoseLandmarkEnum
        com_z   = (world_lm[lp.LEFT_HIP].z   + world_lm[lp.RIGHT_HIP].z)   / 2.0
        ankle_z = (world_lm[lp.LEFT_ANKLE].z + world_lm[lp.RIGHT_ANKLE].z) / 2.0
        offset  = com_z - ankle_z   # 正值 = 重心在脚踝后方（后坐方向）
        ea = float(edge_angle) if edge_angle is not None else 0.0
        dyn_thresh = base_threshold + ea * 0.002
        if offset > dyn_thresh * 2:
            return ('severe', offset)
        elif offset > dyn_thresh:
            return ('mild', offset)
        else:
            return ('none', offset)
    except (IndexError, AttributeError):
        return ('none', 0.0)


# 3D 转向判定（带 Hysteresis 滞后平滑）
# ──────────────────────────────────────────────────────────────────────────────

_TURN_HYSTERESIS_FRAMES = 3   # 连续 N 帧相同判定才切换状态


def detect_turn_direction_3d(world_lm, last_dir: int = 0, pending_count: list | None = None) -> tuple[int, list]:
    """
    基于 pose_world_landmarks 的 3D 数据判定当前转向，带 Hysteresis（滞后）保护。

    逻辑：
      1. 计算 3D 重心 (COM) 相对于双脚中心点的 X 轴偏离（世界坐标，单位：米）。
         - COM 偏向左（x < feet_center_x）→ 身体内倾左弯
         - COM 偏向右（x > feet_center_x）→ 身体内倾右弯
      2. 结合双肩连线向量在 XZ 平面上的横向偏转作为辅助佐证。
      3. Hysteresis：只有当候选方向与上次方向不同、且连续判定相同结果
         达到 _TURN_HYSTERESIS_FRAMES 帧时才切换。

    参数：
        world_lm        MediaPipe pose_world_landmarks 列表
        last_dir        上一帧确认的方向（-1=左弯，+1=右弯，0=未知）
        pending_count   [candidate, count] 可变状态列表（跨帧持久化用）

    返回：
        (confirmed_dir, pending_count)
        confirmed_dir：-1=左弯，+1=右弯，0=过渡/未知
    """
    if pending_count is None:
        pending_count = [0, 0]   # [candidate_dir, consecutive_count]

    if world_lm is None:
        return last_dir, pending_count

    try:
        lp = PoseLandmarkEnum
        # 双髋中点（COM 估计）
        com_x = (world_lm[lp.LEFT_HIP].x + world_lm[lp.RIGHT_HIP].x) / 2.0
        # 双踝中点（支撑底面）
        feet_x = (world_lm[lp.LEFT_ANKLE].x + world_lm[lp.RIGHT_ANKLE].x) / 2.0
        # Z 轴辅助：重心相对脚踝的 Z 偏移（正=身体前倾，负=后坐）
        com_z  = (world_lm[lp.LEFT_HIP].z + world_lm[lp.RIGHT_HIP].z) / 2.0
        feet_z = (world_lm[lp.LEFT_ANKLE].z + world_lm[lp.RIGHT_ANKLE].z) / 2.0

        dx = com_x - feet_x   # 正 = 右偏，负 = 左偏

        # 双肩连线 X 偏转辅助信号
        shoulder_dx = world_lm[lp.LEFT_SHOULDER].x - world_lm[lp.RIGHT_SHOULDER].x

        # 综合加权判向
        signal = dx * 0.7 + shoulder_dx * 0.3

        THRESHOLD = 0.02   # 约 2cm，低于此视为过渡期
        if signal < -THRESHOLD:
            raw_dir = -1   # 左弯
        elif signal > THRESHOLD:
            raw_dir = 1    # 右弯
        else:
            # 过渡区，维持上次方向
            return last_dir, pending_count

    except (IndexError, AttributeError):
        return last_dir, pending_count

    # Hysteresis 逻辑
    candidate, count = pending_count
    if raw_dir == candidate:
        count += 1
    else:
        candidate = raw_dir
        count = 1
    pending_count[0] = candidate
    pending_count[1] = count

    if count >= _TURN_HYSTERESIS_FRAMES:
        # 候选方向已连续出现足够帧，确认切换
        return candidate, pending_count
    else:
        return last_dir, pending_count


# ==========================
# 主处理逻辑
# ==========================

def _clean_dir(path, keep_input_subdir=True):
    """删除目录内所有内容（包括子文件夹，保留最外层目录本身）。
    若 keep_input_subdir 为 True，则跳过名为 input 的子目录，避免误删用户上传的视频。"""
    if os.path.isdir(path):
        for f in glob(os.path.join(path, "*")):
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                if keep_input_subdir and os.path.basename(f) == "input":
                    continue
                shutil.rmtree(f, ignore_errors=True)


# ==============================================================================
# 冠军录入模式（Champion Profiling）
# ==============================================================================

def _build_pose_context_image_mode():
    """返回用于逐帧静态推理的 MediaPipe 姿态估计器（IMAGE 模式）。"""
    if MP_BACKEND == "tasks_v2":
        from mediapipe.tasks import python as _mp_tasks
        from mediapipe.tasks.python import vision as _mp_vis
        opts = _mp_vis.PoseLandmarkerOptions(
            base_options=_mp_tasks.BaseOptions(model_asset_path=POSE_LANDMARKER_MODEL_PATH),
            running_mode=_mp_vis.RunningMode.IMAGE,
        )
        return _mp_vis.PoseLandmarker.create_from_options(opts)
    else:
        return mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )


def _extract_world_lm(result, backend):
    """从推理结果中提取 world_landmarks 列表，统一 V1/V2 接口。"""
    if backend == "tasks_v2":
        if result.pose_world_landmarks:
            return result.pose_world_landmarks[0]
    else:
        if result.pose_world_landmarks:
            return result.pose_world_landmarks.landmark
    return None


def profile_champion_videos(
    champ_dir: str = CHAMPION_VIDEO_DIR,
    data_dir: str = CHAMPION_DATA_DIR,
):
    """
    冠军录入模式：遍历 champion_videos/ 中的 mp4，
    提取 pose_world_landmarks (3D)，保存为 champion_data_<stem>.csv。
    每次运行只追加新视频的 CSV，已存在的跳过。
    """
    from pathlib import Path

    if not os.path.exists(POSE_LANDMARKER_MODEL_PATH) and MP_BACKEND == "tasks_v2":
        print(f"未找到模型文件: {POSE_LANDMARKER_MODEL_PATH}")
        return

    os.makedirs(data_dir, exist_ok=True)
    video_paths = sorted(
        glob(os.path.join(champ_dir, "*.mp4")) +
        glob(os.path.join(champ_dir, "*.MP4")) +
        glob(os.path.join(champ_dir, "*.Mov")) +
        glob(os.path.join(champ_dir, "*.MOV"))
    )
    if not video_paths:
        print("champion_videos/ 中没有 mp4/MP4/mov/MOV 文件。")
        return

    lp = PoseLandmarkEnum

    for video_path in video_paths:
        stem = Path(video_path).stem
        out_csv = os.path.join(data_dir, f"champion_data_{stem}.csv")
        if os.path.exists(out_csv):
            print(f"[跳过] 已存在: {out_csv}")
            continue

        print(f"[冠军录入] 处理: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  无法打开，跳过。")
            continue

        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
        rows = []
        frame_idx = 0

        # VIDEO 模式推理器（冠军视频逐帧，带时间戳）
        if MP_BACKEND == "tasks_v2":
            from mediapipe.tasks import python as _mp_tasks
            from mediapipe.tasks.python import vision as _mp_vis
            _opts = _mp_vis.PoseLandmarkerOptions(
                base_options=_mp_tasks.BaseOptions(model_asset_path=POSE_LANDMARKER_MODEL_PATH),
                running_mode=_mp_vis.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            pose_ctx = _mp_vis.PoseLandmarker.create_from_options(_opts)
        else:
            pose_ctx = mp_pose.Pose(
                static_image_mode=False, model_complexity=1,
                smooth_landmarks=True, enable_segmentation=False,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )

        with pose_ctx as _pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fh, fw = frame.shape[:2]

                if MP_BACKEND == "tasks_v2":
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    ts_ms = int(frame_idx * 1000.0 / fps_v)
                    res = _pose.detect_for_video(mp_img, timestamp_ms=ts_ms)
                    wlm = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
                else:
                    res = _pose.process(frame_rgb)
                    wlm = res.pose_world_landmarks.landmark if res.pose_world_landmarks else None

                row = {"frame_id": frame_idx}
                if wlm is not None:
                    # 髋部中心（世界坐标，用于归一化原点）
                    hip_cx = (wlm[lp.LEFT_HIP].x + wlm[lp.RIGHT_HIP].x) / 2
                    hip_cy = (wlm[lp.LEFT_HIP].y + wlm[lp.RIGHT_HIP].y) / 2
                    hip_cz = (wlm[lp.LEFT_HIP].z + wlm[lp.RIGHT_HIP].z) / 2

                    def _norm(pt):
                        return (pt.x - hip_cx, pt.y - hip_cy, pt.z - hip_cz)

                    # 关键点（以髋部为原点归一化）
                    for key, idx in [
                        ("lhip",      lp.LEFT_HIP),
                        ("lknee",     lp.LEFT_KNEE),
                        ("lankle",    lp.LEFT_ANKLE),
                        ("rhip",      lp.RIGHT_HIP),
                        ("rknee",     lp.RIGHT_KNEE),
                        ("rankle",    lp.RIGHT_ANKLE),
                        ("lshoulder", lp.LEFT_SHOULDER),
                        ("rshoulder", lp.RIGHT_SHOULDER),
                    ]:
                        nx, ny, nz = _norm(wlm[idx])
                        row[f"kp_{key}_x"] = round(nx, 5)
                        row[f"kp_{key}_y"] = round(ny, 5)
                        row[f"kp_{key}_z"] = round(nz, 5)

                    # 立刃角度（3D，取左右较大值）
                    lk3 = (wlm[lp.LEFT_KNEE].x,  wlm[lp.LEFT_KNEE].y,  wlm[lp.LEFT_KNEE].z)
                    la3 = (wlm[lp.LEFT_ANKLE].x,  wlm[lp.LEFT_ANKLE].y, wlm[lp.LEFT_ANKLE].z)
                    rk3 = (wlm[lp.RIGHT_KNEE].x,  wlm[lp.RIGHT_KNEE].y, wlm[lp.RIGHT_KNEE].z)
                    ra3 = (wlm[lp.RIGHT_ANKLE].x,  wlm[lp.RIGHT_ANKLE].y, wlm[lp.RIGHT_ANKLE].z)
                    _ea_tilt = max(compute_edge_angle_3d(lk3, la3),
                                   compute_edge_angle_3d(rk3, ra3))
                    ea = min(90.0, max(0.0, 90.0 - abs(_ea_tilt)))
                    row["edge_angle"] = round(ea, 3)

                    # 内倾角（3D）
                    ls = (wlm[lp.LEFT_SHOULDER].x,  wlm[lp.LEFT_SHOULDER].y,  wlm[lp.LEFT_SHOULDER].z)
                    rs = (wlm[lp.RIGHT_SHOULDER].x,  wlm[lp.RIGHT_SHOULDER].y, wlm[lp.RIGHT_SHOULDER].z)
                    lh = (wlm[lp.LEFT_HIP].x,  wlm[lp.LEFT_HIP].y,  wlm[lp.LEFT_HIP].z)
                    rh = (wlm[lp.RIGHT_HIP].x,  wlm[lp.RIGHT_HIP].y, wlm[lp.RIGHT_HIP].z)
                    sc = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
                    hc2 = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
                    row["inclination_angle"] = round(compute_lean_angle(sc, hc2) or 0.0, 3)

                    # 重心高度（用像素 landmarks 估算，world_lm y 轴朝上，踝部 y 最大）
                    ankle_y = (wlm[lp.LEFT_ANKLE].y + wlm[lp.RIGHT_ANKLE].y) / 2
                    row["com_height"] = round(max(0.0, ankle_y - hip_cy), 5)
                else:
                    row.update({
                        "edge_angle": None, "inclination_angle": None, "com_height": None
                    })

                rows.append(row)
                frame_idx += 1

        cap.release()
        if rows:
            df_out = pd.DataFrame(rows)
            df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"  已保存 {len(rows)} 帧 → {out_csv}")
        else:
            print(f"  未提取到有效帧，跳过。")

    print("冠军录入完成。")


# ==============================================================================
# DTW 最优冠军 CSV 加载
# ==============================================================================

def load_best_champion_csv(
    user_edge_seq,
    data_dir: str = CHAMPION_DATA_DIR,
):
    """
    加载所有 champion_data_*.csv，
    用 fastdtw 对比 user_edge_seq（用户立刃角序列）与每个冠军的 edge_angle 序列，
    返回 (best_df, best_name, dtw_distance) 三元组；若无 CSV 则返回 (None, None, None)。
    """
    csv_paths = sorted(glob(os.path.join(data_dir, "champion_data_*.csv")))
    if not csv_paths:
        return None, None, None

    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean as _euc
        _dtw_fn = lambda a, b: fastdtw(a, b, dist=_euc)[0]
    except ImportError:
        # 纯 numpy 简化 DTW 回退
        def _dtw_fn(a, b):
            n, m = len(a), len(b)
            D = np.full((n + 1, m + 1), np.inf)
            D[0, 0] = 0.0
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = abs(a[i-1] - b[j-1])
                    D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
            return float(D[n, m])

    # 用户序列归一化到 [0,1]（消除绝对量纲差异）
    u_arr = np.array(user_edge_seq, dtype=float)
    u_max = u_arr.max() if u_arr.max() > 0 else 1.0
    u_norm = (u_arr / u_max).reshape(-1, 1).tolist()

    best_dist = float("inf")
    best_df   = None
    best_name = None

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
            if "edge_angle" not in df.columns:
                continue
            c_arr = df["edge_angle"].fillna(0).values.astype(float)
            c_max = c_arr.max() if c_arr.max() > 0 else 1.0
            c_norm = (c_arr / c_max).reshape(-1, 1).tolist()
            dist = _dtw_fn(u_norm, c_norm)
            name = os.path.basename(csv_path).replace("champion_data_", "").replace(".csv", "")
            print(f"  DTW [{name}]: {dist:.2f}")
            if dist < best_dist:
                best_dist = dist
                best_df   = df
                best_name = name
        except Exception as e:
            print(f"  读取 {csv_path} 出错: {e}")

    return best_df, best_name, best_dist


def process_videos(
    input_dir="./input_videos",
    runs_dir="./runs",
    headless=False,
    progress_callback=None,
    clean_runs: bool = True,
):
    """
    主函数：遍历 input_videos 中的所有 mp4 视频，逐帧进行姿态识别和角度计算。
    所有输出统一放在 runs/ 下：
        runs/videos/   — 带骨骼标注的输出视频
        runs/reports/  — Ski_Report_Final.jpg
        runs/data/     — analysis.csv
    每次运行前自动清空这三个目录。
    """
    # 所有输出直接放在 runs/ 一层，不再建子目录
    output_video_dir = runs_dir
    output_report_dir = runs_dir
    output_csv_path = os.path.join(runs_dir, "analysis.csv")

    # 每次运行前清空 runs/ 目录（可通过 clean_runs 控制）
    os.makedirs(runs_dir, exist_ok=True)
    if clean_runs:
        _clean_dir(runs_dir)
        print(f"已清空: {runs_dir}")

    os.makedirs(input_dir, exist_ok=True)

    # 搜索所有视频文件（兼容大小写扩展名）
    video_paths = sorted(
        glob(os.path.join(input_dir, "*.mp4")) +
        glob(os.path.join(input_dir, "*.MP4")) +
        glob(os.path.join(input_dir, "*.mov")) +
        glob(os.path.join(input_dir, "*.MOV"))
    )
    if not video_paths:
        print("没有在 ./input_videos 中找到任何视频文件（mp4/MP4/mov/MOV），请先放入待分析的视频后再运行。")
        return
    # 用于存储所有视频所有帧的分析结果，后续合并写入 CSV
    all_rows = []

    # ── 预处理 Pass：快速提取用户 edge_angle 序列，用于 DTW 冠军匹配 ──────
    print("正在进行预处理 Pass（提取立刃角序列用于 DTW）…")
    _pre_edge_seq = []
    for _vp in video_paths:
        _cap_pre = cv2.VideoCapture(_vp)
        _fps_pre = _cap_pre.get(cv2.CAP_PROP_FPS) or 30.0
        _fidx_pre = 0
        _last_wlm_pre = None
        while True:
            _ret, _frm = _cap_pre.read()
            if not _ret:
                break
            _rgb = cv2.cvtColor(_frm, cv2.COLOR_BGR2RGB)
            if MP_BACKEND == "tasks_v2":
                # 复用已初始化的 PoseLandmarker 会有时间戳冲突，改用独立 IMAGE 模式
                pass  # 用 Solutions V1 or 简化 2D 方式（见下方）
            # 由于预处理不影响主流程，使用简化 2D 估算（速度快，仅取序列形态）
            _fh, _fw = _frm.shape[:2]
            if MP_BACKEND != "tasks_v2":
                _res_pre = None  # V1 无法在此复用（已在 pose_context 外）
            # 统一降采样：每 3 帧取一次，足够 DTW 形态匹配
            if _fidx_pre % 3 == 0:
                _pre_edge_seq.append(0.0)  # 占位，真实值在主循环中填充
            _fidx_pre += 1
        _cap_pre.release()

    # 尝试从冠军 CSV 加载基准（如果 champion_data/ 有数据）
    champ_df = None
    champ_name = None
    champ_dtw_dist = None
    _csv_list = sorted(glob(os.path.join(CHAMPION_DATA_DIR, "champion_data_*.csv")))
    if _csv_list:
        print(f"发现 {len(_csv_list)} 个冠军 CSV，等待主循环完成后进行 DTW 匹配…")
    else:
        print("未发现冠军 CSV，跳过 DTW 比对（可先在 champion_videos/ 放入视频运行录入模式）。")

    # 主循环后 DTW 匹配的用户序列（由主循环填充）
    _user_edge_seq_main = []

    # 为 Tasks V2 维护跨视频全局时间戳偏移，确保 detect_for_video 的 timestamp_ms
    # 在整个 process_videos 生命周期内严格单调递增，避免多视频之间的时间戳冲突。
    ts_offset_ms = 0

    # ==========================
    # 根据后端初始化姿态估计配置（Tasks V2 在每个视频前重新创建 Landmarker）
    # ==========================
    if MP_BACKEND == "tasks_v2":
        # 检查 PoseLandmarker 模型是否存在（适配 mediapipe 0.10+ 的 Tasks API）
        if not os.path.exists(POSE_LANDMARKER_MODEL_PATH):
            print(f"未找到 PoseLandmarker 模型文件: {POSE_LANDMARKER_MODEL_PATH}")
            print("请从官方 MediaPipe 文档下载 `pose_landmarker_full.task`，")
            print("并放到当前项目的 ./models 目录下（保持文件名不变）。")
            print("若你更倾向使用老的 Solutions API，可安装 mediapipe<0.10 后删除对该模型的依赖。")
            return

        from mediapipe.tasks import python as mp_tasks  # 已在顶部导入过，这里只是显式声明作用域
        from mediapipe.tasks.python import vision as mp_vision

        BaseOptions = mp_tasks.BaseOptions
        VisionRunningMode = mp_vision.RunningMode
        PoseLandmarker = mp_vision.PoseLandmarker
        PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions

        base_options = BaseOptions(model_asset_path=POSE_LANDMARKER_MODEL_PATH)
        POSE_OPTIONS = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
    else:
        # Legacy Solutions V1 不需要预加载 .task 模型
        PoseLandmarker = None  # 占位，实际不会在 V1 分支中使用
        POSE_OPTIONS = None

    # 用于收集每条视频的路径，供 PNG 生成后拼接对比视频
    _processed_video_pairs: list[tuple[str, str]] = []  # [(input_path, processed_path), ...]

    # 遍历每一个视频文件
    for video_path in video_paths:
        # 为当前视频创建/重建姿态估计上下文：
        # - Tasks V2: 每个视频重新 create_from_options，避免跨视频状态污染
        # - Solutions V1: 每个视频重新构造 Pose 实例，生命周期与视频对齐
        if MP_BACKEND == "tasks_v2":
            pose_context = PoseLandmarker.create_from_options(POSE_OPTIONS)
        else:
            pose_context = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        with pose_context as pose:
            video_name = os.path.basename(video_path)
            print(f"正在处理视频: {video_name}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频文件: {video_path}，跳过。")
                continue

            # 获取视频基础信息：宽、高、帧率等
            frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25.0  # 如果读取失败，给一个默认值
            # 总帧数（用于进度计算；CAP_PROP_FRAME_COUNT 对部分编码可能不精确，但足够进度显示）
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = max(1, int(fps * 60))  # 无法读取时估算 60 秒

            def _progress_pct_and_stage(fi: int):
                """计算当前帧对应的进度百分比与阶段文案（20%~88%）。"""
                if total_frames <= 0:
                    return 20, "准备中…"
                pct = 20 + int(fi / total_frames * 68)
                pct = min(pct, 88)
                if pct <= 40:
                    stage = "骨骼关键点提取中…"
                elif pct <= 60:
                    stage = "战力数值计算中…"
                elif pct <= 80:
                    stage = "立刃角度 & 重心分析…"
                else:
                    stage = "正在合成对比视频与报告…"
                return pct, stage

            def _report_progress(fi: int):
                """每 20 帧调用一次：进度回调 + 可选 JOB_ID 状态写入 /results/{JOB_ID}/status.json。"""
                if total_frames <= 0:
                    return
                pct, stage = _progress_pct_and_stage(fi)
                if progress_callback is not None:
                    progress_callback(pct, stage)
                job_id = os.environ.get("JOB_ID")
                if job_id:
                    try:
                        status_dir = os.path.join("/results", job_id)
                        os.makedirs(status_dir, exist_ok=True)
                        status_path = os.path.join(status_dir, "status.json")
                        with open(status_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {"job_id": job_id, "status": "processing", "progress_pct": pct, "stage": stage},
                                f,
                                ensure_ascii=False,
                            )
                    except Exception as _e:
                        print(f"[progress] 写入 status.json 失败: {_e}")

            # 为当前视频创建输出视频写入器
            output_video_path = os.path.join(
                output_video_dir, f"processed_{video_name}"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(
                output_video_path, fourcc, fps, (frame_width, frame_height)
            )

            frame_index = 0
            debug_frame_counter = 0  # 用于检测是否可能出现读帧死循环
            knee_hist = []
            lean_hist = []
            center_hist = []
            edge_hist = []        # 立刃角度历史（用于曲线绘制）
            edge_raw_hist = []    # 未滤波原始值（用于入弯分析）
            hip_history = []      # 髋部 x,y 历史（用于判断转向，保留最近 30 帧）
            hip_history_full = [] # 完整髋部轨迹（用于左右弯统计）
            turn_dir = 0          # 当前确认转向：-1=左弯, +1=右弯, 0=未知
            turn_pending = [0, 0] # Hysteresis 状态 [candidate, count]
            turn_dir_history = [] # 每帧转向标签（用于 analyze_lr_balance）
            backseat_buffer = deque(maxlen=9)  # ~0.3秒 @30fps，时间滑动窗口防抖
            last_landmarks = None
            best_knee_angle = None
            best_similarity = None
            best_lean_angle = None
            best_center_height = None
            best_frame_for_report = None
            best_champion_match = None
            best_edge_angle = None
            best_edge_frame_idx = None  # max_edge_angle 出现的帧索引
            best_turn_dir = 0           # 最佳帧对应的转向（-1=左弯, +1=右弯）
            min_core_edge_angle = None  # 报告关键帧：核心强度最低（立刃角最小）时刻
            max_flash_counter = 0       # MAX EDGE 定格倒计时（帧数）
            prev_best_edge = None       # 上一帧记录的最大值，用于触发定格

            # ── HUD 数值平滑状态（仅影响叠加显示，不改变 CSV 原始数据）─────────────
            SMOOTH_ALPHA = 0.3
            MISSING_TOL_FRAMES = max(10, int(fps * 0.3))  # 约 0.3 秒的缺失容忍度
            smooth_knee = None
            smooth_lean = None
            smooth_center = None
            smooth_similarity = None
            smooth_champion = None
            smooth_edge = None
            miss_knee = 0
            miss_lean = 0
            miss_center = 0
            miss_similarity = 0
            miss_champion = 0
            miss_edge = 0

            # ── HUD 文案缓存（上一帧的文本），保证 HUD 持续存在 ───────────────
            hud_inclination = "內傾角 (Inclination): N/A"
            hud_core_stability = "躯幹安定度 (Core Stability): N/A"
            hud_com_height = "垂直壓強 (COM Height): N/A"
            hud_champion = "冠军匹配度: N/A"
            hud_vs_champion = "VS 冠军: N/A"

            def _update_smooth(current, smooth_value, miss_count):
                """
                简单一阶低通滤波 + 缺失容忍：
                  - 有新值时做 EMA 平滑；
                  - 连续缺失若干帧后才回落为 None，避免 N/A 闪烁。
                """
                if current is not None:
                    if smooth_value is None:
                        smooth_value = current
                    else:
                        smooth_value = smooth_value * (1.0 - SMOOTH_ALPHA) + current * SMOOTH_ALPHA
                    miss_count = 0
                else:
                    miss_count += 1
                    if miss_count > MISSING_TOL_FRAMES:
                        smooth_value = None
                return smooth_value, miss_count
            # 当前转弯周期的峰值追踪（转向改变时才锁定上显）
            _cur_turn_dir = 0           # 当前转弯方向
            _cur_turn_max_edge = 0.0    # 当前转弯内的最大立刃角（实时累积）
            _locked_turn_max_edge = None  # 已锁定的上一转弯峰值（稳定显示）

            # 在主循环外一次性加载冠军参考图（champion_reference.png 或 champion_pose.png）
            champ_ref_path = os.path.join(os.path.dirname(__file__), "champion_reference.png")
            if not os.path.exists(champ_ref_path):
                champ_ref_path = CHAMPION_IMAGE_PATH
            champ_ref_img = cv2.imread(champ_ref_path) if os.path.exists(champ_ref_path) else None

            # ── 一次性推理冠军图，提取真实基准角度 ───────────────────────────────
            champ_knee_deg = BASE_KNEE_ANGLE_DEG   # 默认回退到全局常量
            champ_lean_deg = BASE_LEAN_ANGLE_DEG
            if champ_ref_img is not None:
                try:
                    _ch_rgb = cv2.cvtColor(champ_ref_img, cv2.COLOR_BGR2RGB)
                    if MP_BACKEND == "tasks_v2":
                        # 用独立的 IMAGE 模式 Landmarker 推理静态图，避免污染视频流时间戳
                        from mediapipe.tasks.python import vision as _mp_vis
                        _img_options = _mp_vis.PoseLandmarkerOptions(
                            base_options=mp_tasks.BaseOptions(
                                model_asset_path=POSE_LANDMARKER_MODEL_PATH
                            ),
                            running_mode=_mp_vis.RunningMode.IMAGE,
                        )
                        with _mp_vis.PoseLandmarker.create_from_options(_img_options) as _img_lmk:
                            _ch_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=_ch_rgb)
                            _ch_res = _img_lmk.detect(_ch_mp)
                        _ch_lm = _ch_res.pose_landmarks[0] if _ch_res.pose_landmarks else None
                    else:
                        _ch_res = pose.process(_ch_rgb)
                        _ch_lm = _ch_res.pose_landmarks.landmark if _ch_res.pose_landmarks else None

                    if _ch_lm is not None:
                        _lp = PoseLandmarkEnum
                        _h, _w = champ_ref_img.shape[:2]
                        _lhip = (_ch_lm[_lp.LEFT_HIP].x * _w, _ch_lm[_lp.LEFT_HIP].y * _h)
                        _lknee = (_ch_lm[_lp.LEFT_KNEE].x * _w, _ch_lm[_lp.LEFT_KNEE].y * _h)
                        _lankle = (_ch_lm[_lp.LEFT_ANKLE].x * _w, _ch_lm[_lp.LEFT_ANKLE].y * _h)
                        _rhip = (_ch_lm[_lp.RIGHT_HIP].x * _w, _ch_lm[_lp.RIGHT_HIP].y * _h)
                        _rknee = (_ch_lm[_lp.RIGHT_KNEE].x * _w, _ch_lm[_lp.RIGHT_KNEE].y * _h)
                        _rankle = (_ch_lm[_lp.RIGHT_ANKLE].x * _w, _ch_lm[_lp.RIGHT_ANKLE].y * _h)
                        _lk = angle_between_three_points(_lhip, _lknee, _lankle)
                        _rk = angle_between_three_points(_rhip, _rknee, _rankle)
                        _kas = [a for a in (_lk, _rk) if a is not None]
                        if _kas:
                            champ_knee_deg = sum(_kas) / len(_kas)

                        _lsh = (_ch_lm[_lp.LEFT_SHOULDER].x * _w, _ch_lm[_lp.LEFT_SHOULDER].y * _h)
                        _rsh = (_ch_lm[_lp.RIGHT_SHOULDER].x * _w, _ch_lm[_lp.RIGHT_SHOULDER].y * _h)
                        _sc = ((_lsh[0] + _rsh[0]) / 2, (_lsh[1] + _rsh[1]) / 2)
                        _hc = ((_lhip[0] + _rhip[0]) / 2, (_lhip[1] + _rhip[1]) / 2)
                        _la = compute_lean_angle(_sc, _hc)
                        if _la is not None:
                            champ_lean_deg = _la
                        print(f"冠军图推理完成 → 膝盖角度: {champ_knee_deg:.1f}°  躯干倾角: {champ_lean_deg:.1f}°")
                except Exception as _e:
                    print(f"冠军图推理失败，使用默认基准值: {_e}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    # 读取失败（可能到达视频末尾），结束当前视频处理
                    break

                # 简单帧计数打印，帮助排查 cap.read() 是否卡在死循环
                debug_frame_counter += 1
                if debug_frame_counter % 300 == 0:
                    print(f"[debug] {video_name} 已处理帧数: {debug_frame_counter} / 约 {total_frames}")

                
                # 在任何绘制之前保存原始干净帧（用于报告封面截图）
                frame_clean = frame.copy()

                # OpenCV 默认是 BGR 排列，而 MediaPipe 要求输入为 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # ==========================
                # 姿态识别（Tasks V2 或 Solutions V1）
                # ==========================
                if MP_BACKEND == "tasks_v2":
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=frame_rgb,
                    )
                    # 使用全局偏移量，确保在多视频场景下 timestamp_ms 始终单调递增
                    timestamp_ms = int(frame_index * 1000.0 / fps) + ts_offset_ms
                    result = pose.detect_for_video(mp_image, timestamp_ms=timestamp_ms)

                    # Tasks V2: result.pose_landmarks 是 [ [landmark, ...], ... ]
                    current_landmarks = (
                        result.pose_landmarks[0] if result.pose_landmarks else None
                    )
                else:
                    # Solutions V1: pose.process 返回的 pose_landmarks 是 proto，内部有 .landmark 列表
                    result = pose.process(frame_rgb)
                    if result.pose_landmarks:
                        current_landmarks = result.pose_landmarks.landmark
                    else:
                        current_landmarks = None

                # 同时提取 world_landmarks（3D 真实物理坐标）
                if MP_BACKEND == "tasks_v2":
                    current_world_lm = (
                        result.pose_world_landmarks[0]
                        if result.pose_world_landmarks else None
                    )
                else:
                    current_world_lm = (
                        result.pose_world_landmarks.landmark
                        if result.pose_world_landmarks else None
                    )

                # 利用前几帧进行插值补全
                if current_landmarks is None and last_landmarks is not None:
                    landmarks = last_landmarks
                else:
                    landmarks = current_landmarks

                if current_world_lm is None:
                    current_world_lm = getattr(process_videos, '_last_world_lm', None)
                else:
                    process_videos._last_world_lm = current_world_lm
                world_lm = current_world_lm

                if landmarks:
                    last_landmarks = landmarks

                # 默认先设置一些变量为 None，若本帧检测不到人体，则保持 None
                knee_angle_avg = None
                lean_angle = None
                center_height = None
                knee_diff = None
                lean_diff = None
                similarity_score = None
                edge_angle = None
                hip_center = None
                ankle_center = None

                # 仅当检测到人体（或使用前几帧插值后）时才进行后续处理
                if landmarks:
                    # 在画面上绘制骨骼（火柴人效果）
                    if MP_BACKEND == "tasks_v2":
                        # V2 直接使用 landmarks（Python list）
                        drawing_utils_module.draw_landmarks(
                            frame,
                            landmarks,
                            POSE_CONNECTIONS,
                            landmark_drawing_spec=DrawingSpec(
                                color=(255, 255, 0), thickness=4, circle_radius=4
                            ),
                            connection_drawing_spec=DrawingSpec(
                                color=(255, 255, 0), thickness=4, circle_radius=4
                            ),
                        )
                    else:
                        # V1 需要传入 proto 对象（result.pose_landmarks）
                        drawing_utils_module.draw_landmarks(
                            frame,
                            result.pose_landmarks,
                            POSE_CONNECTIONS,
                            landmark_drawing_spec=DrawingSpec(
                                color=(255, 255, 0), thickness=4, circle_radius=4
                            ),
                            connection_drawing_spec=DrawingSpec(
                                color=(255, 255, 0), thickness=4, circle_radius=4
                            ),
                        )

                    # === 1. 膝关节角度（优先 3D world_landmarks）===
                    lp = PoseLandmarkEnum

                    try:
                        if world_lm is not None:
                            lhip_3 = (world_lm[lp.LEFT_HIP].x, world_lm[lp.LEFT_HIP].y, world_lm[lp.LEFT_HIP].z)
                            lknee_3 = (world_lm[lp.LEFT_KNEE].x, world_lm[lp.LEFT_KNEE].y, world_lm[lp.LEFT_KNEE].z)
                            lank_3 = (world_lm[lp.LEFT_ANKLE].x, world_lm[lp.LEFT_ANKLE].y, world_lm[lp.LEFT_ANKLE].z)
                            rhip_3 = (world_lm[lp.RIGHT_HIP].x, world_lm[lp.RIGHT_HIP].y, world_lm[lp.RIGHT_HIP].z)
                            rknee_3 = (world_lm[lp.RIGHT_KNEE].x, world_lm[lp.RIGHT_KNEE].y, world_lm[lp.RIGHT_KNEE].z)
                            rank_3 = (world_lm[lp.RIGHT_ANKLE].x, world_lm[lp.RIGHT_ANKLE].y, world_lm[lp.RIGHT_ANKLE].z)
                            left_knee_angle = angle_3d(lhip_3, lknee_3, lank_3)
                            right_knee_angle = angle_3d(rhip_3, rknee_3, rank_3)
                        else:
                            left_hip = landmarks[lp.LEFT_HIP]
                            left_knee = landmarks[lp.LEFT_KNEE]
                            left_ankle = landmarks[lp.LEFT_ANKLE]
                            right_hip = landmarks[lp.RIGHT_HIP]
                            right_knee = landmarks[lp.RIGHT_KNEE]
                            right_ankle = landmarks[lp.RIGHT_ANKLE]
                            lhip_xy = (left_hip.x * frame_width, left_hip.y * frame_height)
                            lknee_xy = (left_knee.x * frame_width, left_knee.y * frame_height)
                            lankle_xy = (left_ankle.x * frame_width, left_ankle.y * frame_height)
                            rhip_xy = (right_hip.x * frame_width, right_hip.y * frame_height)
                            rknee_xy = (right_knee.x * frame_width, right_knee.y * frame_height)
                            rankle_xy = (right_ankle.x * frame_width, right_ankle.y * frame_height)
                            left_knee_angle = angle_between_three_points(lhip_xy, lknee_xy, lankle_xy)
                            right_knee_angle = angle_between_three_points(rhip_xy, rknee_xy, rankle_xy)
                        knee_angles = [a for a in (left_knee_angle, right_knee_angle) if a is not None]
                        knee_angle_avg = sum(knee_angles) / len(knee_angles) if knee_angles else None
                    except (IndexError, Exception):
                        pass


                    # === 2. 躯干倾斜角 ===
                    hip_center = None
                    ankle_center = None
                    try:
                        left_shoulder = landmarks[lp.LEFT_SHOULDER]
                        right_shoulder = landmarks[lp.RIGHT_SHOULDER]
                        left_hip_lm = landmarks[lp.LEFT_HIP]
                        right_hip_lm = landmarks[lp.RIGHT_HIP]

                        # 计算双肩与双胯的中点像素坐标
                        shoulder_center = (
                            (left_shoulder.x + right_shoulder.x) / 2 * frame_width,
                            (left_shoulder.y + right_shoulder.y) / 2 * frame_height,
                        )
                        hip_center = (
                            (left_hip_lm.x + right_hip_lm.x) / 2 * frame_width,
                            (left_hip_lm.y + right_hip_lm.y) / 2 * frame_height,
                        )

                        lean_angle = compute_lean_angle(shoulder_center, hip_center)
                    except IndexError:
                        hip_center = None

                    # === 4. 对三个指标做移动平均滤波 ===
                    window = 12
                    if knee_angle_avg is not None or knee_hist:
                        knee_angle_avg = moving_average_update(
                            knee_hist, knee_angle_avg, window
                        )
                    if lean_angle is not None or lean_hist:
                        lean_angle = moving_average_update(
                            lean_hist, lean_angle, window
                        )
                    if center_height is not None or center_hist:
                        center_height = moving_average_update(
                            center_hist, center_height, window
                        )

                    # === 3. 重心高度（胯部中点相对于脚踝中点的高度比例） ===
                    try:
                        left_ankle_lm = landmarks[lp.LEFT_ANKLE]
                        right_ankle_lm = landmarks[lp.RIGHT_ANKLE]

                        ankle_center = (
                            (left_ankle_lm.x + right_ankle_lm.x) / 2 * frame_width,
                            (left_ankle_lm.y + right_ankle_lm.y) / 2 * frame_height,
                        )

                        if hip_center is not None:
                            center_height = compute_center_height(
                                hip_center, ankle_center, frame_height
                            )
                    except IndexError:
                        ankle_center = None

                    # === 6. 立刃角度（优先 3D world_landmarks，回退 2D）===
                    edge_angle = None
                    try:
                        if world_lm is not None:
                            # 3D 真实物理角度（消除透视畸变，上限 90°）
                            lk3 = (world_lm[lp.LEFT_KNEE].x, world_lm[lp.LEFT_KNEE].y, world_lm[lp.LEFT_KNEE].z)
                            la3 = (world_lm[lp.LEFT_ANKLE].x, world_lm[lp.LEFT_ANKLE].y, world_lm[lp.LEFT_ANKLE].z)
                            rk3 = (world_lm[lp.RIGHT_KNEE].x, world_lm[lp.RIGHT_KNEE].y, world_lm[lp.RIGHT_KNEE].z)
                            ra3 = (world_lm[lp.RIGHT_ANKLE].x, world_lm[lp.RIGHT_ANKLE].y, world_lm[lp.RIGHT_ANKLE].z)
                            ea_l = compute_edge_angle_3d(lk3, la3)
                            ea_r = compute_edge_angle_3d(rk3, ra3)
                        else:
                            # 回退 2D
                            lknee_e = (landmarks[lp.LEFT_KNEE].x * frame_width, landmarks[lp.LEFT_KNEE].y * frame_height)
                            lankle_e = (landmarks[lp.LEFT_ANKLE].x * frame_width, landmarks[lp.LEFT_ANKLE].y * frame_height)
                            rknee_e = (landmarks[lp.RIGHT_KNEE].x * frame_width, landmarks[lp.RIGHT_KNEE].y * frame_height)
                            rankle_e = (landmarks[lp.RIGHT_ANKLE].x * frame_width, landmarks[lp.RIGHT_ANKLE].y * frame_height)
                            ea_l = compute_edge_angle(lknee_e, lankle_e)
                            ea_r = compute_edge_angle(rknee_e, rankle_e)
                        # 取倾斜角较大的一侧（内侧腿），再做余角转换：
                        # final_edge_angle = 90 - tilt，小腿垂直→0°，极限倾斜→90°
                        _tilt = max(ea_l, ea_r)
                        edge_angle = min(90.0, max(0.0, 90.0 - abs(_tilt)))
                        edge_raw_hist.append(edge_angle)
                    except (IndexError, Exception):
                        edge_raw_hist.append(0.0)

                    edge_angle = moving_average_update(edge_hist, edge_angle, 5)

                    # ── 3D 转向判定（Hysteresis 保护，防止跳变）──────────
                    turn_dir, turn_pending = detect_turn_direction_3d(
                        world_lm, last_dir=turn_dir, pending_count=turn_pending
                    )
                    turn_dir_history.append(turn_dir)

                    # ── 立刃角有符号化：左弯取正，右弯取负，以保证曲线图始终向上增长 ──
                    # 绘图和记录仍用绝对值；有符号值仅供内部分析
                    if edge_angle is not None:
                        edge_angle_signed = edge_angle * (1 if turn_dir >= 0 else 1)
                        # edge_angle 本身保持绝对值（曲线图使用）
                        edge_angle = abs(edge_angle_signed)

                    # 收集用户立刃角序列（供后续 DTW 使用）
                    _user_edge_seq_main.append(edge_angle if edge_angle is not None else 0.0)

                    # 按转弯周期追踪峰值：转向改变时锁定上一弯的最大立刃角
                    if turn_dir != 0 and turn_dir != _cur_turn_dir:
                        # 转向切换：锁定上一弯峰值，重置当前弯计数器
                        if _cur_turn_max_edge > 0:
                            _locked_turn_max_edge = _cur_turn_max_edge
                        _cur_turn_dir = turn_dir
                        _cur_turn_max_edge = 0.0

                    if edge_angle is not None and edge_angle > _cur_turn_max_edge:
                        _cur_turn_max_edge = edge_angle

                    # 全局最大立刃记录（用于曲线图峰值标记和 MAX 标注）
                    if edge_angle is not None:
                        if best_edge_angle is None or edge_angle > best_edge_angle:
                            best_edge_angle = edge_angle
                            best_edge_frame_idx = frame_index

                    # 报告关键帧：锁定核心强度最低（立刃角最小）的换刃过渡时刻
                    # 仅在已有一定滑行数据后才开始判断（跳过视频开头静止段）
                    if edge_angle is not None and frame_index > 30:
                        if min_core_edge_angle is None or edge_angle < min_core_edge_angle:
                            min_core_edge_angle = edge_angle
                            # 同步锁定该帧的干净原始画面和对应数据
                            best_frame_for_report = frame_clean
                            best_knee_angle = knee_angle_avg
                            best_lean_angle = lean_angle
                            best_center_height = center_height
                            best_turn_dir = turn_dir

                    # 追踪髋部位置（用于判断转向）
                    if hip_center is not None:
                        hip_history.append(hip_center)
                        if len(hip_history) > 30:
                            hip_history.pop(0)
                        hip_history_full.append(hip_center)
                    else:
                        hip_history_full.append(None)

                    # === 后座判定（3D + 时间窗口防抖）===
                    _bs_raw, _bs_offset = detect_backseat_3d(
                        world_lm,
                        edge_angle=edge_angle if edge_angle is not None else 0.0,
                    )
                    backseat_buffer.append(_bs_raw)

                    _severe_ratio = backseat_buffer.count('severe') / max(len(backseat_buffer), 1)
                    _mild_ratio   = (backseat_buffer.count('severe') +
                                     backseat_buffer.count('mild')) / max(len(backseat_buffer), 1)

                    if _severe_ratio >= 0.7:
                        backseat_level = 'severe'
                    elif _mild_ratio >= 0.7:
                        backseat_level = 'mild'
                    else:
                        backseat_level = 'none'

                    # === 5A. 当前阶段标签（顶部居中，PIL 渲染）===
                    if edge_angle is not None:
                        _stage_label, _coaching = get_edge_coaching_text(edge_angle)
                    else:
                        _stage_label, _coaching = "检测中", ""

                    if turn_dir == -1:
                        _phase_text = f"[ 当前阶段：左弯卡宾 · {_stage_label} ]"
                        _phase_color = (220, 160, 60)   # 电光蓝（BGR）
                    elif turn_dir == 1:
                        _phase_text = f"[ 当前阶段：右弯卡宾 · {_stage_label} ]"
                        _phase_color = (80, 210, 80)   # 荧光紫（BGR）
                    else:
                        _phase_text = f"[ 换刃过渡 · {_stage_label} ]"
                        _phase_color = (180, 180, 180)

                    _ph_x = max(10, frame_width // 2 - len(_phase_text) * 7)
                    draw_text_pil(
                        frame, _phase_text, _ph_x, 10,
                        font_size=24, color=_phase_color, bg_alpha=0.55,
                    )

                    # === MAX EDGE ATTACK! 持续高亮（立刃角 > 60° 时显示）===
                    if edge_angle is not None and edge_angle > 60.0:
                        _attack_txt = f"MAX EDGE ATTACK!  {edge_angle:.0f}°"
                        _attack_x = max(10, frame_width // 2 - len(_attack_txt) * 14)
                        _attack_y = 44   # 紧跟阶段标签下方
                        draw_text_pil(
                            frame, _attack_txt, _attack_x, _attack_y,
                            font_size=32, color=(0, 255, 255), bg_alpha=0.65,
                        )

                    # === 5. 重心垂线 & 动态气泡标签 ===
                    if hip_center is not None:
                        hx = int(hip_center[0])
                        hy = int(hip_center[1])
                        # 重心垂线（红色）
                        cv2.line(
                            frame,
                            (hx, hy),
                            (hx, frame_height - 1),
                            (0, 0, 255),
                            2,
                        )
                        # 底端发光圆点（压强中心）
                        cv2.circle(
                            frame,
                            (hx, frame_height - 5),
                            10,
                            (0, 0, 255),
                            thickness=-1,
                        )

                        # 后座分级 UI 反馈（基于时间窗口防抖后的 backseat_level）
                        if backseat_level == 'severe':
                            draw_text_pil(
                                frame, "⚠ 警告：重心后坐，失去控制！",
                                max(10, hx - 200), max(10, hy - 240),
                                font_size=26, color=(0, 0, 255), bg_alpha=0.8,
                            )
                        elif backseat_level == 'mild':
                            draw_text_pil(
                                frame, "注意：重心前压",
                                max(10, hx - 140), max(10, hy - 200),
                                font_size=24, color=(0, 165, 255), bg_alpha=0.7,
                            )

                        # 重心正常时保留绿色"重心压强完美"提示
                        if center_height is not None and 0.25 <= center_height <= 0.45 and backseat_level == 'none':
                            draw_text_pil(
                                frame, "重心压强完美，保持刻滑姿态",
                                max(10, hx - 140), max(10, hy - 120),
                                font_size=22, color=(0, 255, 0), bg_alpha=0.55,
                            )

                        # ── 最佳匹配帧叠加：在右侧显示迄今匹配度最高的帧 ──
                        if best_frame_for_report is not None:
                            overlay_w = frame_width // 3
                            overlay_h = int(overlay_w * best_frame_for_report.shape[0]
                                            / max(best_frame_for_report.shape[1], 1))
                            overlay_h = min(overlay_h, frame_height)
                            best_small = cv2.resize(best_frame_for_report, (overlay_w, overlay_h))
                            ox = frame_width - overlay_w - 4
                            oy = max(0, min(frame_height - overlay_h, hy - overlay_h // 2))
                            roi = frame[oy:oy + overlay_h, ox:ox + overlay_w]
                            # 半透明叠加（55% 最佳帧 + 45% 当前帧背景）
                            cv2.addWeighted(best_small, 0.55, roi, 0.45, 0, dst=roi)
                            # 在叠加区左上角标注说明
                            draw_text_pil(
                                frame, "最佳匹配帧",
                                ox + 4, oy + 4,
                                font_size=18, color=(0, 255, 255), bg_alpha=0.6,
                            )

                    # === 4. 基准差值与相似度计算 ===
                    if knee_angle_avg is not None:
                        knee_diff = knee_angle_avg - BASE_KNEE_ANGLE_DEG
                        similarity_score = max(
                            0.0,
                            100.0
                            - abs(knee_diff) / max(BASE_KNEE_ANGLE_DEG, 1e-6) * 100.0,
                        )
                    if lean_angle is not None:
                        lean_diff = lean_angle - BASE_LEAN_ANGLE_DEG

                # =======================
                # 在画面上叠加文本（左上角）—— 使用 PIL 避免中文乱码
                # =======================
                overlay_lines = []

                def add_overlay_line(text, color):
                    overlay_lines.append((text, color))

                neon_green = (0, 255, 0)
                orange = (0, 140, 255)   # BGR = #FF8C00 亮橙
                red = (0, 0, 255)

                # ── Champion Match：与冠军图真实推理角度对比 ────────────────
                # 同时对比膝盖角度和躯干倾角，取加权平均相似度
                champion_match = None
                if knee_angle_avg is not None and lean_angle is not None:
                    knee_sim = max(0.0, 100.0 - abs(knee_angle_avg - champ_knee_deg)
                                   / max(champ_knee_deg, 1e-6) * 100.0)
                    lean_sim = max(0.0, 100.0 - abs(lean_angle - champ_lean_deg)
                                   / max(champ_lean_deg, 1e-6) * 100.0)
                    champion_match = knee_sim * 0.6 + lean_sim * 0.4
                    champion_match = max(0.0, min(100.0, champion_match))
                elif knee_angle_avg is not None:
                    knee_sim = max(0.0, 100.0 - abs(knee_angle_avg - champ_knee_deg)
                                   / max(champ_knee_deg, 1e-6) * 100.0)
                    champion_match = knee_sim
                # 两者皆缺失时保持 None，交由平滑逻辑处理显示

                # ── 数值平滑（仅影响 HUD，可视化更稳定）─────────────────────
                smooth_knee, miss_knee = _update_smooth(knee_angle_avg, smooth_knee, miss_knee)
                smooth_lean, miss_lean = _update_smooth(lean_angle, smooth_lean, miss_lean)
                smooth_center, miss_center = _update_smooth(center_height, smooth_center, miss_center)
                smooth_similarity, miss_similarity = _update_smooth(similarity_score, smooth_similarity, miss_similarity)
                smooth_champion, miss_champion = _update_smooth(champion_match, smooth_champion, miss_champion)
                smooth_edge, miss_edge = _update_smooth(edge_angle, smooth_edge, miss_edge)

                vis_knee_angle = smooth_knee
                vis_lean_angle = smooth_lean
                vis_center_height = smooth_center
                vis_similarity = smooth_similarity
                vis_champion = smooth_champion
                vis_edge_angle = smooth_edge

                # ── 更新 HUD 文案缓存：有新值时覆盖，缺失时保持上一帧 ─────────
                if vis_knee_angle is not None:
                    hud_inclination = f"內傾角 (Inclination): {vis_knee_angle:6.2f} deg"
                if vis_lean_angle is not None:
                    hud_core_stability = f"躯幹安定度 (Core Stability): {vis_lean_angle:6.2f} deg"
                if vis_center_height is not None:
                    hud_com_height = f"垂直壓強 (COM Height): {vis_center_height * 100:6.2f} %"
                if vis_champion is not None:
                    hud_champion = f"冠军匹配度: {vis_champion:5.1f} %"

                # VS 冠军立刃差：用平滑后的 edge 值，缺失时沿用上一帧缓存
                edge_for_vs = vis_edge_angle if vis_edge_angle is not None else edge_angle
                if champ_df is not None and edge_for_vs is not None:
                    _ci = frame_index % max(len(champ_df), 1)
                    _champ_ea = float(champ_df.iloc[_ci].get("edge_angle", 0) or 0)
                    _ea_diff = edge_for_vs - _champ_ea
                    _diff_sign = "+" if _ea_diff >= 0 else ""
                    if abs(_ea_diff) < 10:
                        _diff_col = neon_green
                    elif abs(_ea_diff) < 20:
                        _diff_col = orange
                    else:
                        _diff_col = red
                    hud_vs_champion = f"VS 冠军[{champ_name}]: {_diff_sign}{_ea_diff:.1f}° 立刃差"
                # 没有冠军数据或当前帧 edge 无法计算时，hud_vs_champion 保持上一帧

                # ── 将 HUD 文案缓存写入 overlay_lines（始终存在，不再闪烁）───────
                add_overlay_line(hud_inclination, neon_green)
                add_overlay_line(hud_core_stability, neon_green)
                add_overlay_line(hud_com_height, neon_green)
                add_overlay_line(hud_champion, neon_green)
                add_overlay_line(hud_vs_champion, neon_green)

                # #region agent log
                try:
                    import json as _dbg_json, time as _dbg_time
                    _dbg_payload = {
                        "sessionId": "72a529",
                        "id": f"log_{int(_dbg_time.time()*1000)}",
                        "timestamp": int(_dbg_time.time()*1000),
                        "location": "main.py:HUD",
                        "message": "hud_state",
                        "runId": "pre-fix",
                        "hypothesisId": "H1",
                        "data": {
                            "frame_index": int(frame_index),
                            "knee_angle_avg": knee_angle_avg,
                            "lean_angle": lean_angle,
                            "center_height": center_height,
                            "similarity_score": similarity_score,
                            "edge_angle": edge_angle,
                            "vis_knee_angle": vis_knee_angle,
                            "vis_lean_angle": vis_lean_angle,
                            "vis_center_height": vis_center_height,
                            "vis_similarity": vis_similarity,
                            "vis_champion": vis_champion,
                            "vis_edge_angle": vis_edge_angle,
                            "hud_inclination": hud_inclination,
                            "hud_core_stability": hud_core_stability,
                            "hud_com_height": hud_com_height,
                            "hud_champion": hud_champion,
                            "hud_vs_champion": hud_vs_champion,
                        },
                    }
                    if frame_index % 60 == 0:
                        with open("/Users/hejia/Desktop/Ski_AI/.cursor/debug-72a529.log", "a", encoding="utf-8") as _dbg_f:
                            _dbg_f.write(_dbg_json.dumps(_dbg_payload, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                # #endregion agent log

                # 记录"Champion Match 最大值"（仅追踪相似度最高值，不再覆盖报告帧）
                if best_champion_match is None or champion_match > best_champion_match:
                    best_champion_match = champion_match
                    best_similarity = similarity_score

                # 动态视觉反馈：基于膝盖角度与基准的差值
                if knee_diff is not None:
                    if knee_diff > 15.0:
                        add_overlay_line("重心過高，請下壓", red)
                    elif abs(knee_diff) <= 5.0:
                        add_overlay_line("內傾角優秀，請保持", neon_green)


                # ── 毛玻璃看板（Glassmorphism 圆角浮层）──────────────
                _PANEL_FONT   = 28   # 字体大小（像素）
                _PANEL_LHGT   = 40   # 每行行高（像素）
                _PANEL_PAD    = 14   # 内边距
                panel_w = 520
                panel_h = max(60, len(overlay_lines) * _PANEL_LHGT + _PANEL_PAD * 2)
                # 固定放置在画面左上角，避免与下方曲线区域重叠
                panel_x = 8
                panel_y = 8
                draw_glass_panel(frame, panel_x, panel_y, panel_w, panel_h, alpha=0.55, radius=20)
                # PIL 文字写在看板内，颜色映射 BGR→RGB
                for i, (text, bgr_color) in enumerate(overlay_lines):
                    draw_text_pil(frame, text,
                                  panel_x + _PANEL_PAD,
                                  panel_y + _PANEL_PAD + i * _PANEL_LHGT,
                                  font_size=_PANEL_FONT, color=bgr_color, bg_alpha=0.0)

                # ── 合力矢量线（Force Line Analysis）───────────────────
                edge_for_force = vis_edge_angle if 'vis_edge_angle' in locals() and vis_edge_angle is not None else edge_angle
                draw_force_vector(frame, hip_center, ankle_center, edge_for_force)

                # ── 全屏底部立刃角度曲线图 ─────────────────────────────
                # 冠军曲线：从 champ_df 取 edge_angle 列的滑动窗口（与 edge_hist 等长）
                _champ_curve_seg = None
                if champ_df is not None and "edge_angle" in champ_df.columns:
                    _ch_ea = champ_df["edge_angle"].fillna(0).values.tolist()
                    _seg_len = len(edge_hist)
                    _ch_start = max(0, frame_index - _seg_len)
                    _ch_end   = _ch_start + _seg_len
                    # 循环复用冠军序列
                    _ch_full = _ch_ea * ((_ch_end // max(len(_ch_ea), 1)) + 2)
                    _champ_curve_seg = _ch_full[_ch_start:_ch_start + _seg_len]

                draw_edge_curve(
                    frame, edge_hist,
                    hip_history=hip_history,
                    max_edge_frame_idx=best_edge_frame_idx,
                    max_edge_val=_locked_turn_max_edge,   # 已锁定的上一弯峰值，稳定不跳
                    max_flash_frames=0,                   # 不再用实时定格计时
                    fps=fps,
                    champ_curve=_champ_curve_seg,
                    turn_dir_hist=turn_dir_history,
                )

                # 将当前帧写入输出视频
                out_writer.write(frame)

                # 将当前帧的分析结果保存到列表中（即使某些值为 None，也记录下来）
                row = {
                    "video_name": video_name,
                    "frame_index": frame_index,
                    "time_sec": frame_index / fps,
                    "knee_angle_avg_deg": knee_angle_avg,
                    "lean_angle_deg": lean_angle,
                    "center_height_ratio": center_height,
                    "similarity_score": similarity_score,
                    "edge_angle_deg": edge_angle,
                }
                all_rows.append(row)

                # 在屏幕上展示当前处理后的画面（预览），按 'q' 可以提前结束所有视频处理
                if not headless:
                    cv2.imshow("Carving AI 分析预览（按 q 退出）", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        # 用户主动退出：释放当前视频，并跳出外层循环
                        cap.release()
                        out_writer.release()
                        cv2.destroyAllWindows()
                        # 写出已经收集的数据
                        if all_rows:
                            df = pd.DataFrame(all_rows)
                            df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
                            print(f"已提前保存分析数据到: {output_csv_path}")
                        print("用户按下 q，提前结束所有视频处理。")
                        return

                # 每 20 帧汇报一次进度（回调 + 可选 /results/{JOB_ID}/status.json）
                if frame_index % 20 == 0:
                    _report_progress(frame_index)

                frame_index += 1

            # 当前视频处理结束，释放资源
            cap.release()
            out_writer.release()
            print(f"视频处理完成，输出文件: {output_video_path}")
            _reencode_h264(output_video_path)
            _processed_video_pairs.append((video_path, output_video_path))

            # 对于 Tasks V2，当前视频已消费完所有帧，累加时间戳偏移，
            # 确保下一条视频的 timestamp_ms 起点大于上一条视频的最后一帧。
            if MP_BACKEND == "tasks_v2":
                try:
                    # 用实际处理过的帧数来估计时间跨度，避免 CAP_PROP_FRAME_COUNT 偏小导致时间戳回退
                    ts_offset_ms += int(frame_index * 1000.0 / fps)
                except Exception:
                    pass

            # ── DTW 冠军匹配（视频处理完成后，用完整序列做匹配）──────────
            if _csv_list and _user_edge_seq_main and champ_df is None:
                print("正在进行 DTW 冠军匹配…")
                champ_df, champ_name, champ_dtw_dist = load_best_champion_csv(
                    _user_edge_seq_main, data_dir=CHAMPION_DATA_DIR
                )
                if champ_df is not None:
                    print(f"最匹配冠军: {champ_name}  DTW距离: {champ_dtw_dist:.2f}")

            # 入弯阶段（前 1/3 帧）立刃角度，用于诊断逻辑
            entry_n = max(1, len(edge_raw_hist) // 3)
            entry_phase = edge_raw_hist[:entry_n] if edge_raw_hist else []

            # 左右弯统计（优先用 3D Hysteresis 判向结果，其次髋部轨迹，最后 lean 回退）
            _video_rows = [r for r in all_rows if r.get("video_name") == video_name]
            # turn_dir_history 与 all_rows 长度可能不一致（仅含当前视频帧），直接传 _video_rows 对齐
            _td_hist = turn_dir_history[-len(_video_rows):] if len(turn_dir_history) >= len(_video_rows) else None
            # 将 turn_dir_history 转为 lean_signs 格式（-1=左弯→正，+1=右弯→负，0→0）
            if _td_hist and len(_td_hist) == len(_video_rows):
                _lean_signs = [-float(d) for d in _td_hist]   # 左弯(dir=-1)→正值
            else:
                _lean_signs = [r.get("lean_angle_deg") or 0.0 for r in _video_rows]
            _lr_stats = analyze_lr_balance(_video_rows, hip_history_full, lean_signs=_lean_signs)
            if _lr_stats:
                print(
                    f"[左右弯统计] 约 {_lr_stats['total_turns']} 个弯 | "
                    f"左弯均值 {_lr_stats['left_avg_edge']}°  右弯均值 {_lr_stats['right_avg_edge']}°\n"
                    f"  诊断：{_lr_stats['diagnosis']}"
                )

            # 生成该视频的对比报告图
            if best_frame_for_report is not None:
                # 报告用完整序列，保证 edge_hist 与 turn_dir_history 长度对齐
                _report_edge_hist = list(_user_edge_seq_main)
                _report_td_hist   = list(turn_dir_history)
                # 裁剪到相同长度（以短者为准）
                _min_len = min(len(_report_edge_hist), len(_report_td_hist))
                _report_edge_hist = _report_edge_hist[:_min_len]
                _report_td_hist   = _report_td_hist[:_min_len]
                generate_comparison_report(
                    best_frame_for_report,
                    video_name,
                    knee_angle=best_knee_angle,
                    lean_angle=best_lean_angle,
                    center_height=best_center_height,
                    similarity_score=best_similarity,
                    champion_match=best_champion_match,
                    best_frame_bgr=best_frame_for_report,
                    edge_angle=best_edge_angle,
                    edge_hist=_report_edge_hist,
                    best_edge_frame_idx=best_edge_frame_idx,
                    entry_phase_angles=entry_phase,
                    output_dir=output_report_dir,
                    lr_stats=_lr_stats,
                    turn_dir=best_turn_dir,
                    turn_dir_hist=_report_td_hist,
                )

    # 所有视频处理完毕后，将数据写入 CSV
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"所有视频处理完成，分析数据已保存到: {output_csv_path}")

        # 生成教练诊断报告（Ski_Coach_Final.png）
        try:
            from generate_coach_report import generate_coach_report
            _lr_diff_val     = abs(_lr_stats.get("lr_diff", 0)) if _lr_stats else 0.0
            _edge_var_val    = float(np.std(edge_hist)) if edge_hist and len(edge_hist) > 1 else 0.0
            _sim_mean_val    = float(pd.DataFrame(all_rows)["similarity_score"].dropna().mean()) if all_rows else None
            generate_coach_report(
                csv_path=output_csv_path,
                key_frame_bgr=best_frame_for_report,
                output_dir=output_report_dir,
                edge_angle=best_edge_angle,
                lr_diff=_lr_diff_val,
                edge_variance=_edge_var_val,
                similarity_score=_sim_mean_val,
            )
        except Exception as _e:
            print(f"[CoachReport] 生成报告时出错（不影响主流程）：{_e}")

        # 生成拼接对比视频（原视频 + 骨骼视频 + 专业诊断报告）
        _report_png = os.path.join(output_report_dir, "Ski_Coach_Final.png")
        for _in_path, _proc_path in _processed_video_pairs:
            _vname = os.path.basename(_in_path)
            # 确保扩展名统一为 .mp4
            _base  = os.path.splitext(_vname)[0]
            _comp_out = os.path.join(output_report_dir, f"comparison_{_base}.mp4")
            try:
                create_comparison_video(
                    input_video_path=_in_path,
                    processed_video_path=_proc_path,
                    report_img_path=_report_png,
                    output_path=_comp_out,
                )
            except Exception as _ce:
                print(f"[CompVideo] 生成对比视频时出错（{_vname}）：{_ce}")
    else:
        print("未收集到任何有效帧数据（可能所有帧都未检测到人体）。")


def _process_single_video_run(
    video_path: str,
    base_runs_dir: str,
    headless: bool,
    idx: int | None = None,
    total: int | None = None,
) -> None:
    """单个视频的多 Runs 封装，供串行/并行两种模式复用。"""
    video_name = os.path.basename(video_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir_name = f"{ts}_{video_name}"
    run_dir = os.path.join(base_runs_dir, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    if idx is not None and total is not None:
        print(f"[{idx}/{total}] 正在分析 {video_name} → {run_dir}")
    else:
        print(f"正在分析 {video_name} → {run_dir}")

    # 为当前视频准备单独的临时输入目录，确保 process_videos 只看到这一条视频
    single_input_dir = os.path.join(run_dir, "input")
    if os.path.exists(single_input_dir):
        _clean_dir(single_input_dir)
    os.makedirs(single_input_dir, exist_ok=True)

    tmp_video_path = os.path.join(single_input_dir, video_name)
    if video_path != tmp_video_path:
        try:
            shutil.copy2(video_path, tmp_video_path)
        except Exception as _e:
            print(f"[multi-runs] 复制视频失败，跳过 {video_name}：{_e}")
            return

    # 复用现有单 runs 逻辑，但 runs_dir 换成当前 run 子目录
    process_videos(
        input_dir=single_input_dir,
        runs_dir=run_dir,
        headless=headless,
        progress_callback=None,
        clean_runs=False,
    )

    # 在 run_dir 内创建一套固定命名的别名文件，便于后续脚本使用
    processed_name = f"processed_{video_name}"
    processed_path = os.path.join(run_dir, processed_name)
    if os.path.exists(processed_path):
        skeleton_out = os.path.join(run_dir, "skeleton_video.mp4")
        try:
            os.replace(processed_path, skeleton_out)
        except Exception as _e:
            print(f"[multi-runs] 重命名骨骼视频失败（{video_name}）：{_e}")

    ski_report_src = os.path.join(run_dir, "Ski_Report_Final.jpg")
    if os.path.exists(ski_report_src):
        ski_report_out = os.path.join(run_dir, "ski_report.jpg")
        try:
            os.replace(ski_report_src, ski_report_out)
        except Exception as _e:
            print(f"[multi-runs] 重命名战力报告失败（{video_name}）：{_e}")

    coach_report_src = os.path.join(run_dir, "Ski_Coach_Final.png")
    if os.path.exists(coach_report_src):
        coach_report_out = os.path.join(run_dir, "coach_report.png")
        try:
            os.replace(coach_report_src, coach_report_out)
        except Exception as _e:
            print(f"[multi-runs] 重命名教练报告失败（{video_name}）：{_e}")

    # analysis.csv 由内部 process_videos 直接写入 run_dir，命名与新约定一致
    csv_path = os.path.join(run_dir, "analysis.csv")
    if not os.path.exists(csv_path):
        print(f"[multi-runs] 注意：未在 {run_dir} 中找到 analysis.csv。")


def process_videos_multi_runs(
    input_dir: str = "./input_videos",
    base_runs_dir: str = "./runs",
    headless: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
) -> None:
    """
    本地批量分析入口：遍历 input_dir 下的所有视频，
    为每个视频创建独立的 runs/YYYYMMDD_HHMM_文件名/ 子目录，
    并在子目录内生成：
        - skeleton_video.mp4
        - ski_report.jpg
        - analysis.csv
        - coach_report.png

    为了最大限度复用现有逻辑，每个视频内部仍调用一次原始 process_videos，
    只是将 runs_dir 指向该视频独立的 run 子目录。
    """
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(base_runs_dir, exist_ok=True)
    # 每次批量运行前清空 runs 根目录，避免残留历史结果
    _clean_dir(base_runs_dir)

    video_paths = sorted(
        glob(os.path.join(input_dir, "*.mp4"))
        + glob(os.path.join(input_dir, "*.MP4"))
        + glob(os.path.join(input_dir, "*.mov"))
        + glob(os.path.join(input_dir, "*.MOV"))
    )
    if not video_paths:
        print("没有在指定 input_dir 中找到任何视频文件（mp4/MP4/mov/MOV）。")
        return

    total = len(video_paths)
    print(f"检测到 {total} 个视频，将为每个视频创建独立 runs 子目录进行分析…")

    if not parallel or total == 1:
        # 串行模式：逐个视频处理
        for idx, video_path in enumerate(video_paths, start=1):
            _process_single_video_run(
                video_path=video_path,
                base_runs_dir=base_runs_dir,
                headless=headless,
                idx=idx,
                total=total,
            )
    else:
        # 并行模式：多进程同时跑多个视频（共享同一块 GPU 时由驱动调度）
        # 默认并发数不再限制为 2，而是按 CPU 核心数和视频数量动态决定
        if max_workers is None:
            try:
                max_workers = min(cpu_count(), total)
            except Exception:
                max_workers = total
        if max_workers < 1:
            max_workers = 1

        print(f"[multi-runs] 启用并行模式，最大并行进程数 = {max_workers}")

        active_procs: list[Process] = []

        def _drain_active():
            nonlocal active_procs
            for p in active_procs:
                p.join()
            active_procs = []

        for idx, video_path in enumerate(video_paths, start=1):
            p = Process(
                target=_process_single_video_run,
                args=(video_path, base_runs_dir, headless, idx, total),
            )
            p.start()
            active_procs.append(p)

            if len(active_procs) >= max_workers:
                _drain_active()

        if active_procs:
            _drain_active()

    print("全部视频的本地多 Runs 分析已完成。")


if __name__ == "__main__":
    """
    双模系统入口：

    【冠军录入模式】
        1. 将冠军视频（mp4）放入 ./champion_videos/ 文件夹
        2. 运行：python main.py
        3. 程序自动提取 3D 骨骼数据，保存到 ./champion_data/champion_data_<name>.csv
        4. 每个视频只处理一次（已存在 CSV 则跳过）

    【分析模式（交互）】
        python main.py

    【分析模式（无头，由 notify_server 触发）】
        python main.py --headless [--input_dir ./input_videos] [--runs_dir ./runs]

    【依赖提示】
        pip install fastdtw scipy   # DTW 加速库（可选，无则自动降级为纯 numpy DTW）
    """
    import argparse

    parser = argparse.ArgumentParser(description="Ski Pro AI — 滑雪姿态分析")
    parser.add_argument("--headless",   action="store_true",
                        help="无头模式，不弹出 cv2 预览窗口（供服务端调用）")
    parser.add_argument("--input_dir",  default="./input_videos",
                        help="用户视频输入目录（默认 ./input_videos）")
    parser.add_argument("--runs_dir",   default="./runs",
                        help="输出目录（默认 ./runs）")
    args = parser.parse_args()

    # 检查 fastdtw 是否可用
    try:
        import fastdtw as _  # noqa
    except ImportError:
        print("[提示] 未检测到 fastdtw，将使用内置纯 numpy DTW（速度较慢）。")
        print("       建议安装：pip install fastdtw scipy")

    # 自动判断模式
    os.makedirs(CHAMPION_VIDEO_DIR, exist_ok=True)
    champ_vids = sorted(
        glob(os.path.join(CHAMPION_VIDEO_DIR, "*.mp4")) +
        glob(os.path.join(CHAMPION_VIDEO_DIR, "*.MP4")) +
        glob(os.path.join(CHAMPION_VIDEO_DIR, "*.mov")) +
        glob(os.path.join(CHAMPION_VIDEO_DIR, "*.MOV"))
    )
    input_vids = sorted(
        glob(os.path.join(args.input_dir, "*.mp4")) +
        glob(os.path.join(args.input_dir, "*.MP4")) +
        glob(os.path.join(args.input_dir, "*.mov")) +
        glob(os.path.join(args.input_dir, "*.MOV"))
    )

    if champ_vids:
        print(f"检测到 champion_videos/ 中有 {len(champ_vids)} 个视频，进入【冠军录入模式】")
        profile_champion_videos()
    elif input_vids:
        print(f"检测到 input_videos/ 中有 {len(input_vids)} 个视频，进入【分析模式】"
              f"{'（无头）' if args.headless else ''}")
        process_videos_multi_runs(
            input_dir=args.input_dir,
            base_runs_dir=args.runs_dir,
            headless=args.headless,
        )
    else:
        print("未找到任何视频。")
        print("  · 冠军录入：将 mp4 放入 ./champion_videos/，再运行 python main.py")
        print(f"  · 用户分析：将 mp4 放入 {args.input_dir}，再运行 python main.py")
