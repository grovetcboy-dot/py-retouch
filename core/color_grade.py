"""色彩控制：色温、色调、HSL 分色调整、RGB 通道曲线、3x3 色彩矩阵。"""
import numpy as np
from . import curves


def temperature_tint(img, temp=0.0, tint=0.0):
    """色温 (-1 冷..1 暖) 与色调 (-1 绿..1 品红)，用线性增益近似。"""
    if temp == 0 and tint == 0:
        return img
    r_gain = 1.0 + 0.25 * temp + 0.10 * tint
    g_gain = 1.0 - 0.10 * tint
    b_gain = 1.0 - 0.25 * temp + 0.10 * tint
    gain = np.array([r_gain, g_gain, b_gain], np.float32)
    # 归一化保持整体亮度
    gain *= (1.0 / (gain @ np.array([0.2126, 0.7152, 0.0722], np.float32)))
    return np.clip(img * gain, 0, 1)


def _rgb_to_hsv_rgb(img):
    """cv2 HSV 但保持 RGB 输入输出（内部转 BGR）。"""
    import cv2
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)


def hsl_adjust(img, hue_shifts=None, sat_adjusts=None, lum_adjusts=None):
    """按色相分区调整。hue_shifts/sat_adjusts/lum_adjusts: dict，键为
    red/orange/yellow/green/aqua/blue/purple/magenta，值为 -1..1。
    hue_shift 为色相旋转(归一化)，sat/lum 为乘性/加性调整。"""
    if not (hue_shifts or sat_adjusts or lum_adjusts):
        return img
    import cv2
    hue_shifts = hue_shifts or {}
    sat_adjusts = sat_adjusts or {}
    lum_adjusts = lum_adjusts or {}
    zones = {"red": (-15, 15), "orange": (15, 45), "yellow": (45, 70),
             "green": (70, 160), "aqua": (160, 200), "blue": (200, 260),
             "purple": (260, 290), "magenta": (290, 345)}
    hsv = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h_out, s_out, v_out = h.copy(), s.copy(), v.copy()
    # 红色跨 0 分两个区间
    zone_masks = {}
    for name, (lo, hi) in zones.items():
        if lo < 0:
            m = (h >= 360 + lo) | (h < hi)
        elif hi > 360:
            m = (h >= lo) | (h < hi - 360)
        else:
            m = (h >= lo) & (h < hi)
        zone_masks[name] = m
    # 区间边界软化权重：在边界 ±10 度内线性过渡
    def soft(name):
        lo, hi = zones[name]
        base = zone_masks[name]
        w = base.astype(np.float32)
        for edge in (lo, hi):
            dist = np.abs(h - edge)
            dist = np.minimum(dist, 360 - dist)
            near = (dist < 12) & ~base
            w[near] = np.maximum(w[near], 1 - dist[near] / 12)
        return w
    for name, shift in hue_shifts.items():
        if name in zone_masks and shift:
            w = soft(name)
            h_out = (h_out + shift * 30.0 * w) % 360
    for name, adj in sat_adjusts.items():
        if name in zone_masks and adj:
            w = soft(name)
            target = np.clip(s * (1 + adj), 0, 255)
            s_out = s_out * (1 - w) + target * w
    for name, adj in lum_adjusts.items():
        if name in zone_masks and adj:
            w = soft(name)
            target = np.clip(v + adj * 80, 0, 255)
            v_out = v_out * (1 - w) + target * w
    hsv_out = np.stack([h_out, np.clip(s_out, 0, 255), np.clip(v_out, 0, 255)], axis=-1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB)
    return rgb.astype(np.float32) / 255.0


def rgb_curves(img, master=None, r=None, g=None, b=None):
    """RGB 通道曲线，传入 curves 模块生成的 LUT。"""
    return curves.apply_channel_luts(img, r=r, g=g, b=b, master=master)


def color_matrix(img, matrix=None):
    """3x3 色彩矩阵（作用于线性域）。matrix: 3x3 array-like。"""
    if matrix is None:
        return img
    m = np.asarray(matrix, np.float32)
    return np.clip(img @ m.T, 0, 1)


def split_tone(img, shadow_color=None, highlight_color=None, balance=0.0):
    """分离色调。shadow/highlight_color: (r,g,b) 0-255 或 0-1；
    balance -1 偏阴影..1 偏高光。"""
    if shadow_color is None and highlight_color is None:
        return img
    lum = img.mean(axis=2, keepdims=True)
    hw = np.clip((lum + balance / 2) ** 2, 0, 1)
    sw = 1 - hw
    out = img.copy()
    if shadow_color is not None:
        c = np.asarray(shadow_color, np.float32)
        c = c / 255.0 if c.max() > 1.5 else c
        out = out + sw * c * 0.35
    if highlight_color is not None:
        c = np.asarray(highlight_color, np.float32)
        c = c / 255.0 if c.max() > 1.5 else c
        out = out + hw * c * 0.35
    return np.clip(out, 0, 1)


def apply(img, params):
    """按 dict 应用色彩控制。键: temp, tint, hsl(dict), rgb_curve_r/g/b/master,
    color_matrix, split_shadow, split_highlight, split_balance。"""
    out = img
    out = temperature_tint(out, params.get("temp", 0.0), params.get("tint", 0.0))
    hsl = params.get("hsl")
    if hsl:
        out = hsl_adjust(out, hsl.get("hue"), hsl.get("sat"), hsl.get("lum"))
    cr, cg, cb, cm = (params.get("rgb_curve_" + k) for k in ("r", "g", "b", "master"))
    if any(v is not None for v in (cr, cg, cb, cm)):
        luts = {k: (curves.curve_from_points(v) if v else None)
                for k, v in zip("rgbm", (cr, cg, cb, cm))}
        out = rgb_curves(out, master=luts["m"], r=luts["r"], g=luts["g"], b=luts["b"])
    if params.get("color_matrix"):
        out = color_matrix(out, params["color_matrix"])
    if params.get("split_shadow") or params.get("split_highlight"):
        out = split_tone(out, params.get("split_shadow"), params.get("split_highlight"),
                         params.get("split_balance", 0.0))
    return out
