"""基础调色：曝光、亮度、对比度、高光、阴影、黑白场、饱和度。
全部向量化 NumPy 操作，输入输出均为 float32 RGB [0,1]。"""
import numpy as np
from . import curves


def exposure(img, stops=0.0):
    """以 EV 为单位的曝光调整。"""
    if stops == 0:
        return img
    return np.clip(img * (2.0 ** stops), 0, 1)


def brightness(img, amount=0.0):
    """加性亮度，-1..1。"""
    if amount == 0:
        return img
    return np.clip(img + amount, 0, 1)


def contrast(img, amount=0.0, pivot=0.5):
    """amount 正为增强（-1..1 映射为系数 0..2）。"""
    if amount == 0:
        return img
    return np.clip(pivot + (img - pivot) * (1.0 + amount), 0, 1)


def highlights_shadows(img, highlights=0.0, shadows=0.0, width=0.35):
    """高光/阴影恢复。highlights 负为压暗高光，shadows 正为提亮阴影。"""
    if highlights == 0 and shadows == 0:
        return img
    lum = img.mean(axis=2, keepdims=True)
    sw = np.clip(1 - lum / width, 0, 1) ** 1.5      # 阴影权重
    hw = np.clip((lum - (1 - width)) / width, 0, 1) ** 1.5  # 高光权重
    out = img + shadows * 0.5 * sw + highlights * 0.5 * hw
    return np.clip(out, 0, 1)


def black_white_point(img, black=0.0, white=1.0):
    """黑场/白场。black>0 抬黑（褪色），white<1 压白。"""
    if black == 0 and white == 1:
        return img
    return np.clip(black + (white - black) * img, 0, 1)


def saturation(img, amount=0.0):
    """amount -1..1：负为降饱和，正为加饱和。"""
    if amount == 0:
        return img
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lum = lum[..., None]
    if amount >= 0:
        k = 1.0 + amount * 2.0
        out = lum + (img - lum) * k
    else:
        k = 1.0 + amount
        out = lum + (img - lum) * k
    return np.clip(out, 0, 1)


def vibrance(img, amount=0.0):
    """自然饱和度：只增强低饱和像素，肤色(低饱和橙)保护。"""
    if amount == 0:
        return img
    mx = img.max(axis=2, keepdims=True)
    mn = img.min(axis=2, keepdims=True)
    sat = mx - mn
    # 低饱和权重：越灰增强越多
    w = (1.0 - sat) * np.abs(amount)
    k = 1.0 + w * 2.5
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    out = lum[..., None] + (img - lum[..., None]) * k
    return np.clip(out, 0, 1)


def clarity(img, amount=0.0):
    """清晰度：中频对比（unsharp mask 大半径低强度版）。"""
    if amount == 0:
        return img
    import cv2
    lum = (img.mean(axis=2) * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(lum, (0, 0), sigmaX=15)
    mid = (lum.astype(np.float32) - blur.astype(np.float32)) / 255.0
    return np.clip(img + mid[..., None] * amount * 1.5, 0, 1)


def apply(img, params):
    """按 dict 应用基础调色。键: exposure, brightness, contrast,
    highlights, shadows, black_point, white_point, saturation, vibrance, clarity。"""
    out = img
    out = exposure(out, params.get("exposure", 0.0))
    out = brightness(out, params.get("brightness", 0.0))
    out = contrast(out, params.get("contrast", 0.0))
    out = highlights_shadows(out, params.get("highlights", 0.0), params.get("shadows", 0.0))
    out = black_white_point(out, params.get("black_point", 0.0), params.get("white_point", 1.0))
    out = saturation(out, params.get("saturation", 0.0))
    out = vibrance(out, params.get("vibrance", 0.0))
    out = clarity(out, params.get("clarity", 0.0))
    return out
