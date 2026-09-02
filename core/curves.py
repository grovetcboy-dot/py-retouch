"""曲线系统：控制点 -> 单调样条 -> 256 级 LUT。

所有曲线操作最终都归约为对每个 RGB 通道查一条 256 项的 LUT，
一次 cv2.LUT 即可完成，性能好且可叠加（LUT 复合）。
"""
import numpy as np

N = 256


def _monotonic_pchip(x, y):
    """保单调性的分段三次插值（PCHIP 简化实现），避免样条过冲。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    dx = np.diff(x)
    slope = np.diff(y) / np.where(dx == 0, 1, dx)
    # 端点斜率
    d = np.zeros(n)
    if n == 2:
        d[:] = slope[0]
    else:
        for i in range(1, n - 1):
            if slope[i - 1] * slope[i] <= 0:
                d[i] = 0.0
            else:
                w1 = 2 * dx[i] + dx[i - 1]
                w2 = dx[i] + 2 * dx[i - 1]
                d[i] = (w1 + w2) / (w1 / slope[i - 1] + w2 / slope[i])
        d[0] = ((2 * dx[0] + dx[1]) * slope[0] - dx[0] * slope[1]) / (dx[0] + dx[1]) \
            if dx[0] + dx[1] > 0 else 0.0
        d[-1] = ((2 * dx[-1] + dx[-2]) * slope[-1] - dx[-1] * slope[-2]) / (dx[-1] + dx[-2]) \
            if dx[-1] + dx[-2] > 0 else 0.0
    # 插值到 0..255
    xs = np.arange(N, dtype=float) * (255.0 / (N - 1))
    ys = np.interp(xs, x, y)  # 先线性兜底
    idx = np.clip(np.searchsorted(x, xs) - 1, 0, n - 2)
    for k in np.unique(idx):
        m = idx == k
        h, t = x[k + 1] - x[k], (xs[m] - x[k]) / max(x[k + 1] - x[k], 1e-9)
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        ys[m] = h00 * y[k] + h10 * h * d[k] + h01 * y[k + 1] + h11 * h * d[k + 1]
    return np.clip(ys, 0, 255)


def curve_from_points(points):
    """points: [(x, y), ...] x,y 均为 0-255。返回 256 项 LUT (uint8 索引值数组, 0-255)。"""
    pts = sorted(points)
    xs = [p[0] for p in pts] + [255] if pts and pts[-1][0] != 255 else [p[0] for p in pts]
    ys = [p[1] for p in pts] + [255] if pts and pts[-1][0] != 255 else [p[1] for p in pts]
    if pts and pts[0][0] != 0:
        xs, ys = [0] + xs, [pts[0][1]] + ys
    if len(xs) < 2:
        return np.arange(N, dtype=np.float32)
    return _monotonic_pchip(xs, ys).astype(np.float32)


def identity():
    return np.arange(N, dtype=np.float32)


def gamma_curve(gamma):
    """gamma > 1 提亮中调，< 1 压暗。"""
    x = np.arange(N) / 255.0
    return (np.power(x, 1.0 / gamma) * 255.0).astype(np.float32)


def s_curve(strength=1.0, pivot=0.5):
    """经典 S 曲线：压暗阴影、提亮高光、增强对比。strength 0-1，pivot 为支点。"""
    x = np.arange(N) / 255.0
    if strength == 0:
        y = x
    else:
        k = 2.0 + 5.0 * strength  # sigmoid 陡度
        y = pivot + np.tanh((x - pivot) * k) / np.tanh(k)
        # tanh 端点渐近不到 ±1，把端点重新钉回 0/1
        lo = pivot - 1.0 / np.tanh(k)
        hi = pivot + 1.0 / np.tanh(k)
        y = (y - lo) / max(hi - lo, 1e-9)
    return (np.clip(y, 0, 1) * 255.0).astype(np.float32)


def contrast_curve(amount):
    """以 0.5 为支点的对比曲线，amount 正为增强。"""
    x = np.arange(N) / 255.0
    y = 0.5 + (x - 0.5) * (1 + amount)
    return (np.clip(y, 0, 1) * 255.0).astype(np.float32)


def lift_gain_gamma(lift=0.0, gain=1.0, gamma=1.0):
    """阴影 lift、高光 gain、中调 gamma（0-1 域）。"""
    x = np.arange(N) / 255.0
    y = lift + (1 - lift) * x
    y = np.power(np.clip(y, 0, 1), 1.0 / gamma) * gain
    return (np.clip(y, 0, 1) * 255.0).astype(np.float32)


def shadow_highlight_curve(shadows=0.0, highlights=0.0, width=0.35):
    """只调整阴影或高光区域，中调尽量不动。
    shadows/highlights: -1..1，正为提亮/压暗。"""
    x = np.arange(N) / 255.0
    y = x.copy()
    if shadows:
        w = np.clip(1 - x / max(width, 1e-6), 0, 1)  # 阴影权重
        y = y + shadows * 0.4 * w
    if highlights:
        w = np.clip((x - (1 - width)) / max(width, 1e-6), 0, 1)  # 高光权重
        y = y - highlights * 0.4 * w
    return (np.clip(y, 0, 1) * 255.0).astype(np.float32)


def tone_curve(black=0.0, white=1.0):
    """黑白场：black 抬起（褪色感），white 压低。0-1 域。"""
    x = np.arange(N) / 255.0
    y = black + (white - black) * x
    return (np.clip(y, 0, 1) * 255.0).astype(np.float32)


def composite(*luts):
    """复合多条 LUT（依次应用），返回一条。"""
    out = luts[0]
    for l in luts[1:]:
        out = l[np.clip(out.astype(np.int32), 0, N - 1)]
    return out.astype(np.float32)


def apply_channel_luts(img, r=None, g=None, b=None, master=None):
    """img: float32 [0,1] RGB。任一通道 LUT 为 None 则用 master 或恒等。"""
    img = np.clip(img, 0, 1)
    arr8 = (img * 255.0 + 0.5).astype(np.uint8)
    out = np.empty_like(arr8)
    chans = {"r": r, "g": g, "b": b}
    for i, name in enumerate("rgb"):
        lut = chans[name] if chans[name] is not None else master
        if lut is None:
            out[..., i] = arr8[..., i]
        else:
            lut8 = np.clip(np.asarray(lut), 0, 255).astype(np.uint8)
            out[..., i] = lut8[arr8[..., i]]
    return out.astype(np.float32) / 255.0
