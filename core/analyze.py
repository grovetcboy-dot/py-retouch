"""自动分析：曝光、RGB 分布、色温估计、饱和度、肤色检测。
输出结构化 dict，供 Agent 与自动迭代使用。"""
import numpy as np
import cv2


def analyze(img, sample=400000):
    """img: float32 RGB [0,1]。返回分析结果 dict。"""
    h, w = img.shape[:2]
    n = h * w
    if n > sample:  # 均匀降采样加速
        idx = np.random.default_rng(0).choice(n, sample, replace=False)
        px = img.reshape(-1, 3)[idx]
    else:
        px = img.reshape(-1, 3)
    lum = px @ np.array([0.2126, 0.7152, 0.0722], np.float32)

    res = {}
    # ---- 曝光 ----
    res["mean_luminance"] = float(lum.mean())
    res["median_luminance"] = float(np.median(lum))
    res["std_luminance"] = float(lum.std())
    res["percentile_1"] = float(np.percentile(lum, 1))
    res["percentile_99"] = float(np.percentile(lum, 99))
    res["clip_shadow_pct"] = float((lum < 0.005).mean() * 100)
    res["clip_highlight_pct"] = float((lum > 0.995).mean() * 100)
    res["underexposed"] = res["mean_luminance"] < 0.28
    res["overexposed"] = res["clip_highlight_pct"] > 8.0
    res["flat"] = res["std_luminance"] < 0.12
    res["too_contrasty"] = res["std_luminance"] > 0.32

    # ---- RGB 分布 / 色偏 ----
    ch_mean = px.mean(axis=0)
    res["channel_means"] = {k: float(v) for k, v in zip("rgb", ch_mean)}
    res["color_cast"] = {k: float(v) for k, v in zip("rgb", ch_mean - ch_mean.mean())}
    # 色温估计：R/B 比值，1 为中性，>1 偏暖
    rb = float((ch_mean[0] + 1e-6) / (ch_mean[2] + 1e-6))
    res["warmth_ratio"] = rb
    res["warm"] = rb > 1.08
    res["cool"] = rb < 0.93
    # 色调：G 相对 R+B
    gb = float((ch_mean[1] + 1e-6) / ((ch_mean[0] + ch_mean[2]) / 2 + 1e-6))
    res["tint_ratio"] = gb
    res["green_shift"] = gb > 1.03
    res["magenta_shift"] = gb < 0.97

    # ---- 饱和度 ----
    mx, mn = px.max(axis=1), px.min(axis=1)
    sat = np.where(mx > 1e-6, (mx - mn) / (mx + 1e-9), 0)
    res["mean_saturation"] = float(sat.mean())
    res["low_saturation"] = res["mean_saturation"] < 0.10
    res["oversaturated"] = res["mean_saturation"] > 0.45

    # ---- 直方图（每通道 32 bin）----
    hist = {}
    for i, name in enumerate("rgb"):
        h_, _ = np.histogram(px[:, i], bins=32, range=(0, 1))
        hist[name] = (h_ / len(px) * 100).round(2).tolist()
    res["histograms"] = hist

    # ---- 肤色检测（YCbCr 简易阈值）----
    ycbcr = cv2.cvtColor((px * 255 + 0.5).astype(np.uint8).reshape(-1, 1, 3),
                          cv2.COLOR_RGB2YCrCb).reshape(-1, 3).astype(np.float32)
    skin = ((ycbcr[:, 1] >= 135) & (ycbcr[:, 1] <= 175) &
            (ycbcr[:, 2] >= 85) & (ycbcr[:, 2] <= 130) &
            (ycbcr[:, 0] >= 40))
    skin_pct = float(skin.mean() * 100)
    res["skin_pct"] = skin_pct
    res["has_faces"] = skin_pct > 4.0
    if skin_pct > 1.0:
        res["skin_mean_rgb"] = {k: float(v) for k, v in zip("rgb", px[skin].mean(axis=0))}

    return res


def report(res, path=""):
    """把分析结果转成人类/LLM 可读文本。"""
    lines = []
    if path:
        lines.append(f"图片: {path}")
    expo = ("欠曝" if res["underexposed"] else
            "过曝" if res["overexposed"] else "曝光正常")
    lines.append(f"曝光: {expo} (均值 {res['mean_luminance']:.3f}, "
                 f"中位数 {res['median_luminance']:.3f})")
    lines.append(f"对比: {'平淡' if res['flat'] else '过强' if res['too_contrasty'] else '适中'} "
                 f"(标准差 {res['std_luminance']:.3f})")
    lines.append(f"黑白场: 1% 分位 {res['percentile_1']:.3f}, 99% 分位 {res['percentile_99']:.3f}, "
                 f"高光溢出 {res['clip_highlight_pct']:.1f}%, 死黑 {res['clip_shadow_pct']:.1f}%")
    cast = res["color_cast"]
    lines.append(f"通道均值 R/G/B: {res['channel_means']['r']:.3f}/"
                 f"{res['channel_means']['g']:.3f}/{res['channel_means']['b']:.3f} "
                 f"(色偏 R{cast['r']:+.3f} G{cast['g']:+.3f} B{cast['b']:+.3f})")
    temp = ("偏暖" if res["warm"] else "偏冷" if res["cool"] else "中性")
    lines.append(f"色温: {temp} (R/B = {res['warmth_ratio']:.3f})")
    sat = ("过低" if res["low_saturation"] else "过高" if res["oversaturated"] else "适中")
    lines.append(f"饱和度: {sat} (均值 {res['mean_saturation']:.3f})")
    lines.append(f"肤色区域: {res['skin_pct']:.1f}%"
                 + (" (含人脸/皮肤，调色需保护)" if res["has_faces"] else ""))
    return "\n".join(lines)


def suggestions(res):
    """根据分析结果给出调色参数建议（自动迭代的起点）。"""
    s = {}
    if res["underexposed"]:
        s["exposure"] = round(min(1.5, (0.35 - res["mean_luminance"]) * 4), 2)
    if res["overexposed"]:
        s["highlights"] = -0.4
        s["exposure"] = -0.3
    if res["flat"]:
        s["contrast"] = 0.25
        s.pop("exposure", None)
    if res["too_contrasty"]:
        s["shadows"] = 0.3
        s["highlights"] = -0.2
    if res["warm"]:
        s["temp"] = -round(min(0.5, (res["warmth_ratio"] - 1) * 2), 2)
    if res["cool"]:
        s["temp"] = round(min(0.5, (1 - res["warmth_ratio"]) * 2), 2)
    if res["green_shift"]:
        s["tint"] = -0.1
    if res["magenta_shift"]:
        s["tint"] = 0.1
    if res["low_saturation"]:
        s["vibrance"] = 0.3
    if res["oversaturated"]:
        s["saturation"] = -0.15
    return s


def diff(a, b):
    """比较两次分析结果，量化变化（自动迭代用）。"""
    keys = ["mean_luminance", "std_luminance", "warmth_ratio",
            "mean_saturation", "clip_highlight_pct", "clip_shadow_pct"]
    return {k: round(b[k] - a[k], 4) for k in keys}
