"""胶片模拟：胶片曲线、颗粒、色偏、褪色、高光 Roll-off。"""
import numpy as np


def film_curve(img, toe=0.02, shoulder=0.15, contrast=0.3):
    """胶片 H&D 曲线：趾部(阴影有细节) + 肩部(高光平滑滚降)。"""
    x = np.linspace(0, 1, 256, dtype=np.float32)
    y = x ** (1.0 + toe * 10)
    k = 4.0 / max(shoulder, 1e-3)
    y = np.where(x > 1 - shoulder,
                 1 - shoulder + shoulder * np.tanh((x - (1 - shoulder)) * k) / np.tanh(k * shoulder),
                 y)
    y = 0.5 + (y - 0.5) * (1 + contrast)
    lut = (np.clip(y, 0, 1) * 255).astype(np.uint8)
    img8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    import cv2
    out = cv2.LUT(img8, lut)
    return out.astype(np.float32) / 255.0


def highlight_rolloff(img, strength=0.5):
    """高光滚落：高光压向暖白，模拟胶片高光柔化。"""
    lum = img.mean(axis=2, keepdims=True)
    hw = np.clip((lum - 0.75) / 0.25, 0, 1) ** 2
    warm = np.array([1.04, 1.0, 0.94], np.float32)
    out = img * (1 + (warm - 1) * hw * strength)
    return np.clip(out, 0, 1)


def grain(img, amount=0.3, size=1.0, monochrome=True):
    """胶片颗粒：高斯噪声，按亮度加权(中间调颗粒最多，符合胶片)。"""
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    rng = np.random.default_rng()
    sigma = amount * 0.06
    if monochrome:
        noise = rng.normal(0, sigma, (h, w, 1)).astype(np.float32)
    else:
        noise = rng.normal(0, sigma, (h, w, 3)).astype(np.float32)
    if size != 1.0:
        import cv2
        k = max(1, int(round(size)) | 1)
        if noise.shape[-1] == 1:
            noise = cv2.GaussianBlur(noise, (k, k), 0)
        else:
            for i in range(3):
                noise[..., i] = cv2.GaussianBlur(noise[..., i], (k, k), 0)
    lum = img.mean(axis=2, keepdims=True)
    weight = 4 * lum * (1 - lum) + 0.25  # 中间调最重
    return np.clip(img + noise * weight, 0, 1)


def halation(img, strength=0.3):
    """光晕/泛红：高亮区域红色扩散，模拟胶片卤化反应。"""
    if strength <= 0:
        return img
    import cv2
    lum = img.mean(axis=2)
    mask = np.clip((lum - 0.55) / 0.45, 0, 1) ** 1.3
    blur = cv2.GaussianBlur(mask, (0, 0), sigmaX=14)
    glow = blur[..., None] * strength * 0.9
    add = glow * np.array([1.0, 0.25, 0.08], np.float32)
    return np.clip(img + add, 0, 1)


def color_cast(img, cast=(1.0, 0.99, 0.96), strength=1.0):
    """整体色偏：cast 为 RGB 增益。"""
    c = np.asarray(cast, np.float32)
    c = 1.0 + (c - 1.0) * strength
    return np.clip(img * c, 0, 1)


def fade(img, lift=0.04, strength=1.0):
    """褪色：抬黑、降对比，常见于胶片扫描风。"""
    lift = lift * strength
    if lift <= 0:
        return img
    out = lift + (1 - lift) * img
    shadow = np.clip(1 - img.mean(axis=2, keepdims=True) * 2, 0, 1)
    cool = np.array([-0.02, 0.0, 0.03], np.float32) * strength
    out = out + shadow * cool
    return np.clip(out, 0, 1)


def vignette(img, amount=0.3, feather=1.0):
    """暗角。amount 正为四角变暗。"""
    if amount == 0:
        return img
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    d = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2) / np.sqrt(2)
    mask = np.clip(1 - amount * (d ** (2.0 / max(feather, 0.1))), 0, 1)
    return np.clip(img * mask[..., None], 0, 1)


FILM_STOCKS = {
    "portra400": dict(cast=(1.02, 1.0, 0.97), lift=0.03, grain=0.25,
                      rolloff=0.6, saturation=-0.08, warmth=0.15),
    "gold200":   dict(cast=(1.05, 1.0, 0.9), lift=0.02, grain=0.35,
                      rolloff=0.5, saturation=0.05, warmth=0.3),
    "cinestill": dict(cast=(1.0, 0.99, 1.0), lift=0.0, grain=0.15,
                      rolloff=0.7, saturation=0.0, warmth=0.0, halation=0.5),
    "fuji":      dict(cast=(0.98, 1.0, 1.02), lift=0.04, grain=0.2,
                      rolloff=0.5, saturation=0.1, warmth=-0.1),
    "bw":        dict(cast=(1, 1, 1), lift=0.02, grain=0.4,
                      rolloff=0.5, saturation=-1.0, warmth=0.0),
    # 柯达克罗姆 Kodachrome 64：高饱和、深黑、红/黄浓烈、略偏品红高光
    "kodachrome": dict(cast=(1.04, 0.99, 0.97), lift=0.0, grain=0.3,
                       rolloff=0.45, saturation=0.28, warmth=0.12,
                       contrast=0.3, deep_black=True),
    # CineStill 800T：钨光夜景片，冷调/青蓝，红色光晕明显
    "800t":      dict(cast=(0.97, 1.0, 1.06), lift=0.015, grain=0.25,
                      rolloff=0.7, saturation=0.1, warmth=-0.28,
                      halation=0.55, contrast=0.18, cyan_shadow=True),
    # 柯达 Vision3 250D (5207) 电影负片：柔和对比、自然偏灰、暖白高光、微青阴影、细颗粒
    "5207":      dict(cast=(1.01, 1.0, 1.0), lift=0.012, grain=0.18,
                      rolloff=0.65, saturation=-0.06, warmth=0.06,
                      halation=0.3, contrast=-0.08, cyan_shadow=True),
}


def apply_stock(img, stock, strength=1.0):
    """按预置胶片风格名应用组合效果。"""
    from . import basic_grade
    p = FILM_STOCKS[stock]
    out = img
    out = basic_grade.saturation(out, p["saturation"] * strength)
    if p["warmth"]:
        out = np.clip(out * np.array([1 + p["warmth"] * 0.2 * strength,
                                      1, 1 - p["warmth"] * 0.2 * strength], np.float32), 0, 1)
    if p.get("deep_black"):
        out = np.clip((out - 0.03) / 0.97, 0, 1) * strength + out * (1 - strength)
    if p.get("cyan_shadow"):
        lum = out.mean(axis=2, keepdims=True)
        sw = np.clip(1 - lum * 2.2, 0, 1)
        cyan = np.array([-0.02, 0.01, 0.04], np.float32) * strength
        out = np.clip(out + sw * cyan, 0, 1)
    out = film_curve(out, toe=0.02, shoulder=0.12,
                     contrast=p.get("contrast", 0.15) * strength)
    out = highlight_rolloff(out, p["rolloff"] * strength)
    if p.get("halation"):
        out = halation(out, p["halation"] * strength)
    out = fade(out, p["lift"], strength)
    out = color_cast(out, p["cast"], strength)
    out = grain(out, p["grain"] * strength)
    return out


def apply(img, params):
    """按 dict 应用胶片效果。键: film_stock, film_strength, film_curve(bool),
    film_contrast, rolloff, halation, fade_lift, color_cast, vignette,
    vignette_feather, grain_amount, grain_size。"""
    out = img
    if params.get("film_stock"):
        return apply_stock(out, params["film_stock"], params.get("film_strength", 1.0))
    if params.get("film_curve", False):
        out = film_curve(out, contrast=params.get("film_contrast", 0.3))
    if params.get("rolloff"):
        out = highlight_rolloff(out, params["rolloff"])
    if params.get("halation"):
        out = halation(out, params["halation"])
    if params.get("fade_lift"):
        out = fade(out, params["fade_lift"])
    if params.get("color_cast"):
        out = color_cast(out, params["color_cast"])
    if params.get("vignette"):
        out = vignette(out, params["vignette"], params.get("vignette_feather", 1.0))
    if params.get("grain_amount"):
        out = grain(out, params["grain_amount"], params.get("grain_size", 1.0))
    return out
