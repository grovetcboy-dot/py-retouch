"""调色管线：参数合并、应用顺序、preset 读写、自然语言解析、自动迭代、批量处理。"""
import json
import os
import re
import logging

import numpy as np

from . import imageio, basic_grade, color_grade, film as film_mod, lut as lut_mod, analyze

log = logging.getLogger("retouch")

# 参数应用顺序（基础 -> 色彩 -> LUT -> 胶片/效果）
def apply_grade(img, params):
    out = img
    out = basic_grade.apply(out, params)
    out = color_grade.apply(out, params)
    if params.get("lut_path"):
        strength = params.get("lut_strength", 1.0)
        out = lut_mod.LUT3D.from_cube(params["lut_path"]).apply(out, strength)
        log.info("应用 LUT %s (强度 %.2f)", params["lut_path"], strength)
    out = film_mod.apply(out, params)
    return out


# ---------- Preset ----------
PRESET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "presets")


def load_preset(name, preset_dir=None):
    """name 可为文件名、不含扩展名、或 'user:xxx'。返回参数 dict。"""
    d = preset_dir or PRESET_DIR
    candidates = [name if name.endswith(".json") else name + ".json"]
    if "/" not in name:
        candidates = [os.path.join(d, c) for c in candidates]
    for c in candidates:
        if os.path.isfile(c):
            with open(c) as f:
                data = json.load(f)
            log.info("加载预设 %s", c)
            return data
    raise FileNotFoundError(f"找不到预设: {name} (查找于 {candidates})")


def list_presets(preset_dir=None):
    d = preset_dir or PRESET_DIR
    if not os.path.isdir(d):
        return []
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))


def save_preset(name, params, preset_dir=None, overwrite=False):
    d = preset_dir or PRESET_DIR
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name if name.endswith(".json") else name + ".json")
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"预设已存在: {path} (overwrite=True 覆盖)")
    with open(path, "w") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    log.info("保存预设 %s", path)
    return path


def merge_params(*dicts):
    """合并参数，后者覆盖前者；hsl 子 dict 深合并。"""
    out = {}
    for d in dicts:
        if not d:
            continue
        for k, v in d.items():
            if k == "hsl" and isinstance(v, dict) and isinstance(out.get("hsl"), dict):
                for sk, sv in v.items():
                    if isinstance(sv, dict):
                        out["hsl"].setdefault(sk, {}).update(sv)
                    else:
                        out["hsl"][sk] = sv
            else:
                out[k] = v
    return out


# ---------- 自然语言 -> 参数 ----------
NUM = r"([+-]?\d+(?:\.\d+)?)"

RULES = [
    (rf"(?:曝光|提亮|加曝)[^\d-]*?{NUM}\s*档?|增加曝光\s*{NUM}|曝光\s*{NUM}\s*档", "exposure", float, 1),
    (rf"(?:压暗|减曝|降曝)[^\d]*?{NUM}", "exposure_neg", float, 1),
    (rf"对比[^\d]*?(?:提高|加|增强)[^\d]*?{NUM}|对比度?\s*\+?{NUM}", "contrast_pos", float, 1),
    (rf"对比[^\d]*?(?:降低|减|弱)[^\d]*?{NUM}", "contrast_neg", float, 1),
    (rf"(?:饱和|鲜艳)[^\d]*?(?:提高|加|增强)[^\d]*?{NUM}|饱和度?\s*\+?{NUM}", "saturation_pos", float, 1),
    (rf"(?:饱和|鲜艳)[^\d]*?(?:降低|减|去)[^\d]*?{NUM}", "saturation_neg", float, 1),
    (rf"色温[^\d]*?(?:提高|加|调暖|偏暖)[^\d]*?{NUM}|暖\s*{NUM}|调暖\s*{NUM}", "temp_pos", float, 1),
    (rf"色温[^\d]*?(?:降低|减|调冷|偏冷)[^\d]*?{NUM}|冷\s*{NUM}|调冷\s*{NUM}", "temp_neg", float, 1),
    (rf"高光[^\d]*?(?:压|降|减|收)[^\d]*?{NUM}", "highlights_neg", float, 1),
    (rf"阴影[^\d]*?(?:提|开|加|亮)[^\d]*?{NUM}", "shadows_pos", float, 1),
]

KEYWORD_RULES = [
    (["更暖", "暖一点", "偏暖", "暖调", "金色"], {"temp": 0.25}),
    (["更冷", "冷一点", "偏冷", "冷调", "蓝调"], {"temp": -0.25}),
    (["提亮", "亮一点", "更亮"], {"exposure": 0.4}),
    (["压暗", "暗一点", "更暗", "暗调"], {"exposure": -0.4}),
    (["增强对比", "加对比", "更有质感", "通透"], {"contrast": 0.2, "clarity": 0.2}),
    (["降对比", " softer", "柔和"], {"contrast": -0.15}),
    (["更鲜艳", "更饱和", "色彩浓", "浓郁"], {"vibrance": 0.35, "saturation": 0.1}),
    (["去饱和", "降低饱和", "淡雅", "低饱和"], {"saturation": -0.2}),
    (["黑白", "单色"], {"saturation": -1.0, "contrast": 0.15}),
    (["褪色", "胶片感", "复古", "怀旧"], {"film_stock": "portra400", "film_strength": 0.8}),
    (["电影感", "电影色调", "cinematic"], {"film_stock": "cinestill", "contrast": 0.15,
                                            "saturation": -0.1, "temp": 0.1}),
    (["日系", "小清新", "清透"], {"temp": 0.05, "saturation": -0.1, "black_point": 0.03,
                                   "contrast": -0.1, "exposure": 0.2}),
    (["夜景", "暗调氛围"], {"contrast": 0.2, "vignette": 0.3}),
    (["冷白", "高调"], {"exposure": 0.5, "temp": -0.1, "contrast": -0.05}),
]


def parse_natural_language(text):
    """自然语言 -> 调色参数 dict。支持中文关键词与简单数值表达。"""
    params = {}
    for pattern, key, cast, group in RULES:
        for m in re.finditer(pattern, text):
            val = cast(m.group(1))
            if key.endswith("_neg"):
                params[key[:-4]] = -val
            elif key.endswith("_pos"):
                params[key[:-4]] = val
            else:
                params[key] = val
    for keywords, p in KEYWORD_RULES:
        if any(k in text for k in keywords):
            params.update(p)
    if "颗粒" in text or "grain" in text.lower():
        params["grain_amount"] = 0.3
    if "暗角" in text:
        params["vignette"] = 0.3
    # 预设名直接引用："portra400 风格"
    for stock in film_mod.FILM_STOCKS:
        if stock in text.lower():
            params["film_stock"] = stock
    return params


# ---------- 处理入口 ----------
def process_file(src, dst, params, preview=False, preview_h=600):
    img = imageio.load(src)
    out = apply_grade(img, params)
    imageio.save(dst, out)
    log.info("已保存 %s", dst)
    if preview:
        base, ext = os.path.splitext(dst)
        sheet = imageio.make_contact_sheet([img, out], ["原图", "调色后"], target_h=preview_h)
        ppath = base + "_compare" + ".jpg"
        imageio.save(ppath, sheet)
        log.info("对比图 %s", ppath)
    return out


def process_batch(input_dir, output_dir, params, recursive=False):
    """批量处理文件夹。返回 (成功列表, 失败列表)。"""
    os.makedirs(output_dir, exist_ok=True)
    files = []
    if recursive:
        for root, _, names in os.walk(input_dir):
            files += [os.path.join(root, n) for n in names
                      if n.lower().endswith(imageio.SUPPORTED)]
    else:
        files = [os.path.join(input_dir, n) for n in sorted(os.listdir(input_dir))
                 if n.lower().endswith(imageio.SUPPORTED)]
    ok, fail = [], []
    for i, f in enumerate(files, 1):
        rel = os.path.relpath(f, input_dir)
        dst = os.path.join(output_dir, rel)
        os.makedirs(os.path.dirname(dst) or output_dir, exist_ok=True)
        try:
            process_file(f, dst, params)
            ok.append(f)
            log.info("[%d/%d] %s", i, len(files), rel)
        except Exception as e:
            fail.append((f, str(e)))
            log.error("[%d/%d] 失败 %s: %s", i, len(files), rel, e)
    return ok, fail


# ---------- 自动迭代 ----------
def auto_grade(src, dst, style=None, iterations=3, preview=True):
    """分析 -> 建议 -> 调色 -> 再分析 -> 修正参数 -> 输出。返回最终参数与迭代日志。"""
    img = imageio.load(src)
    res = analyze.analyze(img)
    params = analyze.suggestions(res)
    if style:
        params = merge_params(parse_natural_language(style), params)
    log.info("初始分析:\n%s", analyze.report(res))
    log.info("初始参数: %s", params)

    history = [dict(res)]
    for i in range(iterations):
        out = apply_grade(img, params)
        res2 = analyze.analyze(out)
        delta = analyze.diff(res, res2)
        log.info("迭代 %d: 参数=%s 变化=%s", i + 1, params, delta)
        # 收敛判断与修正
        adj = {}
        if res2["overexposed"]:
            adj["exposure"] = params.get("exposure", 0) - 0.2
        if res2["underexposed"]:
            adj["exposure"] = params.get("exposure", 0) + 0.2
        if res2["flat"]:
            adj["contrast"] = params.get("contrast", 0) + 0.1
        if res2["oversaturated"]:
            adj["saturation"] = params.get("saturation", 0) - 0.1
        if res2["low_saturation"]:
            adj["vibrance"] = params.get("vibrance", 0) + 0.15
        if not adj:
            log.info("已收敛，停止迭代")
            break
        params.update({k: round(v, 3) for k, v in adj.items()})
        res = res2
        history.append(dict(res2))

    out = apply_grade(img, params)
    imageio.save(dst, out)
    if preview:
        base = os.path.splitext(dst)[0]
        imageio.save(base + "_compare.jpg",
                     imageio.make_contact_sheet([img, out], ["原图", "自动调色"]))
    return params, history
