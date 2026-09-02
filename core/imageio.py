"""图像读写：处理 EXIF 方向、ICC 忽略、8/16bit 归一到 float32 [0,1]。"""
import numpy as np
import cv2
from PIL import Image, ImageOps

SUPPORTED = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def load(path):
    """读取图片 -> float32 RGB [0,1] (H,W,3)。自动应用 EXIF 方向。"""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return arr.astype(np.float32)


def save(path, img, quality=95):
    """保存 float32 RGB [0,1] -> 文件（按扩展名选格式，jpg 默认质量 95）。"""
    img = np.clip(img, 0.0, 1.0)
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in ("jpg", "jpeg"):
        arr = (img * 255.0 + 0.5).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == "png":
        arr = (img * 255.0 + 0.5).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    elif ext in ("tif", "tiff"):
        arr = (img * 65535.0 + 0.5).astype(np.uint16)
        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    else:
        Image.fromarray((img * 255.0 + 0.5).astype(np.uint8)).save(path)
    return path


def make_contact_sheet(imgs, labels=None, target_h=600):
    """多张同尺寸缩略图横向拼接，用于前后对比。"""
    import cv2 as _cv
    scaled = []
    for im in imgs:
        h = target_h
        w = max(1, round(im.shape[1] * h / im.shape[0]))
        scaled.append(_cv.resize(im, (w, h), interpolation=_cv.INTER_AREA))
    total_w = sum(s.shape[1] for s in scaled)
    sheet = np.zeros((target_h, total_w, 3), np.float32)
    x = 0
    for s in scaled:
        sheet[:, x:x + s.shape[1]] = s
        x += s.shape[1]
    if labels:
        from PIL import ImageDraw, ImageFont
        pil = Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        except OSError:
            font = ImageFont.load_default()
        x = 10
        for s, lab in zip(scaled, labels):
            d.text((x, 10), str(lab), fill=(255, 255, 60), font=font)
            x += s.shape[1]
        sheet = np.asarray(pil).astype(np.float32) / 255.0
    return sheet
