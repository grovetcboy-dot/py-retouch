"""富士 F-Log2 输入转换：普通 sRGB 照片 -> F-Log2 / F-Gamut，
用于套用富士官方 F-Log2 -> 胶片模拟 的 .cube LUT。

流程: sRGB(显示域) -> 线性 -> BT.709 -> BT.2020(≈F-Gamut) -> F-Log2 编码
（F-Gamut C 与 F-Gamut 用同一近似；色域差异对最终观感影响很小）
"""
import numpy as np

# BT.709 -> BT.2020 原色变换（线性光域）
_BT709_TO_BT2020 = np.array([
    [0.6274040, 0.3292840, 0.0433131],
    [0.0690970, 0.9195400, 0.0113612],
    [0.0163916, 0.0880132, 0.8955952]], np.float32)

# F-Log2 编码常数（按富士锚点标定: 0%/18%/90% 反射率 -> 10bit 码 95/470/705）
_LOG_CUT = 0.000889
_LOG_SLOPE = 8.799461
_LOG_OFFSET = 0.092864
_LOG_A = 0.342877
_LOG_B = 5.555556
_LOG_C = 0.087845
_LOG_D = 0.446895


def srgb_to_linear(img):
    """sRGB 传递函数 -> 线性光。"""
    return np.where(img <= 0.04045,
                    img / 12.92,
                    np.power(np.maximum(img, 1e-10), 2.4))


def linear_to_srgb(img):
    """线性光 -> sRGB 传递函数。"""
    return np.where(img <= 0.0031308,
                    img * 12.92,
                    1.055 * np.power(np.maximum(img, 1e-10), 1 / 2.4) - 0.055)


def srgb_to_flog2(img):
    """float32 sRGB [0,1] -> F-Log2 编码值 [0,1]。"""
    lin = srgb_to_linear(np.clip(img, 0, 1))
    lin2020 = lin @ _BT709_TO_BT2020.T
    lin2020 = np.clip(lin2020, 0, None)
    # F-Log2 编码（按通道）
    log = np.where(
        lin2020 < _LOG_CUT,
        _LOG_SLOPE * lin2020 + _LOG_OFFSET,
        _LOG_A * np.log10(_LOG_B * lin2020 + _LOG_C) + _LOG_D)
    # 10-bit 满量程 1023 对应 1.0，这里 log 值已归一
    return np.clip(log, 0, 1).astype(np.float32)


def apply_fujifilm_lut(img, cube_path, strength=1.0, exposure=0.0):
    """把 sRGB 照片用富士 F-Log2 胶片模拟 LUT 处理。
    exposure: 输入端曝光补偿（档），补偿 log 空间的映射位置。"""
    from .lut import LUT3D
    src = img
    if exposure:
        lin = srgb_to_linear(np.clip(img, 0, 1)) * (2.0 ** exposure)
        src = linear_to_srgb(np.clip(lin, 0, 1))
    flog = srgb_to_flog2(src)
    return LUT3D.from_cube(cube_path).apply(flog, strength)
