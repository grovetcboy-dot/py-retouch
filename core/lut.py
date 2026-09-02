"""LUT 系统：.cube 读取、3D LUT 三线性插值、强度混合。"""
import numpy as np
import cv2


class LUT3D:
    def __init__(self, data, size, name=""):
        """data: (size^3, 3) 或 (size,size,size,3) 的 array，值域 0-1。"""
        self.size = size
        self.name = name
        data = np.asarray(data, np.float32)
        if data.shape == (size ** 3, 3):
            data = data.reshape(size, size, size, 3)
        # .cube 的坐标顺序是 R 快 B 慢：data[b, g, r] 依标准
        self.data = np.ascontiguousarray(data)

    @classmethod
    def from_cube(cls, path):
        """读取 Adobe .cube 3D LUT。"""
        size, table = None, []
        name = ""
        with open(path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                low = line.lower()
                if low.startswith("title"):
                    name = line.split(None, 1)[1].strip('"') if len(line.split(None, 1)) > 1 else ""
                elif low.startswith("lut_3d_size"):
                    size = int(line.split()[1])
                elif low.startswith(("domain_min", "domain_max", "lut_1d_size")):
                    continue
                else:
                    parts = line.split()
                    if len(parts) >= 3:
                        table.append([float(x) for x in parts[:3]])
        if size is None:
            raise ValueError(f"{path}: 缺少 LUT_3D_SIZE")
        data = np.array(table, np.float32)
        if len(data) != size ** 3:
            raise ValueError(f"{path}: 数据行数 {len(data)} != {size**3}")
        return cls(data, size, name)

    def apply(self, img, strength=1.0):
        """三线性插值应用 3D LUT。img: float32 RGB [0,1]，strength 0-1 混合。"""
        out = self._trilinear(img)
        if strength < 1.0:
            out = img * (1 - strength) + out * strength
        return np.clip(out, 0, 1).astype(np.float32)

    def _trilinear(self, img):
        s = self.size - 1
        x = img[..., 0] * s  # R -> 快轴
        y = img[..., 1] * s
        z = img[..., 2] * s
        x0, y0, z0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32), np.floor(z).astype(np.int32)
        x1, y1, z1 = np.minimum(x0 + 1, s), np.minimum(y0 + 1, s), np.minimum(z0 + 1, s)
        dx, dy, dz = (x - x0)[..., None], (y - y0)[..., None], (z - z0)[..., None]
        d = self.data  # data[b, g, r] —— .cube 中第一列是 R，变化最快
        # 顶点取色：注意轴顺序 data[z, y, x]
        c000 = d[z0, y0, x0]; c100 = d[z0, y0, x1]
        c010 = d[z0, y1, x0]; c110 = d[z0, y1, x1]
        c001 = d[z1, y0, x0]; c101 = d[z1, y0, x1]
        c011 = d[z1, y1, x0]; c111 = d[z1, y1, x1]
        out = (c000 * (1 - dx) * (1 - dy) * (1 - dz) + c100 * dx * (1 - dy) * (1 - dz) +
               c010 * (1 - dx) * dy * (1 - dz) + c110 * dx * dy * (1 - dz) +
               c001 * (1 - dx) * (1 - dy) * dz + c101 * dx * (1 - dy) * dz +
               c011 * (1 - dx) * dy * dz + c111 * dx * dy * dz)
        return out.astype(np.float32)

    def to_lut_and_apply_fast(self, img, strength=1.0):
        """更快路径：把 3D LUT 展平成 cv2 支持的 512^3 稀疏不适用时仍用 _trilinear。
        这里提供小图快速近似：直接用 _trilinear（已向量化）。"""
        return self.apply(img, strength)


def apply_cube(img, cube_path, strength=1.0):
    """便捷函数：读取 .cube 并应用。"""
    return LUT3D.from_cube(cube_path).apply(img, strength)


def identity_lut(size=33):
    """生成恒等 3D LUT（用于测试/导出）。"""
    ax = np.linspace(0, 1, size, dtype=np.float32)
    b, g, r = np.meshgrid(ax, ax, ax, indexing="ij")
    data = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=-1)
    return LUT3D(data, size, "identity")


def export_cube(lut3d, path):
    """把 LUT3D 导出为 .cube 文件。"""
    size = lut3d.size
    with open(path, "w") as f:
        f.write(f'TITLE "{lut3d.name or "exported"}"\n')
        f.write(f"LUT_3d_SIZE {size}\n\n")
        flat = lut3d.data.reshape(-1, 3)
        for row in flat:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
    return path
