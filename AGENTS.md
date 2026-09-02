# AGENTS.md — AI 调色 Agent 使用说明

本项目是一个纯代码实现的调色（修图）工具链。AI Agent（Codex / ZCode / Claude 等）
可以用它来**自动分析照片 → 编写/组合调色参数 → 执行 → 检查结果 → 迭代**。

## 环境

- Python 3（系统自带即可），依赖已安装：`numpy`、`opencv-python-headless`、`pillow`
- 运行方式：在项目根目录执行 `python3 retouch.py ...`

## 目录结构

```
修图/
├── retouch.py          # CLI 入口（所有日常操作用它）
├── core/
│   ├── imageio.py      # 读写、EXIF 方向、对比图拼接
│   ├── basic_grade.py  # 曝光/亮度/对比度/高光阴影/黑白场/饱和度/自然饱和度/清晰度
│   ├── color_grade.py  # 色温色调/HSL 分色/RGB 曲线/色彩矩阵/分离色调
│   ├── curves.py       # 曲线引擎（控制点→单调样条→LUT）
│   ├── lut.py          # .cube 3D LUT 读取/三线性插值/强度
│   ├── film.py         # 胶片曲线/颗粒/光晕/色偏/褪色/暗角/胶片风格
│   ├── analyze.py      # 曝光/色温/饱和度/肤色自动分析与建议
│   └── pipeline.py     # 参数合并、预设、自然语言解析、批量、自动迭代
├── presets/            # 调色预设 (.json)
└── output/             # 输出目录
```

## Agent 工作流（推荐循环）

1. **分析**：`python3 retouch.py analyze 图片1.JPG --json`
   读懂返回的 JSON：`underexposed/overexposed/warm/cool/flat/mean_saturation/skin_pct` 等。
2. **生成参数**：根据分析结果 + 用户风格要求，写一个 JSON 参数 dict。
   也可以直接用自然语言：`--style "电影感 更冷一点"`。
3. **执行**：`python3 retouch.py grade in.jpg out.jpg --style "..." [--preset name] [--set key=value ...]`
   会同时生成 `out_compare.jpg`（左右对比图）。
4. **检查**：用视觉能力查看 `out_compare.jpg`，或 `analyze out.jpg` 对比指标。
5. **迭代**：调整参数重新执行，直到满意。
6. **沉淀**：`python3 retouch.py save-preset myname --style "..."` 保存为可复用预设。

## 参数速查（--set key=value）

| key | 范围/类型 | 说明 |
|---|---|---|
| exposure | EV, 如 0.5 | 曝光（档） |
| brightness | -1..1 | 加性亮度 |
| contrast | -1..1 | 对比度 |
| highlights / shadows | -1..1 | 高光/阴影（正提亮） |
| black_point / white_point | 0..1 | 黑场抬升 / 白点 |
| saturation | -1..1 | 饱和度 |
| vibrance | -1..1 | 自然饱和度 |
| clarity | -1..1 | 清晰度 |
| temp / tint | -1..1 | 色温(正暖) / 色调(正品红) |
| hsl | 见下 | 分色调整 |
| rgb_curve_r/g/b/master | [[x,y],...] 0-255 | 通道曲线控制点 |
| color_matrix | 3x3 数组 | 色彩矩阵 |
| split_shadow / split_highlight | [r,g,b] | 分离色调颜色 |
| lut_path / lut_strength | 路径 / 0-1 | .cube LUT |
| film_stock | portra400/gold200/cinestill/fuji/bw | 胶片风格 |
| film_strength | 0-1 | 胶片风格强度 |
| grain_amount / grain_size | 0-1 / 像素 | 颗粒 |
| halation | 0-1 | 红色光晕 |
| fade_lift | 0-0.1 | 褪色抬黑 |
| rolloff | 0-1 | 高光滚落 |
| vignette | 0-1 | 暗角 |
| vignette_feather | 0.1-3 | 暗角羽化 |

`hsl` 结构：
```json
{"hsl": {"hue": {"orange": 0.2}, "sat": {"blue": -0.3}, "lum": {"green": 0.1}}}
```
色相分区：red/orange/yellow/green/aqua/blue/purple/magenta。

## 约定

- 所有图像在库内部为 float32 RGB `[0,1]`。
- 参数 dict 可任意组合叠加，应用顺序：基础 → 色彩 → LUT → 胶片效果。
- 修改代码后先跑 `python3 tests_smoke.py`（若存在）或用 `retouch.py grade` 冒烟。
- 处理含人像的照片时注意 `skin_pct`，避免肤色溢出色相分区。
