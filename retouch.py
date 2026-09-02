#!/usr/bin/env python3
"""retouch — 代码调色命令行工具。

用法:
  python3 retouch.py analyze 图片1.JPG                     # 分析图片
  python3 retouch.py grade 图片1.JPG out.jpg --preset cinestill
  python3 retouch.py grade 图片1.JPG out.jpg --style "电影感 更冷一点"
  python3 retouch.py auto 图片1.JPG out.jpg                # 自动调色迭代
  python3 retouch.py batch in/ out/ --preset portra400     # 批量
  python3 retouch.py presets                               # 列出预设
  python3 retouch.py save-preset myp --style "日系"         # 保存预设
"""
import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import pipeline, analyze, imageio  # noqa: E402
from core.lut import LUT3D  # noqa: E402


def cmd_analyze(args):
    img = imageio.load(args.image)
    res = analyze.analyze(img)
    print(analyze.report(res, args.image))
    print("\n建议参数:", json.dumps(analyze.suggestions(res), ensure_ascii=False))
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))


def _build_params(args):
    params = {}
    if args.preset:
        for p in args.preset:
            params = pipeline.merge_params(params, pipeline.load_preset(p))
    if args.style:
        params = pipeline.merge_params(params, pipeline.parse_natural_language(args.style))
    kv = {}
    for item in args.set or []:
        k, _, v = item.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        kv[k] = v
    params = pipeline.merge_params(params, kv)
    if args.lut:
        params = pipeline.merge_params(params, {"lut_path": args.lut,
                                                "lut_strength": args.lut_strength})
    if not params:
        logging.warning("未指定任何调色参数（--preset/--style/--set/--lut）")
    return params


def cmd_grade(args):
    params = _build_params(args)
    logging.info("参数: %s", json.dumps(params, ensure_ascii=False))
    pipeline.process_file(args.image, args.output, params,
                          preview=not args.no_preview)
    print("完成:", args.output)


def cmd_auto(args):
    params, history = pipeline.auto_grade(args.image, args.output, style=args.style,
                                          iterations=args.iterations, preview=True)
    print("最终参数:", json.dumps(params, ensure_ascii=False))
    print("完成:", args.output)
    if args.save_preset:
        pipeline.save_preset(args.save_preset, params, overwrite=True)
        print("参数已存为预设:", args.save_preset)


def cmd_batch(args):
    params = _build_params(args)
    ok, fail = pipeline.process_batch(args.input, args.output, params, recursive=args.recursive)
    print(f"批量完成: 成功 {len(ok)}, 失败 {len(fail)}")
    for f, e in fail:
        print("  失败:", f, e)


def cmd_presets(args):
    names = pipeline.list_presets()
    if not names:
        print("(无预设)")
    for n in names:
        print("-", n)


def cmd_save_preset(args):
    params = _build_params(args)
    path = pipeline.save_preset(args.name, params, overwrite=args.overwrite)
    print("已保存:", path)
    print(json.dumps(params, ensure_ascii=False, indent=2))


def cmd_lut_info(args):
    lut = LUT3D.from_cube(args.cube)
    print(f"LUT: {lut.name or '(无标题)'}  size={lut.size}  体素={lut.size**3}")


def main():
    ap = argparse.ArgumentParser(prog="retouch", description="代码调色工具")
    ap.add_argument("-v", "--verbose", action="store_true", help="显示日志")
    top = ap.add_subparsers(dest="cmd", required=True)

    p = top.add_parser("analyze", help="分析图片")
    p.add_argument("image")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_analyze)

    def common(p):
        p.add_argument("--preset", action="append", help="预设名，可多次")
        p.add_argument("--style", help="自然语言风格描述")
        p.add_argument("--set", action="append", metavar="K=V", help="直接设置参数，可多次")
        p.add_argument("--lut", help=".cube LUT 路径")
        p.add_argument("--lut-strength", type=float, default=1.0)

    p = top.add_parser("grade", help="调色输出")
    common(p)
    p.add_argument("image")
    p.add_argument("output")
    p.add_argument("--no-preview", action="store_true", help="不生成对比图")
    p.set_defaults(fn=cmd_grade)

    p = top.add_parser("auto", help="自动分析迭代调色")
    p.add_argument("image")
    p.add_argument("output")
    p.add_argument("--style", help="目标风格描述")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--save-preset", help="把最终参数保存为预设")
    p.set_defaults(fn=cmd_auto)

    p = top.add_parser("batch", help="批量处理文件夹")
    common(p)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("-r", "--recursive", action="store_true")
    p.set_defaults(fn=cmd_batch)

    p = top.add_parser("presets", help="列出预设")
    p.set_defaults(fn=cmd_presets)

    p = top.add_parser("save-preset", help="保存预设")
    common(p)
    p.add_argument("name")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_save_preset)

    p = top.add_parser("lut-info", help="查看 .cube LUT 信息")
    p.add_argument("cube")
    p.set_defaults(fn=cmd_lut_info)

    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    if args.cmd not in ("analyze", "presets", "lut-info"):
        logging.getLogger().setLevel(logging.INFO)
    try:
        args.fn(args)
    except FileNotFoundError as e:
        sys.exit(f"错误: {e}")
    except Exception as e:
        if args.verbose:
            raise
        sys.exit(f"错误: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
