from __future__ import annotations

import argparse
from pathlib import Path

from .solver import compose_image, load_pieces, open_image, solve_by_edges, solve_with_reference


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTF 拼图命令行自动拼接")
    parser.add_argument("pieces", type=Path, help="拼图碎片文件夹")
    parser.add_argument("-o", "--output", type=Path, default=Path("puzzle_result.png"), help="输出 PNG 路径")
    parser.add_argument("-r", "--original", type=Path, help="完整原图，可选")
    parser.add_argument("--rows", type=int, default=0, help="矩形拼图行数，0 表示自动推断")
    parser.add_argument("--cols", type=int, default=0, help="矩形拼图列数，0 表示自动推断")
    parser.add_argument("--with-background", action="store_true", help="导出时包含半透明原图底图")
    parser.add_argument("--background-opacity", type=float, default=0.35, help="底图透明度，默认 0.35")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pieces = load_pieces(args.pieces)
    if not pieces:
        raise SystemExit(f"没有在碎片文件夹中找到图片：{args.pieces}")

    original = open_image(args.original) if args.original else None
    grid = (args.rows, args.cols) if args.rows or args.cols else None
    if original is not None:
        result = solve_with_reference(pieces, original, grid=grid)
    else:
        result = solve_by_edges(pieces, grid=grid)

    output = compose_image(
        pieces,
        original=original,
        include_background=bool(original is not None and args.with_background),
        background_opacity=args.background_opacity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)

    score = f", average_score={result.average_score:.2f}" if result.average_score is not None else ""
    print(f"done: mode={result.mode}, grid={result.rows}x{result.cols}{score}, output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
