from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctf_puzzle_solver.image_tools import lsb_bitplane, split_scrambled_image
from ctf_puzzle_solver.solver import load_pieces, open_image, solve_by_edges, solve_with_reference


def build_case(root: Path, rows: int, cols: int, order: list[tuple[int, int]]) -> tuple[Path, Path]:
    width = cols * 120
    height = rows * 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for y in range(height):
        for x in range(width):
            image.putpixel(
                (x, y),
                (
                    (x * 255) // max(1, width),
                    (y * 255) // max(1, height),
                    ((x * 7 + y * 11) * 255) // max(1, width * 7 + height * 11),
                ),
            )

    for row in range(rows):
        for col in range(cols):
            draw.rectangle((col * 120 + 3, row * 90 + 3, (col + 1) * 120 - 4, (row + 1) * 90 - 4), outline=(255, 255, 0), width=2)
            draw.text((col * 120 + 42, row * 90 + 36), f"{row},{col}", fill=(255, 255, 255))

    original_path = root / "original.png"
    pieces_dir = root / "pieces"
    pieces_dir.mkdir(parents=True, exist_ok=True)
    image.save(original_path)

    for index, (row, col) in enumerate(order):
        piece = image.crop((col * 120, row * 90, (col + 1) * 120, (row + 1) * 90))
        piece.save(pieces_dir / f"piece_{index:02d}_{row}_{col}.png")

    return original_path, pieces_dir


def assert_expected_positions(pieces) -> None:
    for piece in pieces:
        parts = piece.path.stem.split("_")
        expected = (int(parts[-2]), int(parts[-1]))
        if piece.matched_cell != expected:
            raise AssertionError(f"{piece.name}: expected {expected}, got {piece.matched_cell}")


def run_case(rows: int, cols: int, order: list[tuple[int, int]]) -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="shrimp_jigsaw_"))
    try:
        original_path, pieces_dir = build_case(temp_root, rows, cols, order)

        reference_pieces = load_pieces(pieces_dir)
        solve_with_reference(reference_pieces, open_image(original_path), grid=(rows, cols))
        assert_expected_positions(reference_pieces)

        edge_pieces = load_pieces(pieces_dir)
        solve_by_edges(edge_pieces, grid=(rows, cols))
        assert_expected_positions(edge_pieces)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def run_split_and_lsb_case() -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="shrimp_tools_"))
    try:
        original_path, pieces_dir = build_case(
            temp_root,
            2,
            3,
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        )
        pieces = load_pieces(pieces_dir)
        sheet = Image.new("RGBA", (360, 180), (0, 0, 0, 255))
        for piece in pieces:
            row = int(piece.path.stem.split("_")[-2])
            col = int(piece.path.stem.split("_")[-1])
            sheet.alpha_composite(piece.image, (col * 120, row * 90))
        sheet_path = temp_root / "sheet.png"
        sheet.save(sheet_path)

        split_result = split_scrambled_image(sheet_path, temp_root / "split", rows=2, cols=3)
        if split_result.count != 6 or split_result.rows != 2 or split_result.cols != 3:
            raise AssertionError(f"unexpected split result: {split_result}")

        uneven_sheet = Image.new("RGBA", (367, 183), (0, 0, 0, 255))
        for piece in pieces:
            row = int(piece.path.stem.split("_")[-2])
            col = int(piece.path.stem.split("_")[-1])
            uneven_sheet.alpha_composite(piece.image.resize((122, 91)), (col * 122, row * 91))
        uneven_path = temp_root / "uneven_sheet.png"
        uneven_sheet.save(uneven_path)
        normalized = split_scrambled_image(uneven_path, temp_root / "normalized", rows=2, cols=3, normalize_piece_size=True)
        sizes = {open_image(path).size for path in normalized.folder.glob("piece_*.png")}
        if len(sizes) != 1:
            raise AssertionError(f"normalized split sizes differ: {sizes}")

        lsb = lsb_bitplane(open_image(original_path), "R", 0)
        if lsb.size != (360, 180):
            raise AssertionError(f"unexpected LSB preview size: {lsb.size}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def run_large_grid_inference_case() -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="shrimp_large_grid_"))
    try:
        rows, cols = 12, 20
        cell = 24
        sheet = Image.new("RGBA", (cols * cell, rows * cell), (0, 0, 0, 255))
        for row in range(rows):
            for col in range(cols):
                tile = Image.new(
                    "RGBA",
                    (cell, cell),
                    ((col * 17 + row * 3) % 255, (row * 19 + col * 5) % 255, (row * 29 + col * 11) % 255, 255),
                )
                sheet.alpha_composite(tile, (col * cell, row * cell))
        sheet_path = temp_root / "large_sheet.png"
        sheet.save(sheet_path)

        split_result = split_scrambled_image(sheet_path, temp_root / "large_split", normalize_piece_size=True)
        if split_result.rows != rows or split_result.cols != cols or split_result.count != rows * cols:
            raise AssertionError(f"unexpected large grid split result: {split_result}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> int:
    run_case(3, 3, [(2, 1), (0, 0), (1, 2), (2, 2), (0, 2), (1, 0), (2, 0), (0, 1), (1, 1)])
    run_case(
        3,
        4,
        [(2, 3), (0, 0), (1, 2), (2, 0), (0, 3), (1, 0), (2, 1), (0, 1), (1, 3), (2, 2), (0, 2), (1, 1)],
    )
    run_split_and_lsb_case()
    run_large_grid_inference_case()
    print("smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
