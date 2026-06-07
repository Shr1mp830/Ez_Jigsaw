from __future__ import annotations

import builtins
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import ceil, log
from pathlib import Path
from statistics import median
from typing import Callable, Iterable

from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}
MAX_EDGE_SOLVE_PIECES = 260
ProgressCallback = Callable[[float, str], None]
_builtin_min = builtins.min
_builtin_max = builtins.max
_builtin_sum = builtins.sum
_builtin_zip = builtins.zip
_builtin_enumerate = builtins.enumerate


@dataclass
class Piece:
    id: int
    name: str
    path: Path
    image: Image.Image
    x: float = 0.0
    y: float = 0.0
    score: float | None = None
    matched_cell: tuple[int, int] | None = None

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


@dataclass
class SolveResult:
    mode: str
    rows: int
    cols: int
    canvas_width: int
    canvas_height: int
    average_score: float | None = None


def report_progress(progress_callback: ProgressCallback | None, value: float, message: str) -> None:
    if progress_callback is None:
        return
    progress_callback(max(0.0, min(100.0, value)), message)


def load_pieces(folder: str | Path) -> list[Piece]:
    folder_path = Path(folder)
    files = [
        path
        for path in sorted(folder_path.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    def load_one(item: tuple[int, Path]) -> Piece:
        index, path = item
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGBA")
        return Piece(index, path.name, path, image)

    pieces: list[Piece] = []
    if len(files) < 4:
        for item in enumerate(files):
            pieces.append(load_one(item))
        return pieces

    max_workers = min(8, len(files))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pieces.extend(executor.map(load_one, enumerate(files)))
    return pieces


def open_image(path: str | Path) -> Image.Image:
    with Image.open(path) as raw:
        return ImageOps.exif_transpose(raw).convert("RGBA")


def scatter_pieces(pieces: list[Piece], start_x: int = 0, start_y: int = 0, spacing: int = 32) -> None:
    if not pieces:
        return

    columns = ceil(len(pieces) ** 0.5)
    typical_width = median(piece.width for piece in pieces)
    shelf_width = int(max(640, typical_width * columns + spacing * max(0, columns - 1)))
    x = start_x
    y = start_y
    row_height = 0
    for piece in pieces:
        if x > start_x and x + piece.width > start_x + shelf_width:
            x = start_x
            y += row_height + spacing
            row_height = 0
        piece.x = x
        piece.y = y
        piece.score = None
        piece.matched_cell = None
        x += piece.width + spacing
        row_height = max(row_height, piece.height)


def infer_grid(piece_count: int, target_aspect: float = 1.0) -> tuple[int, int]:
    if piece_count <= 0:
        return (0, 0)

    target_aspect = max(target_aspect, 0.05)
    best_rows = 1
    best_cols = piece_count
    best_score = float("inf")

    for rows in range(1, piece_count + 1):
        cols = ceil(piece_count / rows)
        empty_cells = rows * cols - piece_count
        grid_aspect = cols / rows
        score = abs(log(grid_aspect / target_aspect)) + empty_cells * 0.35
        if score < best_score:
            best_score = score
            best_rows = rows
            best_cols = cols

    return best_rows, best_cols


def infer_reference_grid(original: Image.Image, pieces: list[Piece]) -> tuple[int, int]:
    if not pieces:
        return (0, 0)

    piece_count = len(pieces)
    target_aspect = original.width / max(1, original.height)
    typical_width = median(piece.width for piece in pieces)
    typical_height = median(piece.height for piece in pieces)
    best_rows, best_cols = infer_grid(piece_count, target_aspect)
    best_score = float("inf")

    for rows in range(1, piece_count + 1):
        cols = ceil(piece_count / rows)
        empty_cells = rows * cols - piece_count
        cell_width = original.width / cols
        cell_height = original.height / rows
        grid_aspect = cols / rows
        aspect_score = abs(log(grid_aspect / target_aspect))
        size_score = (
            abs(cell_width - typical_width) / max(1.0, typical_width)
            + abs(cell_height - typical_height) / max(1.0, typical_height)
        )
        score = aspect_score + size_score * 0.8 + empty_cells * 0.35
        if score < best_score:
            best_score = score
            best_rows = rows
            best_cols = cols

    return best_rows, best_cols


def solve_with_reference(
    pieces: list[Piece],
    original: Image.Image,
    grid: tuple[int, int] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SolveResult:
    if not pieces:
        return SolveResult("reference", 0, 0, original.width, original.height)

    report_progress(progress_callback, 5, "正在推断原图网格")
    rows, cols = normalize_grid(grid, len(pieces)) or infer_reference_grid(original, pieces)
    cells: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for row in range(rows):
        for col in range(cols):
            left = round(col * original.width / cols)
            top = round(row * original.height / rows)
            right = round((col + 1) * original.width / cols)
            bottom = round((row + 1) * original.height / rows)
            cells.append((row, col, (left, top, right, bottom)))

    cost_matrix: list[list[float]] = []
    total_regions = max(1, len(pieces) * len(cells))
    checked_regions = 0
    for piece in pieces:
        piece_scores: list[float] = []
        for cell_index, (_, _, box) in enumerate(cells):
            cell_image = original.crop(box)
            piece_scores.append(region_score(piece.image, cell_image))
            checked_regions += 1
            if checked_regions % max(1, total_regions // 20) == 0:
                report_progress(progress_callback, 10 + checked_regions / total_regions * 65, "正在匹配原图区域")
        cost_matrix.append(piece_scores)

    report_progress(progress_callback, 80, "正在计算最优分配")
    assignment = min_cost_assignment(cost_matrix)
    score_sum = 0.0
    score_count = 0

    for piece_index, cell_index in enumerate(assignment):
        if cell_index < 0:
            continue
        row, col, box = cells[cell_index]
        score = cost_matrix[piece_index][cell_index]
        piece = pieces[piece_index]
        piece.x = box[0]
        piece.y = box[1]
        piece.score = score
        piece.matched_cell = (row, col)
        score_sum += score
        score_count += 1

    average_score = score_sum / score_count if score_count else None
    report_progress(progress_callback, 100, "原图匹配完成")
    return SolveResult("reference", rows, cols, original.width, original.height, average_score)


def solve_by_edges(
    pieces: list[Piece],
    grid: tuple[int, int] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SolveResult:
    if not pieces:
        return SolveResult("edges", 0, 0, 0, 0)
    if len(pieces) > MAX_EDGE_SOLVE_PIECES:
        raise ValueError(
            f"无原图边缘拼接当前限制 {MAX_EDGE_SOLVE_PIECES} 个碎片；"
            "碎片更多时建议使用完整原图或分批处理"
        )

    report_progress(progress_callback, 3, "正在准备边缘拼接")
    piece_aspect = median(piece.width for piece in pieces) / max(1, median(piece.height for piece in pieces))
    target_grid_aspect = 1.0 / max(0.05, piece_aspect)
    rows, cols = normalize_grid(grid, len(pieces)) or infer_grid(len(pieces), target_grid_aspect)
    typical_width = int(median(piece.width for piece in pieces))
    typical_height = int(median(piece.height for piece in pieces))

    edge_cache = {piece.id: extract_edges(piece.image) for piece in pieces}
    right_scores: dict[tuple[int, int], float] = {}
    down_scores: dict[tuple[int, int], float] = {}
    total_pairs = max(1, len(pieces) * (len(pieces) - 1))
    checked_pairs = 0
    for left_piece in pieces:
        for right_piece in pieces:
            if left_piece.id == right_piece.id:
                continue
            right_scores[(left_piece.id, right_piece.id)] = edge_score(
                edge_cache[left_piece.id]["right"],
                edge_cache[right_piece.id]["left"],
            )
            down_scores[(left_piece.id, right_piece.id)] = edge_score(
                edge_cache[left_piece.id]["bottom"],
                edge_cache[right_piece.id]["top"],
            )
            checked_pairs += 1
            if checked_pairs % max(1, total_pairs // 25) == 0:
                report_progress(progress_callback, 5 + checked_pairs / total_pairs * 35, "正在计算碎片边缘相似度")

    ids = [piece.id for piece in pieces]
    gaps_right_scores = gaps_adjust_scores(ids, right_scores)
    gaps_down_scores = gaps_adjust_scores(ids, down_scores)
    candidate_right_scores = confidence_adjust_scores(ids, gaps_right_scores)
    candidate_down_scores = confidence_adjust_scores(ids, gaps_down_scores)
    by_id = {piece.id: piece for piece in pieces}
    all_scores = list(right_scores.values()) + list(down_scores.values())
    score_scale = max(1.0, median(all_scores) if all_scores else 1.0)

    best_left_fit = {piece_id: min((right_scores[(other_id, piece_id)] for other_id in ids if other_id != piece_id), default=score_scale) for piece_id in ids}
    best_right_fit = {piece_id: min((right_scores[(piece_id, other_id)] for other_id in ids if other_id != piece_id), default=score_scale) for piece_id in ids}
    best_top_fit = {piece_id: min((down_scores[(other_id, piece_id)] for other_id in ids if other_id != piece_id), default=score_scale) for piece_id in ids}
    best_bottom_fit = {piece_id: min((down_scores[(piece_id, other_id)] for other_id in ids if other_id != piece_id), default=score_scale) for piece_id in ids}

    def border_cost(best_fit: float) -> float:
        return 0.35 * score_scale / (1.0 + best_fit / score_scale)

    border_left = {piece_id: border_cost(best_left_fit[piece_id]) for piece_id in ids}
    border_right = {piece_id: border_cost(best_right_fit[piece_id]) for piece_id in ids}
    border_top = {piece_id: border_cost(best_top_fit[piece_id]) for piece_id in ids}
    border_bottom = {piece_id: border_cost(best_bottom_fit[piece_id]) for piece_id in ids}

    try:
        grid_layout, best_score = genetic_edge_grid(
            ids,
            rows,
            cols,
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            candidate_right_scores=candidate_right_scores,
            candidate_down_scores=candidate_down_scores,
            progress_callback=progress_callback,
        )
    except Exception:
        report_progress(progress_callback, 78, "高级拼接异常，切换稳态拼接")
        fallback_grid = beam_edge_grid(
            ids,
            rows,
            cols,
            candidate_right_scores or right_scores,
            candidate_down_scores or down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            beam_width=3 if len(ids) > 160 else 6,
            direction="row",
        )
        grid_layout, best_score = optimize_edge_grid(
            fallback_grid,
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            max_passes=1 if len(ids) > 160 else 3,
        )

    for piece in pieces:
        piece.matched_cell = None
        piece.score = None

    for row, line in enumerate(grid_layout):
        for col, piece_id in enumerate(line):
            if piece_id is None:
                continue
            piece = by_id[piece_id]
            piece.x = col * typical_width
            piece.y = row * typical_height
            piece.matched_cell = (row, col)

    average_score = best_score / max(1, len(pieces))
    report_progress(progress_callback, 100, "边缘拼接完成")
    return SolveResult(
        "edges",
        rows,
        cols,
        max(1, cols * typical_width),
        max(1, rows * typical_height),
        average_score,
    )


def confidence_adjust_scores(
    ids: list[int],
    scores: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    if len(ids) < 3:
        return scores

    values = list(scores.values())
    score_scale = max(1.0, median(values) if values else 1.0)
    outgoing = build_direction_rank(ids, scores, outgoing=True)
    incoming = build_direction_rank(ids, scores, outgoing=False)
    adjusted: dict[tuple[int, int], float] = {}

    for first_id in ids:
        best_out, second_out, best_out_id = outgoing[first_id]
        out_margin = max(0.0, second_out - best_out)
        out_ambiguity = score_scale / (1.0 + out_margin / score_scale)
        for second_id in ids:
            if first_id == second_id:
                continue
            raw_score = scores[(first_id, second_id)]
            best_in, second_in, best_in_id = incoming[second_id]
            in_margin = max(0.0, second_in - best_in)
            in_ambiguity = score_scale / (1.0 + in_margin / score_scale)
            relative_penalty = max(0.0, raw_score - best_out) * 0.42 + max(0.0, raw_score - best_in) * 0.42
            ambiguity_penalty = (out_ambiguity + in_ambiguity) * 0.045
            mutual_bonus = score_scale * 0.08 if best_out_id == second_id and best_in_id == first_id else 0.0
            adjusted[(first_id, second_id)] = max(0.0, raw_score + relative_penalty + ambiguity_penalty - mutual_bonus)

    return adjusted


def gaps_adjust_scores(
    ids: list[int],
    scores: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    if len(ids) < 3:
        return scores

    values = list(scores.values())
    score_scale = max(1.0, median(values) if values else 1.0)
    outgoing = build_direction_rank(ids, scores, outgoing=True)
    incoming = build_direction_rank(ids, scores, outgoing=False)
    adjusted: dict[tuple[int, int], float] = {}

    for first_id in ids:
        best_out, second_out, best_out_id = outgoing[first_id]
        out_gap = max(score_scale * 0.04, second_out - best_out)
        for second_id in ids:
            if first_id == second_id:
                continue
            raw_score = scores[(first_id, second_id)]
            best_in, second_in, best_in_id = incoming[second_id]
            in_gap = max(score_scale * 0.04, second_in - best_in)
            out_rank_cost = max(0.0, raw_score - best_out) / out_gap
            in_rank_cost = max(0.0, raw_score - best_in) / in_gap
            normalized_cost = (out_rank_cost + in_rank_cost) * 0.5

            buddy_factor = 1.0
            if best_out_id == second_id and best_in_id == first_id:
                buddy_factor = 0.72
            elif best_out_id == second_id or best_in_id == first_id:
                buddy_factor = 0.86

            ambiguity = score_scale / (1.0 + (out_gap + in_gap) / (score_scale * 0.5))
            adjusted[(first_id, second_id)] = raw_score * buddy_factor + normalized_cost * score_scale * 0.18 + ambiguity * 0.08

    return adjusted


def build_direction_rank(
    ids: list[int],
    scores: dict[tuple[int, int], float],
    outgoing: bool,
) -> dict[int, tuple[float, float, int]]:
    ranks: dict[int, tuple[float, float, int]] = {}
    for piece_id in ids:
        if outgoing:
            candidates = sorted((scores[(piece_id, other_id)], other_id) for other_id in ids if other_id != piece_id)
        else:
            candidates = sorted((scores[(other_id, piece_id)], other_id) for other_id in ids if other_id != piece_id)
        if not candidates:
            ranks[piece_id] = (1.0, 1.0, piece_id)
            continue
        best_score, best_id = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else best_score
        ranks[piece_id] = (best_score, second_score, best_id)
    return ranks


def min_cost_assignment(cost_matrix: list[list[float]]) -> list[int]:
    if not cost_matrix:
        return []

    rows = len(cost_matrix)
    cols = len(cost_matrix[0])
    if cols < rows:
        raise ValueError("可用网格数量少于碎片数量，无法完成唯一匹配")
    if any(len(row) != cols for row in cost_matrix):
        raise ValueError("代价矩阵列数不一致")

    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    p = [0] * (cols + 1)
    way = [0] * (cols + 1)

    for row in range(1, rows + 1):
        p[0] = row
        col0 = 0
        min_values = [float("inf")] * (cols + 1)
        used = [False] * (cols + 1)

        while True:
            used[col0] = True
            row0 = p[col0]
            delta = float("inf")
            col1 = 0
            for col in range(1, cols + 1):
                if used[col]:
                    continue
                current = cost_matrix[row0 - 1][col - 1] - u[row0] - v[col]
                if current < min_values[col]:
                    min_values[col] = current
                    way[col] = col0
                if min_values[col] < delta:
                    delta = min_values[col]
                    col1 = col

            for col in range(cols + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    min_values[col] -= delta

            col0 = col1
            if p[col0] == 0:
                break

        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break

    assignment = [-1] * rows
    for col in range(1, cols + 1):
        if p[col] != 0:
            assignment[p[col] - 1] = col - 1
    return assignment


def genetic_edge_grid(
    ids: list[int],
    rows: int,
    cols: int,
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
    candidate_right_scores: dict[tuple[int, int], float] | None = None,
    candidate_down_scores: dict[tuple[int, int], float] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[list[int | None]], float]:
    population: list[tuple[float, list[list[int | None]]]] = []
    large_puzzle = len(ids) > 180
    beam_width = 10 if len(ids) <= 64 else (3 if large_puzzle else 6)
    directions = ("row", "snake", "col") if large_puzzle else (
        "row",
        "row_reverse",
        "snake",
        "col",
        "col_reverse",
        "col_snake",
    )

    report_progress(progress_callback, 42, "正在横向生成候选行")
    strip_candidate: tuple[float, list[list[int | None]]] | None = None
    strip_sources = [(right_scores, down_scores)]
    if candidate_right_scores is not None and candidate_down_scores is not None:
        strip_sources.append((candidate_right_scores, candidate_down_scores))

    for source_index, (strip_right_scores, strip_down_scores) in _builtin_enumerate(strip_sources):
        if source_index:
            report_progress(progress_callback, 46, "正在生成置信度候选行")
        strip_grid = strip_edge_grid(
            ids,
            rows,
            cols,
            strip_right_scores,
            strip_down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
        )
        if strip_grid is None:
            continue
        strip_grid, score = optimize_edge_grid(
            strip_grid,
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            max_passes=1 if large_puzzle else 4,
        )
        population.append((score, strip_grid))
        if strip_candidate is None or score < strip_candidate[0]:
            strip_candidate = (score, copy_grid(strip_grid))

        column_grid = column_strip_edge_grid(
            ids,
            rows,
            cols,
            strip_right_scores,
            strip_down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
        )
        if column_grid is None:
            continue
        column_grid, column_score = optimize_edge_grid(
            column_grid,
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            max_passes=1 if large_puzzle else 4,
        )
        population.append((column_score, column_grid))
        if strip_candidate is None or column_score < strip_candidate[0]:
            strip_candidate = (column_score, copy_grid(column_grid))

    if strip_candidate is not None and large_puzzle:
        report_progress(progress_callback, 78, "正在纵向对齐候选行")
        strip_candidate_grid = optimize_row_alignment(
            strip_candidate[1],
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            max_passes=2,
        )
        report_progress(progress_callback, 83, "正在校准整行偏移")
        strip_candidate_grid, _shift_score = optimize_row_shifts(
            strip_candidate_grid,
            ids,
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            max_shift=min(6, max(1, cols // 3)),
            max_passes=2,
        )
        report_progress(progress_callback, 88, "正在修复局部错段")
        strip_candidate_grid, _segment_score = optimize_row_segment_swaps(
            strip_candidate_grid,
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            max_length=4,
            max_passes=4,
        )
        report_progress(progress_callback, 94, "正在优化低置信区域")
        strip_candidate_grid, _bad_score = optimize_bad_region_swaps(
            strip_candidate_grid,
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            candidate_limit=72,
            max_passes=3,
        )
        best_grid, best_score = optimize_block_swaps(
            strip_candidate_grid,
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            max_passes=0,
        )
        return best_grid, best_score

    report_progress(progress_callback, 45, "正在合并可信局部块")
    buddy_grid = buddy_cluster_edge_grid(
        ids,
        rows,
        cols,
        right_scores,
        down_scores,
        border_left,
        border_right,
        border_top,
        border_bottom,
    )
    if buddy_grid is not None:
        buddy_grid, score = optimize_edge_grid(
            buddy_grid,
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            max_passes=1 if large_puzzle else 4,
        )
        population.append((score, buddy_grid))

    for direction in directions:
        direction_index = directions.index(direction)
        report_progress(
            progress_callback,
            48 + direction_index / max(1, len(directions)) * 20,
            f"正在生成 {direction} 候选布局",
        )
        grid = beam_edge_grid(
            ids,
            rows,
            cols,
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            beam_width=beam_width,
            direction=direction,
        )
        grid, score = optimize_edge_grid(
            grid,
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            max_passes=2 if large_puzzle else 8,
        )
        population.append((score, grid))

    population = unique_ranked_grids(population)[:8]
    generations = 5 if len(ids) <= 80 else (1 if large_puzzle else 3)
    for generation in range(generations):
        report_progress(progress_callback, 72 + generation / max(1, generations) * 18, "正在全局优化候选布局")
        next_population = list(population)
        parents = [grid for _score, grid in population[: min(3 if large_puzzle else 5, len(population))]]
        for index, parent in _builtin_enumerate(parents):
            mutated = mutate_edge_grid(parent, index + generation + 1)
            mutated, score = optimize_edge_grid(
                mutated,
                right_scores,
                down_scores,
                border_left,
                border_right,
                border_top,
                border_bottom,
                max_passes=1 if large_puzzle else 3,
            )
            next_population.append((score, mutated))

        for left_index in range(len(parents)):
            for right_index in range(left_index + 1, len(parents)):
                if large_puzzle and left_index > 0:
                    continue
                child = crossover_edge_grid(parents[left_index], parents[right_index])
                child, score = optimize_edge_grid(
                    child,
                    right_scores,
                    down_scores,
                    border_left,
                    border_right,
                    border_top,
                    border_bottom,
                    max_passes=1 if large_puzzle else 3,
                )
                next_population.append((score, child))

        population = unique_ranked_grids(next_population)[:10]

    best_score, best_grid = population[0]
    if large_puzzle and strip_candidate is not None and strip_candidate[0] <= best_score * 1.65:
        best_score, best_grid = strip_candidate
    best_grid, best_score = optimize_block_swaps(
        best_grid,
        right_scores,
        down_scores,
        border_left,
        border_right,
        border_top,
        border_bottom,
        max_passes=1 if large_puzzle else 2,
    )
    return best_grid, best_score


def strip_edge_grid(
    ids: list[int],
    rows: int,
    cols: int,
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
) -> list[list[int | None]] | None:
    if len(ids) < 4 or cols < 2:
        return None

    candidates = generate_row_candidates(ids, rows, cols, right_scores, border_left, border_right)
    if len(candidates) < rows:
        return None

    grid = select_row_candidate_grid(
        candidates,
        rows,
        cols,
        right_scores,
        down_scores,
        border_left,
        border_right,
        border_top,
        border_bottom,
    )
    if grid is None:
        chains = build_horizontal_chains(ids, cols, right_scores)
        if len(chains) < 2:
            return None

        row_chains = sorted(chains, key=lambda chain: (-len(chain), chain[0]))[:rows]
        order = order_row_chains(row_chains, down_scores, border_top, border_bottom)
        grid = [[None for _col in range(cols)] for _row in range(rows)]
        for row, chain_index in enumerate(order[:rows]):
            chain = row_chains[chain_index]
            start_col = best_chain_start_col(chain, cols, border_left, border_right)
            for offset, piece_id in enumerate(chain[:cols]):
                grid[row][start_col + offset] = piece_id

    fill_remaining_grid_cells(
        grid,
        ids,
        rows,
        cols,
        right_scores,
        down_scores,
        border_left,
        border_right,
        border_top,
        border_bottom,
    )
    return grid


def column_strip_edge_grid(
    ids: list[int],
    rows: int,
    cols: int,
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
) -> list[list[int | None]] | None:
    transposed = strip_edge_grid(
        ids,
        cols,
        rows,
        down_scores,
        right_scores,
        border_top,
        border_bottom,
        border_left,
        border_right,
    )
    if transposed is None:
        return None

    grid: list[list[int | None]] = [[None for _col in range(cols)] for _row in range(rows)]
    for col in range(cols):
        for row in range(rows):
            grid[row][col] = transposed[col][row]
    return grid


def generate_row_candidates(
    ids: list[int],
    rows: int,
    cols: int,
    right_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
) -> list[tuple[float, tuple[int, ...]]]:
    neighbor_limit = _builtin_min(36, _builtin_max(8, cols * 2), len(ids) - 1)
    beam_width = 12 if len(ids) > 120 else 18
    start_limit = len(ids)
    min_candidate_length = cols if cols <= 6 else _builtin_max(4, cols - _builtin_min(3, _builtin_max(2, cols // 6)))
    right_neighbors = {
        piece_id: [other_id for _score, other_id in sorted((right_scores[(piece_id, other_id)], other_id) for other_id in ids if other_id != piece_id)]
        for piece_id in ids
    }
    left_neighbors = {
        piece_id: [other_id for _score, other_id in sorted((right_scores[(other_id, piece_id)], other_id) for other_id in ids if other_id != piece_id)]
        for piece_id in ids
    }
    start_ids = sorted(ids, key=lambda piece_id: border_left[piece_id])[:start_limit]
    seen: set[tuple[int, ...]] = set()
    candidates: list[tuple[float, tuple[int, ...]]] = []

    for start_id in start_ids:
        states: list[tuple[float, tuple[int, ...]]] = [(0.0, (start_id,))]
        for _col in range(1, cols):
            next_states: list[tuple[float, tuple[int, ...]]] = []
            for score, chain in states:
                used = set(chain)
                tail_id = chain[-1]
                for candidate_id in right_neighbors[tail_id][:neighbor_limit]:
                    if candidate_id in used:
                        continue
                    next_states.append((score + right_scores[(tail_id, candidate_id)], (*chain, candidate_id)))

                head_id = chain[0]
                for candidate_id in left_neighbors[head_id][:neighbor_limit]:
                    if candidate_id in used:
                        continue
                    next_states.append((score + right_scores[(candidate_id, head_id)], (candidate_id, *chain)))

            if not next_states:
                break
            next_states.sort(key=lambda state: state[0] + border_left[state[1][0]] + border_right[state[1][-1]])
            states = next_states[:beam_width]

            if cols > 6 and _col >= min_candidate_length - 1:
                for score, chain in states:
                    add_row_candidate(candidates, seen, score, chain, cols, border_left, border_right)

        for score, chain in states:
            if len(chain) >= min_candidate_length:
                add_row_candidate(candidates, seen, score, chain, cols, border_left, border_right)

    for chain in build_horizontal_chains(ids, cols, right_scores):
        if len(chain) < min_candidate_length:
            continue
        chain_tuple = tuple(chain)
        if chain_tuple in seen:
            continue
        score = (
            border_left[chain[0]]
            + border_right[chain[-1]]
            + sum(right_scores[(chain[index], chain[index + 1])] for index in range(len(chain) - 1))
        )
        add_row_candidate(candidates, seen, score, chain_tuple, cols, border_left, border_right)

    candidates.sort(key=lambda candidate: candidate[0])
    return diversify_row_candidates(candidates, ids, max(rows * 30, 240), max(3, rows // 3))


def add_row_candidate(
    candidates: list[tuple[float, tuple[int, ...]]],
    seen: set[tuple[int, ...]],
    score: float,
    chain: tuple[int, ...],
    cols: int,
    border_left: dict[int, float],
    border_right: dict[int, float],
) -> None:
    if len(chain) > cols or chain in seen:
        return
    if cols <= 6 and len(chain) != cols:
        return

    seen.add(chain)
    missing = cols - len(chain)
    normalized = (score + border_left[chain[0]] + border_right[chain[-1]]) / max(1, len(chain))
    candidates.append((normalized * (1.0 + missing * 0.16), chain))


def diversify_row_candidates(
    candidates: list[tuple[float, tuple[int, ...]]],
    ids: list[int],
    limit: int,
    per_piece_limit: int,
) -> list[tuple[float, tuple[int, ...]]]:
    selected: list[tuple[float, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    uncovered = set(ids)
    for candidate in candidates:
        _score, chain = candidate
        if chain in seen or not (set(chain) & uncovered):
            continue
        selected.append(candidate)
        seen.add(chain)
        uncovered.difference_update(chain)
        if not uncovered:
            break

    usage = {piece_id: 0 for piece_id in ids}
    for _score, chain in selected:
        for piece_id in chain:
            usage[piece_id] += 1

    for candidate in candidates:
        _score, chain = candidate
        if chain in seen:
            continue
        if all(usage[piece_id] >= per_piece_limit for piece_id in chain):
            continue
        selected.append(candidate)
        seen.add(chain)
        for piece_id in chain:
            usage[piece_id] += 1
        if len(selected) >= limit:
            return selected

    for candidate in candidates:
        if candidate[1] in seen:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def select_row_candidate_grid(
    candidates: list[tuple[float, tuple[int, ...]]],
    rows: int,
    cols: int,
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
) -> list[list[int | None]] | None:
    placements = build_shifted_row_placements(
        candidates,
        cols,
        right_scores,
        border_left,
        border_right,
        border_top,
        border_bottom,
    )
    if len(placements) < rows:
        return None

    beam_width = 260 if len(candidates) > 100 else 120
    score_scale = max(1.0, median(placement[0] for placement in placements[: min(len(placements), 120)]))
    top_layers = build_anchored_row_layers(placements, rows, cols, down_scores, beam_width, "top", score_scale)
    bottom_layers = build_anchored_row_layers(placements, rows, cols, down_scores, beam_width, "bottom", score_scale)
    best_option = choose_anchored_row_option(placements, rows, cols, top_layers, bottom_layers, down_scores, score_scale)
    if best_option is None:
        return None

    _best_score, best_chosen, orientation, split = best_option
    grid: list[list[int | None]] = [[None for _col in range(cols)] for _row in range(rows)]
    place_row_option(grid, placements, best_chosen, orientation, split)
    return grid


def build_anchored_row_layers(
    placements: list[tuple[float, int, int, tuple[int | None, ...], frozenset[int], float, float]],
    rows: int,
    cols: int,
    down_scores: dict[tuple[int, int], float],
    beam_width: int,
    anchor: str,
    score_scale: float,
) -> list[list[tuple[float, tuple[int, ...], frozenset[int]]]]:
    layers: list[list[tuple[float, tuple[int, ...], frozenset[int]]]] = [[(0.0, tuple(), frozenset())]]
    for depth in range(rows):
        next_states: list[tuple[float, tuple[int, ...], frozenset[int]]] = []
        for score, chosen, used in layers[-1]:
            previous_placement = chosen[-1] if chosen else None
            for placement_index, placement in _builtin_enumerate(placements):
                row_score, _candidate_index, _offset, column_ids, placement_set, top_cost, bottom_cost = placement
                if placement_set & used:
                    continue
                if previous_placement is None:
                    increment = top_cost if anchor == "top" else bottom_cost
                elif anchor == "top":
                    previous_columns = placements[previous_placement][3]
                    increment = row_down_candidate_score(previous_columns, column_ids, down_scores)
                else:
                    previous_columns = placements[previous_placement][3]
                    increment = row_down_candidate_score(column_ids, previous_columns, down_scores)
                if increment == float("inf"):
                    continue

                missing = cols - len(placement_set)
                increment += row_score * 0.75 + missing * score_scale * 0.18
                next_states.append((score + increment, (*chosen, placement_index), used | placement_set))

        if not next_states:
            break
        target_cells = (depth + 1) * cols
        next_states.sort(key=lambda state: state[0] + max(0, target_cells - len(state[2])) * score_scale * 0.35)
        layers.append(next_states[:beam_width])

    return layers


def choose_anchored_row_option(
    placements: list[tuple[float, int, int, tuple[int | None, ...], frozenset[int], float, float]],
    rows: int,
    cols: int,
    top_layers: list[list[tuple[float, tuple[int, ...], frozenset[int]]]],
    bottom_layers: list[list[tuple[float, tuple[int, ...], frozenset[int]]]],
    down_scores: dict[tuple[int, int], float],
    score_scale: float,
) -> tuple[float, tuple[int, ...], str, int] | None:
    best: tuple[float, tuple[int, ...], str, int] | None = None

    if len(top_layers) > rows:
        for score, chosen, used in top_layers[rows]:
            if not chosen:
                continue
            total = score + placements[chosen[-1]][6] + coverage_penalty(rows * cols, len(used), score_scale)
            best = min_row_option(best, (total, chosen, "top", rows))

    if len(bottom_layers) > rows:
        for score, chosen, used in bottom_layers[rows]:
            if not chosen:
                continue
            total = score + placements[chosen[-1]][5] + coverage_penalty(rows * cols, len(used), score_scale)
            best = min_row_option(best, (total, chosen, "bottom", 0))

    combine_limit = 160
    for split in range(1, rows):
        bottom_count = rows - split
        if len(top_layers) <= split or len(bottom_layers) <= bottom_count:
            continue
        for top_score, top_chosen, top_used in top_layers[split][:combine_limit]:
            if not top_chosen:
                continue
            top_columns = placements[top_chosen[-1]][3]
            for bottom_score, bottom_chosen, bottom_used in bottom_layers[bottom_count][:combine_limit]:
                if not bottom_chosen or top_used & bottom_used:
                    continue
                bottom_top_columns = placements[bottom_chosen[-1]][3]
                seam_score = row_down_candidate_score(top_columns, bottom_top_columns, down_scores)
                if seam_score == float("inf"):
                    continue
                used_count = len(top_used | bottom_used)
                total = (
                    top_score
                    + bottom_score
                    + seam_score
                    + coverage_penalty(rows * cols, used_count, score_scale)
                    + abs(split - rows / 2) * score_scale * 0.02
                )
                best = min_row_option(best, (total, (*top_chosen, *bottom_chosen), "meet", split))

    return best


def min_row_option(
    current: tuple[float, tuple[int, ...], str, int] | None,
    candidate: tuple[float, tuple[int, ...], str, int],
) -> tuple[float, tuple[int, ...], str, int]:
    if current is None or candidate[0] < current[0]:
        return candidate
    return current


def coverage_penalty(total_cells: int, used_count: int, score_scale: float) -> float:
    return _builtin_max(0, total_cells - used_count) * score_scale * 0.45


def place_row_option(
    grid: list[list[int | None]],
    placements: list[tuple[float, int, int, tuple[int | None, ...], frozenset[int], float, float]],
    chosen: tuple[int, ...],
    orientation: str,
    split: int,
) -> None:
    rows = len(grid)
    if orientation == "top":
        for row, placement_index in _builtin_enumerate(chosen[:rows]):
            place_row_placement(grid, row, placements[placement_index][3])
        return

    if orientation == "bottom":
        for depth, placement_index in _builtin_enumerate(chosen[:rows]):
            place_row_placement(grid, rows - 1 - depth, placements[placement_index][3])
        return

    top_chosen = chosen[:split]
    bottom_chosen = chosen[split:]
    for row, placement_index in _builtin_enumerate(top_chosen):
        place_row_placement(grid, row, placements[placement_index][3])
    for offset, placement_index in _builtin_enumerate(reversed(bottom_chosen)):
        place_row_placement(grid, split + offset, placements[placement_index][3])


def place_row_placement(
    grid: list[list[int | None]],
    row: int,
    column_ids: tuple[int | None, ...],
) -> None:
    for col, piece_id in _builtin_enumerate(column_ids):
        grid[row][col] = piece_id


def build_shifted_row_placements(
    candidates: list[tuple[float, tuple[int, ...]]],
    cols: int,
    right_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
) -> list[tuple[float, int, int, tuple[int | None, ...], frozenset[int], float, float]]:
    max_shift = _builtin_min(cols - 1, _builtin_max(2, cols // 4))
    min_placed = _builtin_max(2, cols - _builtin_min(3, max_shift))
    placements: list[tuple[float, int, int, tuple[int | None, ...], frozenset[int], float, float]] = []
    seen: set[tuple[int | None, ...]] = set()

    for candidate_index, (_candidate_score, chain) in _builtin_enumerate(candidates):
        for offset in range(-max_shift, max_shift + 1):
            column_ids: list[int | None] = [None] * cols
            for chain_index, piece_id in _builtin_enumerate(chain):
                col = chain_index + offset
                if 0 <= col < cols:
                    column_ids[col] = piece_id

            column_tuple = tuple(column_ids)
            if column_tuple in seen:
                continue
            placed = [piece_id for piece_id in column_tuple if piece_id is not None]
            if len(placed) < min_placed:
                continue

            seen.add(column_tuple)
            placement_set = frozenset(placed)
            horizontal_score = row_horizontal_placement_score(column_tuple, right_scores, border_left, border_right)
            top_cost = sum(border_top[piece_id] for piece_id in placed) / max(1, len(placed))
            bottom_cost = sum(border_bottom[piece_id] for piece_id in placed) / max(1, len(placed))
            placements.append((horizontal_score, candidate_index, offset, column_tuple, placement_set, top_cost, bottom_cost))

    placements.sort(key=lambda placement: placement[0])
    return placements


def row_horizontal_placement_score(
    column_ids: tuple[int | None, ...],
    right_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
) -> float:
    total = 0.0
    parts = 0.0
    cols = len(column_ids)
    first_id = column_ids[0] if cols else None
    last_id = column_ids[-1] if cols else None
    if first_id is not None:
        total += border_left[first_id]
        parts += 1.0
    if last_id is not None:
        total += border_right[last_id]
        parts += 1.0

    for col in range(cols - 1):
        left_id = column_ids[col]
        right_id = column_ids[col + 1]
        if left_id is None or right_id is None:
            continue
        total += right_scores[(left_id, right_id)]
        parts += 1.0

    placed = sum(1 for piece_id in column_ids if piece_id is not None)
    missing = cols - placed
    return (total / max(1.0, parts)) * (1.0 + missing * 0.45)


def row_down_candidate_score(
    upper_columns: tuple[int | None, ...],
    lower_columns: tuple[int | None, ...],
    down_scores: dict[tuple[int, int], float],
) -> float:
    total = 0.0
    overlap = 0
    limit = min(len(upper_columns), len(lower_columns))
    for index in range(limit):
        upper_id = upper_columns[index]
        lower_id = lower_columns[index]
        if upper_id is None or lower_id is None:
            continue
        total += down_scores[(upper_id, lower_id)]
        overlap += 1
    if overlap <= 0:
        return float("inf")
    missing = limit - overlap
    return (total / overlap) * (1.0 + missing * 0.18)


def build_horizontal_chains(
    ids: list[int],
    cols: int,
    right_scores: dict[tuple[int, int], float],
    choices_per_piece: int = 10,
) -> list[list[int]]:
    chains: list[list[int]] = [[piece_id] for piece_id in ids]
    piece_chain = {piece_id: index for index, piece_id in _builtin_enumerate(ids)}
    candidates: list[tuple[float, int, int]] = []

    for piece_id in ids:
        best = sorted((right_scores[(piece_id, other_id)], other_id) for other_id in ids if other_id != piece_id)
        for score, other_id in best[:choices_per_piece]:
            candidates.append((score, piece_id, other_id))

    for _score, left_id, right_id in sorted(candidates):
        left_chain_index = piece_chain[left_id]
        right_chain_index = piece_chain[right_id]
        if left_chain_index == right_chain_index:
            continue
        left_chain = chains[left_chain_index]
        right_chain = chains[right_chain_index]
        if not left_chain or not right_chain:
            continue
        if left_chain[-1] != left_id or right_chain[0] != right_id:
            continue
        if len(left_chain) + len(right_chain) > cols:
            continue

        left_chain.extend(right_chain)
        for piece_id in right_chain:
            piece_chain[piece_id] = left_chain_index
        chains[right_chain_index] = []

    return [chain for chain in chains if chain]


def optimize_row_alignment(
    grid: list[list[int | None]],
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
    max_passes: int = 2,
) -> list[list[int | None]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows <= 1 or cols <= 1:
        return grid

    for _pass_index in range(max_passes):
        improved = False
        for row in range(rows):
            best_delta = 0.0
            best_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
            for col in range(cols - 1):
                first = (row, col)
                second = (row, col + 1)
                if grid[row][col] is None or grid[row][col + 1] is None:
                    continue

                affected = affected_score_positions(grid, first, second)
                before = local_edge_score(
                    grid,
                    affected,
                    right_scores,
                    down_scores,
                    border_left,
                    border_right,
                    border_top,
                    border_bottom,
                )
                swap_cells(grid, first, second)
                after = local_edge_score(
                    grid,
                    affected,
                    right_scores,
                    down_scores,
                    border_left,
                    border_right,
                    border_top,
                    border_bottom,
                )
                swap_cells(grid, first, second)

                delta = after - before
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (first, second)

            if best_pair is not None:
                swap_cells(grid, best_pair[0], best_pair[1])
                improved = True

        if not improved:
            break

    return grid


def optimize_row_shifts(
    grid: list[list[int | None]],
    ids: list[int],
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
    max_shift: int = 4,
    max_passes: int = 1,
) -> tuple[list[list[int | None]], float]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows <= 1 or cols <= 2 or max_shift <= 0:
        return grid, edge_grid_score(grid, right_scores, down_scores, border_left, border_right, border_top, border_bottom)

    best_grid = copy_grid(grid)
    best_score = edge_grid_score(best_grid, right_scores, down_scores, border_left, border_right, border_top, border_bottom)
    shift_values = [shift for shift in range(-max_shift, max_shift + 1) if shift != 0]

    for _pass_index in range(max_passes):
        improved = False
        for row in range(rows):
            row_best_grid = best_grid
            row_best_score = best_score
            for shift in shift_values:
                candidate = shifted_row_grid(best_grid, row, shift)
                fill_remaining_grid_cells(
                    candidate,
                    ids,
                    rows,
                    cols,
                    right_scores,
                    down_scores,
                    border_left,
                    border_right,
                    border_top,
                    border_bottom,
                )
                candidate_score = edge_grid_score(
                    candidate,
                    right_scores,
                    down_scores,
                    border_left,
                    border_right,
                    border_top,
                    border_bottom,
                )
                if candidate_score < row_best_score:
                    row_best_grid = candidate
                    row_best_score = candidate_score

            if row_best_score < best_score:
                best_grid = row_best_grid
                best_score = row_best_score
                improved = True

        if not improved:
            break

    return best_grid, best_score


def shifted_row_grid(grid: list[list[int | None]], row: int, shift: int) -> list[list[int | None]]:
    shifted_grid = copy_grid(grid)
    cols = len(shifted_grid[row])
    shifted_row: list[int | None] = [None] * cols
    for col, piece_id in _builtin_enumerate(shifted_grid[row]):
        if piece_id is None:
            continue
        shifted_col = col + shift
        if 0 <= shifted_col < cols:
            shifted_row[shifted_col] = piece_id
    shifted_grid[row] = shifted_row
    return shifted_grid


def optimize_row_segment_swaps(
    grid: list[list[int | None]],
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
    max_length: int = 4,
    max_passes: int = 3,
) -> tuple[list[list[int | None]], float]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows <= 1 or cols <= 3:
        return grid, edge_grid_score(grid, right_scores, down_scores, border_left, border_right, border_top, border_bottom)

    current_score = edge_grid_score(grid, right_scores, down_scores, border_left, border_right, border_top, border_bottom)
    focus_start = rows // 3 if rows * cols > 180 else 0
    max_length = max(2, min(max_length, cols))

    for _pass_index in range(max_passes):
        best_delta = 0.0
        best_swap: tuple[tuple[int, int], tuple[int, int], int] | None = None

        for length in range(2, max_length + 1):
            starts = [(row, col) for row in range(focus_start, rows) for col in range(cols - length + 1)]
            for first_index, first in _builtin_enumerate(starts):
                first_cells = row_segment_cells(first[0], first[1], length)
                for second in starts[first_index + 1 :]:
                    second_cells = row_segment_cells(second[0], second[1], length)
                    if first_cells & second_cells:
                        continue

                    affected = expanded_positions(rows, cols, first_cells | second_cells)
                    before = local_edge_score(
                        grid,
                        affected,
                        right_scores,
                        down_scores,
                        border_left,
                        border_right,
                        border_top,
                        border_bottom,
                    )
                    swap_row_segments(grid, first, second, length)
                    after = local_edge_score(
                        grid,
                        affected,
                        right_scores,
                        down_scores,
                        border_left,
                        border_right,
                        border_top,
                        border_bottom,
                    )
                    swap_row_segments(grid, first, second, length)

                    delta = after - before
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (first, second, length)

        if best_swap is None:
            break

        first, second, length = best_swap
        swap_row_segments(grid, first, second, length)
        current_score += best_delta

    return grid, current_score


def row_segment_cells(row: int, col: int, length: int) -> set[tuple[int, int]]:
    return {(row, col + offset) for offset in range(length)}


def swap_row_segments(
    grid: list[list[int | None]],
    first: tuple[int, int],
    second: tuple[int, int],
    length: int,
) -> None:
    first_row, first_col = first
    second_row, second_col = second
    for offset in range(length):
        swap_cells(grid, (first_row, first_col + offset), (second_row, second_col + offset))


def optimize_bad_region_swaps(
    grid: list[list[int | None]],
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
    candidate_limit: int = 64,
    max_passes: int = 2,
) -> tuple[list[list[int | None]], float]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows <= 1 or cols <= 1:
        return grid, edge_grid_score(grid, right_scores, down_scores, border_left, border_right, border_top, border_bottom)

    current_score = edge_grid_score(grid, right_scores, down_scores, border_left, border_right, border_top, border_bottom)
    for _pass_index in range(max_passes):
        positions = ranked_bad_positions(
            grid,
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            candidate_limit,
        )
        best_delta = 0.0
        best_pair: tuple[tuple[int, int], tuple[int, int]] | None = None

        for first_index, first in _builtin_enumerate(positions):
            for second in positions[first_index + 1 :]:
                if grid[first[0]][first[1]] is None or grid[second[0]][second[1]] is None:
                    continue
                affected = affected_score_positions(grid, first, second)
                before = local_edge_score(
                    grid,
                    affected,
                    right_scores,
                    down_scores,
                    border_left,
                    border_right,
                    border_top,
                    border_bottom,
                )
                swap_cells(grid, first, second)
                after = local_edge_score(
                    grid,
                    affected,
                    right_scores,
                    down_scores,
                    border_left,
                    border_right,
                    border_top,
                    border_bottom,
                )
                swap_cells(grid, first, second)

                delta = after - before
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (first, second)

        if best_pair is None:
            break

        swap_cells(grid, best_pair[0], best_pair[1])
        current_score += best_delta

    return grid, current_score


def ranked_bad_positions(
    grid: list[list[int | None]],
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
    limit: int,
) -> list[tuple[int, int]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    focus_start = rows // 3 if rows * cols > 180 else 0
    scored: list[tuple[float, int, int]] = []
    for row in range(focus_start, rows):
        for col in range(cols):
            if grid[row][col] is None:
                continue
            score = edge_position_score(
                grid,
                row,
                col,
                right_scores,
                down_scores,
                border_left,
                border_right,
                border_top,
                border_bottom,
            )
            scored.append((score, row, col))
    scored.sort(reverse=True)
    return [(row, col) for _score, row, col in scored[:limit]]


def order_row_chains(
    chains: list[list[int]],
    down_scores: dict[tuple[int, int], float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
    beam_width: int = 24,
) -> list[int]:
    if len(chains) <= 1:
        return [0] if chains else []

    top_cost = [sum(border_top[piece_id] for piece_id in chain) / max(1, len(chain)) for chain in chains]
    bottom_cost = [sum(border_bottom[piece_id] for piece_id in chain) / max(1, len(chain)) for chain in chains]
    between_cost: dict[tuple[int, int], float] = {}
    for first_index, first_chain in _builtin_enumerate(chains):
        for second_index, second_chain in _builtin_enumerate(chains):
            if first_index == second_index:
                continue
            overlap = min(len(first_chain), len(second_chain))
            between_cost[(first_index, second_index)] = sum(
                down_scores[(first_chain[col], second_chain[col])] for col in range(overlap)
            ) / max(1, overlap)

    states: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = [(0.0, tuple(), tuple(range(len(chains))))]
    for row_index in range(len(chains)):
        next_states: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
        for score, ordered, remaining in states:
            previous = ordered[-1] if ordered else None
            for candidate in remaining:
                if previous is None:
                    increment = top_cost[candidate]
                else:
                    increment = between_cost[(previous, candidate)]
                if row_index == len(chains) - 1:
                    increment += bottom_cost[candidate]
                next_remaining = tuple(index for index in remaining if index != candidate)
                next_states.append((score + increment, (*ordered, candidate), next_remaining))
        next_states.sort(key=lambda state: state[0])
        states = next_states[: max(1, beam_width)]

    return list(min(states, key=lambda state: state[0])[1])


def best_chain_start_col(chain: list[int], cols: int, border_left: dict[int, float], border_right: dict[int, float]) -> int:
    if len(chain) >= cols:
        return 0
    best_score = float("inf")
    best_start = 0
    for start_col in range(cols - len(chain) + 1):
        score = 0.0
        if start_col == 0:
            score += border_left[chain[0]]
        if start_col + len(chain) == cols:
            score += border_right[chain[-1]]
        if score < best_score:
            best_score = score
            best_start = start_col
    return best_start


def buddy_cluster_edge_grid(
    ids: list[int],
    rows: int,
    cols: int,
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
) -> list[list[int | None]] | None:
    if len(ids) < 2:
        return None

    all_scores = list(right_scores.values()) + list(down_scores.values())
    score_scale = max(1.0, median(all_scores) if all_scores else 1.0)
    best_right = {
        piece_id: min((right_scores[(piece_id, other_id)], other_id) for other_id in ids if other_id != piece_id)[1]
        for piece_id in ids
    }
    best_left = {
        piece_id: min((right_scores[(other_id, piece_id)], other_id) for other_id in ids if other_id != piece_id)[1]
        for piece_id in ids
    }
    best_bottom = {
        piece_id: min((down_scores[(piece_id, other_id)], other_id) for other_id in ids if other_id != piece_id)[1]
        for piece_id in ids
    }
    best_top = {
        piece_id: min((down_scores[(other_id, piece_id)], other_id) for other_id in ids if other_id != piece_id)[1]
        for piece_id in ids
    }

    candidates: list[tuple[float, int, int, int, int]] = []
    for first_id in ids:
        right_id = best_right[first_id]
        right_score = right_scores[(first_id, right_id)]
        if best_left[right_id] == first_id and right_score <= score_scale * 0.9:
            candidates.append((right_score, first_id, right_id, 0, 1))

        bottom_id = best_bottom[first_id]
        down_score = down_scores[(first_id, bottom_id)]
        if best_top[bottom_id] == first_id and down_score <= score_scale * 0.9:
            candidates.append((down_score, first_id, bottom_id, 1, 0))

    if not candidates:
        return None

    clusters: dict[int, dict[int, tuple[int, int]]] = {piece_id: {piece_id: (0, 0)} for piece_id in ids}
    piece_cluster = {piece_id: piece_id for piece_id in ids}
    for _score, first_id, second_id, row_offset, col_offset in sorted(candidates):
        merge_buddy_clusters(clusters, piece_cluster, first_id, second_id, row_offset, col_offset, rows, cols)

    return pack_cluster_grid(
        clusters,
        ids,
        rows,
        cols,
        right_scores,
        down_scores,
        border_left,
        border_right,
        border_top,
        border_bottom,
        score_scale,
    )


def merge_buddy_clusters(
    clusters: dict[int, dict[int, tuple[int, int]]],
    piece_cluster: dict[int, int],
    first_id: int,
    second_id: int,
    row_offset: int,
    col_offset: int,
    rows: int,
    cols: int,
) -> bool:
    first_cluster = piece_cluster[first_id]
    second_cluster = piece_cluster[second_id]
    first_map = clusters[first_cluster]
    second_map = clusters[second_cluster]
    first_row, first_col = first_map[first_id]

    if first_cluster == second_cluster:
        return second_map[second_id] == (first_row + row_offset, first_col + col_offset)

    second_row, second_col = second_map[second_id]
    translate_row = first_row + row_offset - second_row
    translate_col = first_col + col_offset - second_col
    occupied = set(first_map.values())
    translated: dict[int, tuple[int, int]] = {}
    for piece_id, (row, col) in second_map.items():
        position = (row + translate_row, col + translate_col)
        if position in occupied:
            return False
        translated[piece_id] = position

    merged = {**first_map, **translated}
    row_values = [row for row, _col in merged.values()]
    col_values = [col for _row, col in merged.values()]
    if max(row_values) - min(row_values) + 1 > rows or max(col_values) - min(col_values) + 1 > cols:
        return False

    first_map.update(translated)
    del clusters[second_cluster]
    for piece_id in translated:
        piece_cluster[piece_id] = first_cluster
    return True


def pack_cluster_grid(
    clusters: dict[int, dict[int, tuple[int, int]]],
    ids: list[int],
    rows: int,
    cols: int,
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
    score_scale: float,
) -> list[list[int | None]]:
    grid: list[list[int | None]] = [[None for _col in range(cols)] for _row in range(rows)]
    cluster_maps = sorted(clusters.values(), key=lambda cluster: (-len(cluster), cluster_area(cluster)))

    for cluster in cluster_maps:
        normalized = normalize_cluster_positions(cluster)
        if not normalized:
            continue
        height = max(row for _piece_id, row, _col in normalized) + 1
        width = max(col for _piece_id, _row, col in normalized) + 1
        if height > rows or width > cols:
            continue

        placement = best_cluster_placement(
            grid,
            normalized,
            rows,
            cols,
            height,
            width,
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
            score_scale,
        )
        if placement is None:
            continue
        place_cluster(grid, normalized, placement[0], placement[1])

    fill_remaining_grid_cells(
        grid,
        ids,
        rows,
        cols,
        right_scores,
        down_scores,
        border_left,
        border_right,
        border_top,
        border_bottom,
    )
    return grid


def normalize_cluster_positions(cluster: dict[int, tuple[int, int]]) -> list[tuple[int, int, int]]:
    if not cluster:
        return []
    min_row = min(row for row, _col in cluster.values())
    min_col = min(col for _row, col in cluster.values())
    return [(piece_id, row - min_row, col - min_col) for piece_id, (row, col) in cluster.items()]


def cluster_area(cluster: dict[int, tuple[int, int]]) -> int:
    if not cluster:
        return 0
    rows = [row for row, _col in cluster.values()]
    cols = [col for _row, col in cluster.values()]
    return (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def best_cluster_placement(
    grid: list[list[int | None]],
    cluster: list[tuple[int, int, int]],
    rows: int,
    cols: int,
    height: int,
    width: int,
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
    score_scale: float,
) -> tuple[int, int] | None:
    best_score = float("inf")
    best_position: tuple[int, int] | None = None
    placed_count = sum(1 for row in grid for piece_id in row if piece_id is not None)

    for top in range(rows - height + 1):
        for left in range(cols - width + 1):
            if cluster_collides(grid, cluster, top, left):
                continue
            score = cluster_placement_score(
                grid,
                cluster,
                top,
                left,
                rows,
                cols,
                right_scores,
                down_scores,
                border_left,
                border_right,
                border_top,
                border_bottom,
                score_scale,
                placed_count,
            )
            if score < best_score:
                best_score = score
                best_position = (top, left)

    return best_position


def cluster_collides(grid: list[list[int | None]], cluster: list[tuple[int, int, int]], top: int, left: int) -> bool:
    return any(grid[top + row][left + col] is not None for _piece_id, row, col in cluster)


def cluster_placement_score(
    grid: list[list[int | None]],
    cluster: list[tuple[int, int, int]],
    top: int,
    left: int,
    rows: int,
    cols: int,
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
    score_scale: float,
    placed_count: int,
) -> float:
    score = 0.0
    parts = 0.0
    contacts = 0
    cluster_by_position = {(top + row, left + col): piece_id for piece_id, row, col in cluster}

    for piece_id, row, col in cluster:
        absolute_row = top + row
        absolute_col = left + col
        neighbors = (
            (absolute_row, absolute_col - 1, "left"),
            (absolute_row, absolute_col + 1, "right"),
            (absolute_row - 1, absolute_col, "top"),
            (absolute_row + 1, absolute_col, "bottom"),
        )
        for neighbor_row, neighbor_col, side in neighbors:
            if (neighbor_row, neighbor_col) in cluster_by_position:
                continue
            if neighbor_col < 0:
                score += border_left[piece_id]
                parts += 1.0
            elif neighbor_col >= cols:
                score += border_right[piece_id]
                parts += 1.0
            elif neighbor_row < 0:
                score += border_top[piece_id]
                parts += 1.0
            elif neighbor_row >= rows:
                score += border_bottom[piece_id]
                parts += 1.0
            else:
                neighbor_id = grid[neighbor_row][neighbor_col]
                if neighbor_id is None:
                    continue
                contacts += 1
                parts += 1.0
                if side == "left":
                    score += right_scores[(neighbor_id, piece_id)]
                elif side == "right":
                    score += right_scores[(piece_id, neighbor_id)]
                elif side == "top":
                    score += down_scores[(neighbor_id, piece_id)]
                else:
                    score += down_scores[(piece_id, neighbor_id)]

    if placed_count and contacts == 0:
        score += score_scale * 0.2
        parts += 1.0
    return score / max(1.0, parts)


def place_cluster(grid: list[list[int | None]], cluster: list[tuple[int, int, int]], top: int, left: int) -> None:
    for piece_id, row, col in cluster:
        grid[top + row][left + col] = piece_id


def fill_remaining_grid_cells(
    grid: list[list[int | None]],
    ids: list[int],
    rows: int,
    cols: int,
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
) -> None:
    placed = {piece_id for row in grid for piece_id in row if piece_id is not None}
    remaining = [piece_id for piece_id in ids if piece_id not in placed]
    if not remaining:
        return

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] is not None or not remaining:
                continue
            placed_map = {
                (placed_row, placed_col): piece_id
                for placed_row in range(rows)
                for placed_col, piece_id in enumerate(grid[placed_row])
                if piece_id is not None
            }
            remaining_tuple = tuple(remaining)
            best_piece = min(
                remaining,
                key=lambda piece_id: beam_cell_increment(
                    placed_map,
                    remaining_tuple,
                    piece_id,
                    row,
                    col,
                    rows,
                    cols,
                    right_scores,
                    down_scores,
                    border_left,
                    border_right,
                    border_top,
                    border_bottom,
                ),
            )
            grid[row][col] = best_piece
            remaining.remove(best_piece)


def optimize_block_swaps(
    grid: list[list[int | None]],
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
    max_passes: int = 1,
) -> tuple[list[list[int | None]], float]:
    current_score = edge_grid_score(grid, right_scores, down_scores, border_left, border_right, border_top, border_bottom)
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows <= 1 and cols <= 1:
        return grid, current_score

    shapes = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]
    if rows * cols <= 120:
        shapes.extend([(1, 4), (4, 1), (2, 3), (3, 2)])
    else:
        shapes.extend([(1, 4), (4, 1), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3)])

    for _pass_index in range(max_passes):
        best_delta = 0.0
        best_swap: tuple[tuple[int, int], tuple[int, int], int, int] | None = None

        for height, width in shapes:
            if height > rows or width > cols:
                continue
            starts = [(row, col) for row in range(rows - height + 1) for col in range(cols - width + 1)]
            for index, first in _builtin_enumerate(starts):
                first_cells = block_cells(first[0], first[1], height, width)
                for second in starts[index + 1 :]:
                    second_cells = block_cells(second[0], second[1], height, width)
                    if first_cells & second_cells:
                        continue
                    if block_is_empty(grid, first_cells) and block_is_empty(grid, second_cells):
                        continue

                    affected = expanded_positions(rows, cols, first_cells | second_cells)
                    before = local_edge_score(
                        grid,
                        affected,
                        right_scores,
                        down_scores,
                        border_left,
                        border_right,
                        border_top,
                        border_bottom,
                    )
                    swap_blocks(grid, first, second, height, width)
                    after = local_edge_score(
                        grid,
                        affected,
                        right_scores,
                        down_scores,
                        border_left,
                        border_right,
                        border_top,
                        border_bottom,
                    )
                    swap_blocks(grid, first, second, height, width)

                    delta = after - before
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (first, second, height, width)

        if best_swap is None:
            break

        first, second, height, width = best_swap
        swap_blocks(grid, first, second, height, width)
        current_score += best_delta

    return grid, current_score


def block_cells(row: int, col: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(row + row_offset, col + col_offset) for row_offset in range(height) for col_offset in range(width)}


def block_is_empty(grid: list[list[int | None]], cells: set[tuple[int, int]]) -> bool:
    return all(grid[row][col] is None for row, col in cells)


def expanded_positions(rows: int, cols: int, cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    positions: set[tuple[int, int]] = set()
    for row, col in cells:
        for candidate in ((row, col), (row, col - 1), (row - 1, col), (row, col + 1), (row + 1, col)):
            candidate_row, candidate_col = candidate
            if 0 <= candidate_row < rows and 0 <= candidate_col < cols:
                positions.add(candidate)
    return positions


def swap_blocks(grid: list[list[int | None]], first: tuple[int, int], second: tuple[int, int], height: int, width: int) -> None:
    first_row, first_col = first
    second_row, second_col = second
    for row_offset in range(height):
        for col_offset in range(width):
            first_position = (first_row + row_offset, first_col + col_offset)
            second_position = (second_row + row_offset, second_col + col_offset)
            swap_cells(grid, first_position, second_position)


def unique_ranked_grids(population: list[tuple[float, list[list[int | None]]]]) -> list[tuple[float, list[list[int | None]]]]:
    seen: set[tuple[int | None, ...]] = set()
    ranked: list[tuple[float, list[list[int | None]]]] = []
    for score, grid in sorted(population, key=lambda item: item[0]):
        signature = tuple(piece_id for row in grid for piece_id in row)
        if signature in seen:
            continue
        seen.add(signature)
        ranked.append((score, copy_grid(grid)))
    return ranked


def mutate_edge_grid(grid: list[list[int | None]], seed: int) -> list[list[int | None]]:
    mutated = copy_grid(grid)
    positions = [(row, col) for row in range(len(mutated)) for col in range(len(mutated[row]))]
    if len(positions) < 2:
        return mutated
    swaps = max(1, min(6, len(positions) // 8))
    for index in range(swaps):
        first = positions[(seed * 7 + index * 11) % len(positions)]
        second = positions[(seed * 13 + index * 17 + 3) % len(positions)]
        if first != second:
            swap_cells(mutated, first, second)
    return mutated


def crossover_edge_grid(first: list[list[int | None]], second: list[list[int | None]]) -> list[list[int | None]]:
    rows = len(first)
    cols = len(first[0]) if rows else 0
    total = rows * cols
    child_flat: list[int | None] = [None] * total
    used: set[int] = set()
    split = max(1, total // 2)
    first_flat = [piece_id for row in first for piece_id in row]
    second_flat = [piece_id for row in second for piece_id in row]

    for index, piece_id in enumerate(first_flat[:split]):
        child_flat[index] = piece_id
        if piece_id is not None:
            used.add(piece_id)

    write_index = 0
    for piece_id in second_flat:
        if piece_id is None:
            continue
        if piece_id in used:
            continue
        while write_index < total and child_flat[write_index] is not None:
            write_index += 1
        if write_index >= total:
            break
        child_flat[write_index] = piece_id
        used.add(piece_id)

    return [child_flat[row * cols : (row + 1) * cols] for row in range(rows)]


def copy_grid(grid: list[list[int | None]]) -> list[list[int | None]]:
    return [list(row) for row in grid]


def beam_edge_grid(
    ids: list[int],
    rows: int,
    cols: int,
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
    beam_width: int = 8,
    direction: str = "row",
) -> list[list[int | None]]:
    total_cells = rows * cols
    initial_state = (0.0, tuple(), tuple(ids))
    states: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = [initial_state]
    positions = traversal_positions(rows, cols, direction)

    for cell_index in range(min(total_cells, len(ids))):
        row, col = positions[cell_index]
        next_states: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []

        for score, placed, remaining in states:
            placed_map = {positions[index]: piece_id for index, piece_id in enumerate(placed)}
            for candidate_id in remaining:
                increment = beam_cell_increment(
                    placed_map,
                    remaining,
                    candidate_id,
                    row,
                    col,
                    rows,
                    cols,
                    right_scores,
                    down_scores,
                    border_left,
                    border_right,
                    border_top,
                    border_bottom,
                )
                new_remaining = tuple(piece_id for piece_id in remaining if piece_id != candidate_id)
                next_states.append((score + increment, (*placed, candidate_id), new_remaining))

        next_states.sort(key=lambda state: state[0])
        states = next_states[: max(1, beam_width)]

    best_placed = min(states, key=lambda state: state[0])[1] if states else tuple()
    grid: list[list[int | None]] = [[None for _col in range(cols)] for _row in range(rows)]
    for index, piece_id in enumerate(best_placed):
        row, col = positions[index]
        grid[row][col] = piece_id
    return grid


def traversal_positions(rows: int, cols: int, direction: str) -> list[tuple[int, int]]:
    if direction == "row_reverse":
        return [(row, col) for row in reversed(range(rows)) for col in reversed(range(cols))]
    if direction == "snake":
        return [(row, col) for row in range(rows) for col in (range(cols) if row % 2 == 0 else reversed(range(cols)))]
    if direction == "col":
        return [(row, col) for col in range(cols) for row in range(rows)]
    if direction == "col_reverse":
        return [(row, col) for col in reversed(range(cols)) for row in reversed(range(rows))]
    if direction == "col_snake":
        return [(row, col) for col in range(cols) for row in (range(rows) if col % 2 == 0 else reversed(range(rows)))]
    return [(row, col) for row in range(rows) for col in range(cols)]


def beam_cell_increment(
    placed_map: dict[tuple[int, int], int],
    remaining: tuple[int, ...],
    candidate_id: int,
    row: int,
    col: int,
    rows: int,
    cols: int,
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
) -> float:
    score = 0.0
    parts = 0.0

    left_id = placed_map.get((row, col - 1))
    top_id = placed_map.get((row - 1, col))
    if col == 0:
        score += border_left[candidate_id]
        parts += 1.0
    elif left_id is not None:
        score += right_scores[(left_id, candidate_id)]
        parts += 1.0

    if row == 0:
        score += border_top[candidate_id]
        parts += 1.0
    elif top_id is not None:
        score += down_scores[(top_id, candidate_id)]
        parts += 1.0

    future = tuple(piece_id for piece_id in remaining if piece_id != candidate_id)
    if col == cols - 1:
        score += border_right[candidate_id]
        parts += 1.0
    elif (row, col + 1) not in placed_map and future:
        score += min(right_scores[(candidate_id, other_id)] for other_id in future) * 0.14
        parts += 0.14

    if row == rows - 1:
        score += border_bottom[candidate_id]
        parts += 1.0
    elif (row + 1, col) not in placed_map and future:
        score += min(down_scores[(candidate_id, other_id)] for other_id in future) * 0.14
        parts += 0.14

    return score / max(1.0, parts)


def optimize_edge_grid(
    grid: list[list[int | None]],
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
    max_passes: int = 8,
) -> tuple[list[list[int | None]], float]:
    current_score = edge_grid_score(grid, right_scores, down_scores, border_left, border_right, border_top, border_bottom)
    positions = [(row, col) for row in range(len(grid)) for col in range(len(grid[row]))]

    for _ in range(max_passes):
        best_delta = 0.0
        best_pair: tuple[tuple[int, int], tuple[int, int]] | None = None

        for index, first in _builtin_enumerate(positions):
            for second in positions[index + 1 :]:
                if grid[first[0]][first[1]] is None and grid[second[0]][second[1]] is None:
                    continue

                affected = affected_score_positions(grid, first, second)
                before = local_edge_score(
                    grid,
                    affected,
                    right_scores,
                    down_scores,
                    border_left,
                    border_right,
                    border_top,
                    border_bottom,
                )
                swap_cells(grid, first, second)
                after = local_edge_score(
                    grid,
                    affected,
                    right_scores,
                    down_scores,
                    border_left,
                    border_right,
                    border_top,
                    border_bottom,
                )
                swap_cells(grid, first, second)

                delta = after - before
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (first, second)

        if best_pair is None:
            break

        swap_cells(grid, best_pair[0], best_pair[1])
        current_score += best_delta

    return grid, current_score


def edge_grid_score(
    grid: list[list[int | None]],
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
) -> float:
    total = 0.0
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            total += edge_position_score(
                grid,
                row,
                col,
                right_scores,
                down_scores,
                border_left,
                border_right,
                border_top,
                border_bottom,
            )
    return total


def local_edge_score(
    grid: list[list[int | None]],
    positions: set[tuple[int, int]],
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
) -> float:
    total = 0.0
    for row, col in positions:
        total += edge_position_score(
            grid,
            row,
            col,
            right_scores,
            down_scores,
            border_left,
            border_right,
            border_top,
            border_bottom,
        )
    return total


def edge_position_score(
    grid: list[list[int | None]],
    row: int,
    col: int,
    right_scores: dict[tuple[int, int], float],
    down_scores: dict[tuple[int, int], float],
    border_left: dict[int, float],
    border_right: dict[int, float],
    border_top: dict[int, float],
    border_bottom: dict[int, float],
) -> float:
    if not hasattr(right_scores, "get") or not hasattr(down_scores, "get"):
        return float("inf")

    piece_id = grid[row][col]
    if piece_id is None:
        return 0.0

    rows = len(grid)
    cols = len(grid[row])
    score = 0.0

    left_id = grid[row][col - 1] if col > 0 else None
    if col == 0 or left_id is None:
        score += border_left[piece_id]

    top_id = grid[row - 1][col] if row > 0 else None
    if row == 0 or top_id is None:
        score += border_top[piece_id]

    right_id = grid[row][col + 1] if col + 1 < cols else None
    if col + 1 >= cols or right_id is None:
        score += border_right[piece_id]
    else:
        score += right_scores.get((piece_id, right_id), float("inf"))

    bottom_id = grid[row + 1][col] if row + 1 < rows else None
    if row + 1 >= rows or bottom_id is None:
        score += border_bottom[piece_id]
    else:
        score += down_scores.get((piece_id, bottom_id), float("inf"))

    return score


def affected_score_positions(
    grid: list[list[int | None]],
    first: tuple[int, int],
    second: tuple[int, int],
) -> set[tuple[int, int]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    positions: set[tuple[int, int]] = set()
    for row, col in (first, second):
        for candidate in ((row, col), (row, col - 1), (row - 1, col), (row, col + 1), (row + 1, col)):
            candidate_row, candidate_col = candidate
            if 0 <= candidate_row < rows and 0 <= candidate_col < cols:
                positions.add(candidate)
    return positions


def swap_cells(grid: list[list[int | None]], first: tuple[int, int], second: tuple[int, int]) -> None:
    grid[first[0]][first[1]], grid[second[0]][second[1]] = grid[second[0]][second[1]], grid[first[0]][first[1]]


def compose_image(
    pieces: Iterable[Piece],
    original: Image.Image | None = None,
    include_background: bool = False,
    background_opacity: float = 0.35,
    padding: int = 12,
) -> Image.Image:
    piece_list = list(pieces)
    if not piece_list and original is None:
        return Image.new("RGBA", (1, 1), (255, 255, 255, 0))

    min_x = 0
    min_y = 0
    max_x = original.width if original is not None and include_background else 0
    max_y = original.height if original is not None and include_background else 0

    for piece in piece_list:
        min_x = min(min_x, int(piece.x))
        min_y = min(min_y, int(piece.y))
        max_x = max(max_x, int(piece.x) + piece.width)
        max_y = max(max_y, int(piece.y) + piece.height)

    offset_x = -min_x + padding
    offset_y = -min_y + padding
    width = max(1, max_x - min_x + padding * 2)
    height = max(1, max_y - min_y + padding * 2)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))

    if original is not None and include_background:
        background = with_opacity(original, background_opacity)
        canvas.alpha_composite(background, (offset_x, offset_y))

    for piece in piece_list:
        canvas.alpha_composite(piece.image, (int(piece.x) + offset_x, int(piece.y) + offset_y))

    return canvas


def with_opacity(image: Image.Image, opacity: float) -> Image.Image:
    opacity = max(0.0, min(1.0, opacity))
    rgba = image.convert("RGBA").copy()
    alpha = rgba.getchannel("A")
    alpha = alpha.point(lambda value: int(value * opacity))
    rgba.putalpha(alpha)
    return rgba


def region_score(piece_image: Image.Image, reference_image: Image.Image, sample_size: int = 48) -> float:
    piece = piece_image.convert("RGBA").resize((sample_size, sample_size), Image.Resampling.BILINEAR)
    reference = reference_image.convert("RGBA").resize((sample_size, sample_size), Image.Resampling.BILINEAR)
    piece_pixels = list(piece.getdata())
    reference_pixels = list(reference.getdata())

    total = 0.0
    count = 0
    for source, target in zip(piece_pixels, reference_pixels):
        alpha = source[3] / 255.0
        if alpha <= 0.05:
            continue
        dr = source[0] - target[0]
        dg = source[1] - target[1]
        db = source[2] - target[2]
        total += (dr * dr + dg * dg + db * db) * alpha
        count += 1
    return total / max(1, count)


def extract_edges(image: Image.Image, length: int = 96, thickness: int = 3) -> dict[str, list[tuple[int, int, int]]]:
    rgb = flatten_alpha(image)
    width, height = rgb.size
    t = max(1, min(thickness, width, height))
    crops = {
        "top": rgb.crop((0, 0, width, t)).resize((length, 1), Image.Resampling.BILINEAR),
        "bottom": rgb.crop((0, height - t, width, height)).resize((length, 1), Image.Resampling.BILINEAR),
        "left": rgb.crop((0, 0, t, height)).resize((1, length), Image.Resampling.BILINEAR),
        "right": rgb.crop((width - t, 0, width, height)).resize((1, length), Image.Resampling.BILINEAR),
    }
    return {side: list(crop.getdata()) for side, crop in crops.items()}


def flatten_alpha(image: Image.Image, background: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (*background, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def edge_score(a: list[tuple[int, int, int]], b: list[tuple[int, int, int]]) -> float:
    if not a or not b:
        return float("inf")

    color_total = 0.0
    gradient_total = 0.0
    weight_total = 0.0
    gradient_weight_total = 0.0
    previous_a: tuple[int, int, int] | None = None
    previous_b: tuple[int, int, int] | None = None
    count = 0
    for pixel_a, pixel_b in _builtin_zip(a, b):
        weight = 1.0
        if previous_a is not None and previous_b is not None:
            texture = pixel_distance(previous_a, pixel_a) + pixel_distance(previous_b, pixel_b)
            weight += _builtin_min(3.0, texture / 3500.0)
        color_total += pixel_distance(pixel_a, pixel_b) * weight
        weight_total += weight
        if previous_a is not None and previous_b is not None:
            gradient_weight = _builtin_max(1.0, weight)
            gradient_total += pixel_distance(pixel_delta(previous_a, pixel_a), pixel_delta(previous_b, pixel_b)) * gradient_weight
            gradient_weight_total += gradient_weight
        previous_a = pixel_a
        previous_b = pixel_b
        count += 1

    color_score = color_total / _builtin_max(1.0, weight_total)
    gradient_score = gradient_total / _builtin_max(1.0, gradient_weight_total)
    return color_score + gradient_score * 0.55


def pixel_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    dr = a[0] - b[0]
    dg = a[1] - b[1]
    db = a[2] - b[2]
    return dr * dr + dg * dg + db * db


def pixel_delta(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def normalize_grid(grid: tuple[int, int] | None, piece_count: int) -> tuple[int, int] | None:
    if grid is None:
        return None
    rows, cols = grid
    if rows <= 0 and cols <= 0:
        return None
    if rows <= 0:
        rows = ceil(piece_count / cols)
    if cols <= 0:
        cols = ceil(piece_count / rows)
    if rows * cols < piece_count:
        raise ValueError(f"指定网格 {rows} x {cols} 无法容纳 {piece_count} 个碎片")
    return rows, cols
