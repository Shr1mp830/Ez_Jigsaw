from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median

from PIL import Image, ImageOps


@dataclass
class SplitResult:
    folder: Path
    rows: int
    cols: int
    count: int


def split_scrambled_image(
    image_path: str | Path,
    output_folder: str | Path,
    rows: int = 0,
    cols: int = 0,
    normalize_piece_size: bool = False,
) -> SplitResult:
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGBA")
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    row_cuts, col_cuts = detect_grid_cuts(image, rows=rows, cols=cols)
    target_size = normalized_piece_size(row_cuts, col_cuts) if normalize_piece_size else None
    for old_file in output_path.glob("piece_*.png"):
        old_file.unlink()

    count = 0
    for row in range(len(row_cuts) - 1):
        for col in range(len(col_cuts) - 1):
            box = (col_cuts[col], row_cuts[row], col_cuts[col + 1], row_cuts[row + 1])
            piece = image.crop(box)
            if target_size is not None and piece.size != target_size:
                piece = piece.resize(target_size, Image.Resampling.LANCZOS)
            piece.save(output_path / f"piece_{count:03d}_{row}_{col}.png")
            count += 1

    return SplitResult(output_path, len(row_cuts) - 1, len(col_cuts) - 1, count)


def normalized_piece_size(row_cuts: list[int], col_cuts: list[int]) -> tuple[int, int]:
    widths = [max(1, col_cuts[index + 1] - col_cuts[index]) for index in range(len(col_cuts) - 1)]
    heights = [max(1, row_cuts[index + 1] - row_cuts[index]) for index in range(len(row_cuts) - 1)]
    if not widths or not heights:
        return (1, 1)
    return (max(1, round(median(widths))), max(1, round(median(heights))))


def detect_grid_cuts(image: Image.Image, rows: int = 0, cols: int = 0) -> tuple[list[int], list[int]]:
    width, height = image.size
    if rows < 0 or cols < 0:
        raise ValueError("行数和列数不能为负数")

    row_count = rows if rows > 0 else None
    col_count = cols if cols > 0 else None
    x_profile = seam_profile(image, axis="x")
    y_profile = seam_profile(image, axis="y")

    if col_count is None:
        col_count = infer_parts_from_profile(x_profile, width)
    if row_count is None:
        row_count = infer_parts_from_profile(y_profile, height)

    col_cuts = choose_cuts(x_profile, width, col_count)
    row_cuts = choose_cuts(y_profile, height, row_count)
    row_cuts, col_cuts = refine_rectangular_grid_cuts(
        row_cuts,
        col_cuts,
        x_profile,
        y_profile,
        width,
        height,
        row_count,
        col_count,
    )
    return row_cuts, col_cuts


def refine_rectangular_grid_cuts(
    row_cuts: list[int],
    col_cuts: list[int],
    x_profile: list[float],
    y_profile: list[float],
    width: int,
    height: int,
    row_count: int | None,
    col_count: int | None,
) -> tuple[list[int], list[int]]:
    if row_count is not None and col_count is None:
        inferred_cols = infer_parts_from_profile(x_profile, width)
        expected_width = width / inferred_cols if inferred_cols else 0
        if inferred_cols and accepts_inferred_parts(x_profile, width, inferred_cols, expected_width):
            col_cuts = choose_cuts(x_profile, width, inferred_cols)

    if col_count is not None and row_count is None:
        inferred_rows = infer_parts_from_profile(y_profile, height)
        expected_height = height / inferred_rows if inferred_rows else 0
        if inferred_rows and accepts_inferred_parts(y_profile, height, inferred_rows, expected_height):
            row_cuts = choose_cuts(y_profile, height, inferred_rows)

    return row_cuts, col_cuts


def infer_parts_from_profile(profile: list[float], length: int) -> int | None:
    if length <= 16 or not profile:
        return None

    threshold = auto_peak_threshold(profile)
    peak_positions = dominant_peak_positions(profile, length, threshold)
    if len(peak_positions) >= 2:
        gaps = [
            peak_positions[index + 1] - peak_positions[index]
            for index in range(len(peak_positions) - 1)
            if peak_positions[index + 1] > peak_positions[index]
        ]
        if gaps:
            cell = median(gaps)
            inferred_parts = round(length / cell) if cell else 0
            expected_cell = length / inferred_parts if inferred_parts else 0
            if accepts_inferred_parts(profile, length, inferred_parts, expected_cell):
                return inferred_parts

    candidates: list[tuple[float, int, float, float]] = []
    max_parts = min(80, max(2, length // 8))
    for parts in range(2, max_parts + 1):
        cell = length / parts
        if cell < 8:
            continue
        strengths = expected_cut_strengths(profile, length, parts)
        if not strengths:
            continue
        sampled = sample_edge_strengths(strengths)
        average = sum(sampled) / len(sampled)
        middle = sorted(sampled)[len(sampled) // 2]
        full_average = sum(strengths) / len(strengths)
        confidence = (middle / max(1.0, threshold)) * 0.55 + (average / max(1.0, threshold)) * 0.3
        confidence += min(1.5, full_average / max(1.0, threshold)) * 0.15
        confidence += min(0.22, parts / max(1, length) * 4.0)
        if middle >= threshold * 0.75 or average >= threshold:
            candidates.append((confidence, parts, average, middle))

    if not candidates:
        return None
    strong_candidates = [
        item for item in candidates if item[2] >= threshold * 2.0 and item[3] >= threshold * 1.8
    ]
    if strong_candidates:
        return max(strong_candidates, key=lambda item: item[1])[1]

    best_confidence, best_parts, _average, _middle = max(candidates, key=lambda item: item[0])
    if best_confidence < 0.82:
        return None
    return best_parts


def dominant_peak_positions(profile: list[float], length: int, threshold: float) -> list[int]:
    min_gap = max(8, length // 80)
    peaks: list[tuple[float, int]] = []
    for index, value in enumerate(profile):
        if value < threshold:
            continue
        left = profile[index - 1] if index > 0 else -1.0
        right = profile[index + 1] if index + 1 < len(profile) else -1.0
        if value >= left and value >= right:
            peaks.append((value, index + 1))

    selected: list[int] = []
    for _value, position in sorted(peaks, reverse=True):
        if position < min_gap or length - position < min_gap:
            continue
        if all(abs(position - chosen) >= min_gap for chosen in selected):
            selected.append(position)
        if len(selected) >= 80:
            break
    return sorted(selected)


def sample_edge_strengths(strengths: list[float], sample_count: int = 5) -> list[float]:
    if len(strengths) <= sample_count:
        return strengths
    indexes = {
        0,
        len(strengths) // 4,
        len(strengths) // 2,
        len(strengths) * 3 // 4,
        len(strengths) - 1,
    }
    return [strengths[index] for index in sorted(indexes)]


def median_cut_size(cuts: list[int]) -> float:
    if len(cuts) < 2:
        return 0.0
    sizes = [max(1, cuts[index + 1] - cuts[index]) for index in range(len(cuts) - 1)]
    return float(median(sizes))


def uniform_cuts(length: int, parts: int) -> list[int]:
    if parts <= 1 or length <= 1:
        return [0, length]
    if parts >= length:
        return list(range(0, length + 1))

    cuts = [0]
    for part in range(1, parts):
        cut = round(part * length / parts)
        cut = max(cuts[-1] + 1, min(length - (parts - part), cut))
        cuts.append(cut)
    cuts.append(length)
    return cuts


def regular_grid_cuts(profile: list[float], length: int, parts: int) -> list[int] | None:
    if parts > length:
        return None
    if accepts_inferred_parts(profile, length, parts, length / max(1, parts)):
        return uniform_cuts(length, parts)
    return None


def accepts_inferred_parts(profile: list[float], length: int, parts: int, expected_cell: float) -> bool:
    if parts < 2 or parts > 80 or expected_cell <= 0:
        return False
    cell = length / parts
    if abs(cell - expected_cell) / expected_cell > 0.18:
        return False
    threshold = auto_peak_threshold(profile)
    strengths = expected_cut_strengths(profile, length, parts)
    if not strengths:
        return False
    average = sum(strengths) / len(strengths)
    middle = sorted(strengths)[len(strengths) // 2]
    return average >= threshold * 1.35 or middle >= threshold * 0.9


def expected_cut_strengths(profile: list[float], length: int, parts: int) -> list[float]:
    strengths: list[float] = []
    for part in range(1, parts):
        target = round(part * length / parts)
        window = max(2, round(length / parts * 0.08))
        left = max(1, target - window)
        right = min(length - 1, target + window)
        strengths.append(max(profile[position - 1] for position in range(left, right + 1)))
    return strengths


def seam_profile(image: Image.Image, axis: str) -> list[float]:
    rgb = Image.new("RGBA", image.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(rgb, image.convert("RGBA"))
    rgb = rgb.convert("RGB")
    pixels = rgb.load()
    width, height = rgb.size

    if axis == "x":
        step = max(1, height // 512)
        profile: list[float] = []
        for x in range(1, width):
            total = 0.0
            count = 0
            for y in range(0, height, step):
                total += abs(pixels[x, y][0] - pixels[x - 1, y][0])
                total += abs(pixels[x, y][1] - pixels[x - 1, y][1])
                total += abs(pixels[x, y][2] - pixels[x - 1, y][2])
                count += 1
            profile.append(total / max(1, count))
        return smooth_profile(profile)

    if axis == "y":
        step = max(1, width // 512)
        profile = []
        for y in range(1, height):
            total = 0.0
            count = 0
            for x in range(0, width, step):
                total += abs(pixels[x, y][0] - pixels[x, y - 1][0])
                total += abs(pixels[x, y][1] - pixels[x, y - 1][1])
                total += abs(pixels[x, y][2] - pixels[x, y - 1][2])
                count += 1
            profile.append(total / max(1, count))
        return smooth_profile(profile)

    raise ValueError(f"未知轴：{axis}")


def smooth_profile(values: list[float], radius: int = 2) -> list[float]:
    if not values:
        return values
    smoothed: list[float] = []
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        smoothed.append(sum(values[left:right]) / (right - left))
    return smoothed


def choose_cuts(profile: list[float], length: int, expected_parts: int | None) -> list[int]:
    if length <= 1:
        return [0, length]

    if expected_parts is not None and expected_parts > 1:
        regular_cuts = regular_grid_cuts(profile, length, expected_parts)
        if regular_cuts is not None:
            return regular_cuts

        cuts = [0]
        for part in range(1, expected_parts):
            target = round(part * length / expected_parts)
            window = max(4, round(length / expected_parts * 0.28))
            left = max(1, target - window)
            right = min(length - 1, target + window)
            best = max(range(left, right + 1), key=lambda position: profile[position - 1])
            cuts.append(best)
        cuts.append(length)
        return sorted(set(cuts))

    threshold = auto_peak_threshold(profile)
    min_gap = max(10, length // 60)
    peaks: list[tuple[float, int]] = []
    for index, value in enumerate(profile):
        if value < threshold:
            continue
        left = profile[index - 1] if index > 0 else -1.0
        right = profile[index + 1] if index + 1 < len(profile) else -1.0
        if value >= left and value >= right:
            peaks.append((value, index + 1))

    selected: list[int] = []
    for _score, position in sorted(peaks, reverse=True):
        if position < min_gap or length - position < min_gap:
            continue
        if all(abs(position - chosen) >= min_gap for chosen in selected):
            selected.append(position)
        if len(selected) >= 40:
            break

    return [0, *sorted(selected), length]


def auto_peak_threshold(profile: list[float]) -> float:
    if not profile:
        return 0.0
    center = median(profile)
    deviations = [abs(value - center) for value in profile]
    spread = median(deviations) or 1.0
    return center + spread * 5.0


def lsb_bitplane(image: Image.Image, channel: str, bit: int = 0) -> Image.Image:
    if bit < 0 or bit > 7:
        raise ValueError("LSB bit 必须在 0 到 7 之间")

    channel = channel.upper()
    rgba = image.convert("RGBA")
    pixels = list(rgba.getdata())

    if channel == "RGB":
        output = [
            (
                255 if ((pixel[0] >> bit) & 1) else 0,
                255 if ((pixel[1] >> bit) & 1) else 0,
                255 if ((pixel[2] >> bit) & 1) else 0,
            )
            for pixel in pixels
        ]
        result = Image.new("RGB", rgba.size)
        result.putdata(output)
        return result

    channel_map = {"R": 0, "G": 1, "B": 2, "A": 3}
    if channel not in channel_map:
        raise ValueError("通道必须是 R/G/B/A/RGB")

    index = channel_map[channel]
    output = [255 if ((pixel[index] >> bit) & 1) else 0 for pixel in pixels]
    result = Image.new("L", rgba.size)
    result.putdata(output)
    return result.convert("RGB")
