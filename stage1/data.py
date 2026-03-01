from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


@dataclasses.dataclass(frozen=True)
class Record:
    stem: str
    image_path: Path
    json_path: Path


def discover_records(
    data_root: Path,
    *,
    max_data_num: int | None = None,
    seed: int | None = None,
) -> list[Record]:
    if max_data_num is not None and max_data_num < 1:
        raise ValueError("max_data_num must be >= 1 (or None).")
    data_root = data_root.resolve()
    records: list[Record] = []
    for img_path in tqdm(sorted(data_root.glob("**/*.png"))):
        if img_path.name.endswith(("_mask.png", "_grid_mask.png", "_wave_mask.png")):
            continue
        stem = img_path.with_suffix("").name
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            continue
        records.append(Record(stem=stem, image_path=img_path, json_path=json_path))

    if max_data_num is None or max_data_num >= len(records):
        return records
    rng = random.Random(seed)
    return rng.sample(records, k=max_data_num)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")


def _pil_to_chw_float01(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _pil_mask_to_hw_float01(mask: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, round(v))))


def _random_crop_box(w: int, h: int, size: int, rng: random.Random) -> tuple[int, int, int, int]:
    if w < size or h < size:
        raise ValueError(f"image smaller than patch: {(w, h)} < {size}")
    left = rng.randint(0, w - size)
    top = rng.randint(0, h - size)
    return left, top, left + size, top + size


def _grid_positions(length: int, spacing: float, offset: float) -> list[int]:
    if length <= 0 or not np.isfinite(spacing) or spacing <= 0:
        return []
    start_k = int(np.ceil((0.0 - offset) / spacing))
    end_k = int(np.floor(((length - 1) - offset) / spacing))
    coords: list[int] = []
    prev = None
    for k in range(start_k, end_k + 1):
        pos = offset + spacing * k
        idx = int(round(pos))
        if 0 <= idx < length and idx != prev:
            coords.append(idx)
            prev = idx
    return coords


def _rotate_xy(x: float, y: float, cx: float, cy: float, angle_deg: float) -> tuple[int, int]:
    theta = np.deg2rad(angle_deg)
    dx = x - cx
    dy = y - cy
    xr = dx * np.cos(theta) - dy * np.sin(theta) + cx
    yr = dx * np.sin(theta) + dy * np.cos(theta) + cy
    return int(round(xr)), int(round(yr))


def build_grid_intersection_mask_from_json(meta: dict, *, width: int, height: int) -> Image.Image:
    """
    json内の x_grid / y_grid と画像サイズから、主要グリッド交点の2値マスクを生成する。
    generator 側では元キャンバス左上を原点に格子が置かれるため、pad/crop がある場合は
    json上の width/height と実画像サイズとの差分をオフセットとして扱う。
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if not bool(meta.get("gridlines", True)):
        return Image.fromarray(mask, mode="L")

    try:
        x_grid = float(meta["x_grid"])
        y_grid = float(meta["y_grid"])
    except Exception as e:
        raise ValueError("json label source requires numeric x_grid and y_grid.") from e

    if not (np.isfinite(x_grid) and np.isfinite(y_grid) and x_grid > 0 and y_grid > 0):
        raise ValueError(f"invalid grid size in json: x_grid={x_grid!r} y_grid={y_grid!r}")

    meta_width = meta.get("width")
    meta_height = meta.get("height")
    try:
        offset_x = (float(width) - float(meta_width)) / 2.0 if meta_width is not None else 0.0
        offset_y = (float(height) - float(meta_height)) / 2.0 if meta_height is not None else 0.0
    except Exception:
        offset_x = 0.0
        offset_y = 0.0

    xs = _grid_positions(width, x_grid, offset_x)
    ys = _grid_positions(height, y_grid, offset_y)
    angle_value = meta["rotate_applied"] if "rotate_applied" in meta else meta.get("rotate", 0.0)
    angle = float(angle_value if angle_value is not None else 0.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    for y in ys:
        for x in xs:
            xi, yi = _rotate_xy(x, y, cx, cy, angle) if angle else (x, y)
            if 0 <= xi < width and 0 <= yi < height:
                mask[yi, xi] = 255
    return Image.fromarray(mask, mode="L")


def _auto_grid_intersection_mask(
    rgb: Image.Image,
    meta: dict,
    *,
    corner_thresh: float = 0.01,
) -> Image.Image:
    """
    OpenCVのHarris cornerで「グリッド交点っぽい点」マスクを自動生成。
    本番の正解ラベルが無い場合の代替。
    """
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("stage1 auto-label requires opencv-python (cv2).") from e

    arr = np.asarray(rgb, dtype=np.uint8)
    # グリッドの色（jsonに入っている想定: [R,G,B]）
    minor = np.array(meta.get("grid_line_color_minor", [254, 223, 219]), dtype=np.float32)
    major = np.array(meta.get("grid_line_color_major", [255, 0, 0]), dtype=np.float32)

    # 近い色の画素をグリッド候補として抽出（ノイズ/しわにより完全一致しないので距離で）
    arr_f = arr.astype(np.float32)
    d_minor = np.linalg.norm(arr_f - minor[None, None, :], axis=-1)
    d_major = np.linalg.norm(arr_f - major[None, None, :], axis=-1)
    grid = np.minimum(d_minor, d_major)

    grid_mask = (grid < 60.0).astype(np.uint8) * 255
    grid_mask = cv2.medianBlur(grid_mask, 3)
    grid_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # Harris corners
    src = (grid_mask.astype(np.float32) / 255.0)
    harris = cv2.cornerHarris(src, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    thr = float(harris.max()) * float(corner_thresh)
    corner_bin = (harris > thr).astype(np.uint8)

    # connected components -> centroid per blob
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(corner_bin, connectivity=8)
    out = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
    for i in range(1, num):
        x, y = centroids[i]
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < out.shape[0] and 0 <= xi < out.shape[1]:
            out[yi, xi] = 255
    return Image.fromarray(out, mode="L")


class GridIntersectionPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[Record],
        *,
        patch_size: int = 256,
        samples_per_epoch: int = 10_000,
        seed: int = 0,
        label_source: Literal["auto", "file", "json"] = "json",
        mask_suffix: str = "_grid_mask.png",
        label_cache_dir: Path | None = None,
    ) -> None:
        self.records = records
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.rng = random.Random(seed)
        self.label_source = label_source
        self.mask_suffix = mask_suffix
        self.label_cache_dir = label_cache_dir

        if self.label_cache_dir is not None:
            self.label_cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.records:
            raise ValueError("no records found (need matching .png and .json).")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _label_path_for_record(self, r: Record) -> Path:
        if self.label_cache_dir is not None:
            cache_root = self.label_cache_dir / self.label_source
            return cache_root / f"{r.stem}{self.mask_suffix}"
        return r.image_path.with_name(f"{r.stem}{self.mask_suffix}")

    def _load_or_make_label(self, r: Record, rgb: Image.Image, meta: dict) -> Image.Image:
        label_path = self._label_path_for_record(r)
        if self.label_source == "file":
            if not label_path.exists():
                raise FileNotFoundError(f"label not found: {label_path}")
            return Image.open(label_path).convert("L")

        if self.label_source == "json":
            if label_path.exists():
                return Image.open(label_path).convert("L")
            w, h = rgb.size
            mask = build_grid_intersection_mask_from_json(meta, width=w, height=h)
            if self.label_cache_dir is not None:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                mask.save(label_path)
            return mask

        if label_path.exists():
            return Image.open(label_path).convert("L")

        mask = _auto_grid_intersection_mask(rgb, meta)
        if self.label_cache_dir is not None:
            label_path.parent.mkdir(parents=True, exist_ok=True)
            mask.save(label_path)
        return mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.records[self.rng.randrange(0, len(self.records))]
        rgb = _load_rgb(r.image_path)
        meta = _load_json(r.json_path)
        label = self._load_or_make_label(r, rgb, meta)

        w, h = rgb.size
        left, top, right, bottom = _random_crop_box(w, h, self.patch_size, self.rng)
        rgb_patch = rgb.crop((left, top, right, bottom))
        label_patch = label.crop((left, top, right, bottom))

        x = _pil_to_chw_float01(rgb_patch)
        y = _pil_mask_to_hw_float01(label_patch)
        return x, y


def precompute_grid_intersection_labels(
    records: list[Record],
    *,
    label_source: Literal["auto", "file", "json"] = "json",
    mask_suffix: str = "_grid_mask.png",
    label_cache_dir: Path | None = None,
    progress: bool = True,
) -> dict[str, int]:
    """
    label_source=auto の場合に、全recordの交点マスクを事前生成して保存する。
    進捗(%)を見たい時や、学習中の遅延を減らしたい時に使う。
    """
    if label_source == "file":
        return {"seen": len(records), "created": 0, "skipped": len(records)}

    if label_cache_dir is not None:
        label_cache_dir = label_cache_dir / label_source
        label_cache_dir.mkdir(parents=True, exist_ok=True)

    seen = 0
    created = 0
    skipped = 0
    it = records
    if progress:
        it = tqdm(records, desc="[stage1] precompute labels", unit="rec")
    for r in it:
        seen += 1
        if label_cache_dir is not None:
            out_path = label_cache_dir / f"{r.stem}{mask_suffix}"
        else:
            out_path = r.image_path.with_name(f"{r.stem}{mask_suffix}")

        if out_path.exists():
            skipped += 1
            continue

        rgb = _load_rgb(r.image_path)
        meta = _load_json(r.json_path)
        w, h = rgb.size
        if label_source == "json":
            mask = build_grid_intersection_mask_from_json(meta, width=w, height=h)
        else:
            mask = _auto_grid_intersection_mask(rgb, meta)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(out_path)
        created += 1

    return {"seen": seen, "created": created, "skipped": skipped}
