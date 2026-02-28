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
    for img_path in sorted(data_root.glob("**/*.png")):
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


def _random_crop_box(w: int, h: int, size: int, rng: random.Random) -> tuple[int, int, int, int]:
    if w < size or h < size:
        raise ValueError(f"image smaller than patch: {(w, h)} < {size}")
    left = rng.randint(0, w - size)
    top = rng.randint(0, h - size)
    return left, top, left + size, top + size


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
        label_source: Literal["auto", "file"] = "auto",
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
            return self.label_cache_dir / f"{r.stem}{self.mask_suffix}"
        return r.image_path.with_name(f"{r.stem}{self.mask_suffix}")

    def _load_or_make_label(self, r: Record, rgb: Image.Image, meta: dict) -> Image.Image:
        label_path = self._label_path_for_record(r)
        if self.label_source == "file":
            if not label_path.exists():
                raise FileNotFoundError(f"label not found: {label_path}")
            return Image.open(label_path).convert("L")

        if label_path.exists():
            return Image.open(label_path).convert("L")

        mask = _auto_grid_intersection_mask(rgb, meta)
        if self.label_cache_dir is not None:
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
    label_source: Literal["auto", "file"] = "auto",
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
        mask = _auto_grid_intersection_mask(rgb, meta)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(out_path)
        created += 1

    return {"seen": seen, "created": created, "skipped": skipped}
