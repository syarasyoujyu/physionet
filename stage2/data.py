from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
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
    rng = random.Random(seed) if max_data_num is not None else None
    seen = 0
    for img_path in sorted(data_root.glob("**/*.png")):
        if img_path.name.endswith(("_mask.png", "_grid_mask.png", "_wave_mask.png")):
            continue
        stem = img_path.with_suffix("").name
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            continue
        r = Record(stem=stem, image_path=img_path, json_path=json_path)
        if max_data_num is None:
            records.append(r)
            continue

        # reservoir sampling (uniform without replacement)
        seen += 1
        if len(records) < max_data_num:
            records.append(r)
            continue
        assert rng is not None
        j = rng.randrange(0, seen)
        if j < max_data_num:
            records[j] = r
    return records


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


def build_waveform_mask_from_json(meta: dict, *, width: int, height: int, line_width: int = 2) -> Image.Image:
    """
    json内の leads[*].plotted_pixels を使って、全リードの波形をunionした2値マスクを生成。
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for lead in meta.get("leads", []):
        pts = lead.get("plotted_pixels", [])
        if not pts:
            continue
        prev = None
        for xy in pts:
            if not (isinstance(xy, list) and len(xy) == 2):
                continue
            try:
                x_f = float(xy[0])
                y_f = float(xy[1])
            except Exception:
                continue
            if not (np.isfinite(x_f) and np.isfinite(y_f)):
                continue
            x = _clamp_int(x_f, 0, width - 1)
            y = _clamp_int(y_f, 0, height - 1)
            cur = (x, y)
            if prev is not None:
                draw.line([prev, cur], fill=255, width=line_width)
            prev = cur
    return mask


def _random_crop_box(w: int, h: int, size: int, rng: random.Random) -> tuple[int, int, int, int]:
    if w < size or h < size:
        raise ValueError(f"image smaller than patch: {(w, h)} < {size}")
    left = rng.randint(0, w - size)
    top = rng.randint(0, h - size)
    return left, top, left + size, top + size


def _crop_around_point(
    w: int, h: int, size: int, cx: int, cy: int, rng: random.Random
) -> tuple[int, int, int, int]:
    # top-left is sampled so that (cx, cy) is guaranteed to be inside the crop
    left_lo = max(0, cx - size + 1)
    left_hi = min(w - size, cx)
    top_lo = max(0, cy - size + 1)
    top_hi = min(h - size, cy)
    if left_lo > left_hi or top_lo > top_hi:
        return _random_crop_box(w, h, size, rng)
    left = rng.randint(left_lo, left_hi)
    top = rng.randint(top_lo, top_hi)
    return left, top, left + size, top + size


class WaveformPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[Record],
        *,
        patch_size: int = 256,
        samples_per_epoch: int = 10_000,
        seed: int = 0,
        cache_dir: Path | None = None,
        mask_suffix: str = "_wave_mask.png",
        line_width: int = 2,
        positive_sample_prob: float = 0.7,
    ) -> None:
        self.records = records
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.rng = random.Random(seed)
        self.cache_dir = cache_dir
        self.mask_suffix = mask_suffix
        self.line_width = line_width
        self.positive_sample_prob = float(positive_sample_prob)

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.records:
            raise ValueError("no records found (need matching .png and .json).")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _mask_path(self, r: Record) -> Path:
        if self.cache_dir is None:
            return r.image_path.with_name(f"{r.stem}{self.mask_suffix}")
        return self.cache_dir / f"{r.stem}{self.mask_suffix}"

    def _load_or_make_mask(self, r: Record, meta: dict, *, w: int, h: int) -> Image.Image:
        mp = self._mask_path(r)
        if mp.exists():
            return Image.open(mp).convert("L")
        mask = build_waveform_mask_from_json(meta, width=w, height=h, line_width=self.line_width)
        if self.cache_dir is not None:
            mask.save(mp)
        return mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.records[self.rng.randrange(0, len(self.records))]
        rgb = _load_rgb(r.image_path)
        meta = _load_json(r.json_path)
        w, h = rgb.size
        mask = self._load_or_make_mask(r, meta, w=w, h=h)

        # positive patch sampling: 波形座標から中心点を選ぶ
        if self.rng.random() < self.positive_sample_prob:
            leads = meta.get("leads", [])
            pts = []
            for lead in leads:
                pp = lead.get("plotted_pixels", [])
                if pp:
                    pts.append(pp[self.rng.randrange(0, len(pp))])
            if pts:
                chosen = pts[self.rng.randrange(0, len(pts))]
                if isinstance(chosen, list) and len(chosen) == 2:
                    try:
                        x0 = float(chosen[0])
                        y0 = float(chosen[1])
                    except Exception:
                        x0, y0 = None, None
                else:
                    x0, y0 = None, None

                if x0 is not None and y0 is not None and np.isfinite(x0) and np.isfinite(y0):
                    cx = _clamp_int(x0, 0, w - 1)
                    cy = _clamp_int(y0, 0, h - 1)
                    box = _crop_around_point(w, h, self.patch_size, cx, cy, self.rng)
                else:
                    box = _random_crop_box(w, h, self.patch_size, self.rng)
            else:
                box = _random_crop_box(w, h, self.patch_size, self.rng)
        else:
            box = _random_crop_box(w, h, self.patch_size, self.rng)

        rgb_patch = rgb.crop(box)
        mask_patch = mask.crop(box)
        x = _pil_to_chw_float01(rgb_patch)
        y = _pil_mask_to_hw_float01(mask_patch)
        return x, y


def precompute_waveform_masks(
    records: list[Record],
    *,
    cache_dir: Path | None = None,
    mask_suffix: str = "_wave_mask.png",
    line_width: int = 2,
    progress: bool = True,
) -> dict[str, int]:
    """
    jsonから波形マスクを事前生成して保存する。
    進捗(%)を見たい時や、学習中の遅延を減らしたい時に使う。
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    seen = 0
    created = 0
    skipped = 0
    it = records
    if progress:
        it = tqdm(records, desc="[stage2] precompute masks", unit="rec")

    for r in it:
        seen += 1
        if cache_dir is not None:
            out_path = cache_dir / f"{r.stem}{mask_suffix}"
        else:
            out_path = r.image_path.with_name(f"{r.stem}{mask_suffix}")

        if out_path.exists():
            skipped += 1
            continue

        rgb = _load_rgb(r.image_path)
        meta = _load_json(r.json_path)
        w, h = rgb.size
        mask = build_waveform_mask_from_json(meta, width=w, height=h, line_width=line_width)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(out_path)
        created += 1

    return {"seen": seen, "created": created, "skipped": skipped}
