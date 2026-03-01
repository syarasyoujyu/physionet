from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from stage2.data import (
    Record,
    _load_json,
    _load_rgb,
    _pil_mask_to_hw_float01,
    _pil_to_chw_float01,
    _random_crop_box,
    build_waveform_mask_from_json,
    discover_records,
)
from stage2.model import UNetRes


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: str
    box: tuple[int, int, int, int]
    x: torch.Tensor  # (3,H,W) float01
    y: torch.Tensor  # (1,H,W) float01


def _setup_logger(level: str) -> logging.Logger:
    lvl = getattr(logging, level.upper(), None)
    if not isinstance(lvl, int):
        raise ValueError(f"invalid log level: {level!r}")
    logger = logging.getLogger("stage2.inference")
    logger.setLevel(lvl)
    logger.propagate = False
    logger.handlers.clear()
    h = logging.StreamHandler()
    h.setLevel(lvl)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
    return logger


def _load_checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"unsupported checkpoint format: {type(obj)}")


def _mask_path(record: Record, *, mask_suffix: str, cache_dir: Path | None) -> Path:
    if cache_dir is None:
        return record.image_path.with_name(f"{record.stem}{mask_suffix}")
    return cache_dir / f"{record.stem}{mask_suffix}"


def _load_or_make_mask(
    record: Record,
    *,
    meta: dict,
    w: int,
    h: int,
    line_width: int,
    mask_suffix: str,
    cache_dir: Path | None,
) -> Image.Image:
    mp = _mask_path(record, mask_suffix=mask_suffix, cache_dir=cache_dir)
    if mp.exists():
        return Image.open(mp).convert("L")

    mask = build_waveform_mask_from_json(meta, width=w, height=h, line_width=int(line_width))
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        mask.save(mp)
    return mask


def _center_crop_box(w: int, h: int, size: int) -> tuple[int, int, int, int]:
    if w < size or h < size:
        raise ValueError(f"image smaller than patch: {(w, h)} < {size}")
    left = (w - size) // 2
    top = (h - size) // 2
    return left, top, left + size, top + size


def _make_samples(
    records: list[Record],
    *,
    num_samples: int,
    seed: int,
    patch_size: int,
    crop: str,
    cache_dir: Path | None,
    mask_suffix: str,
    line_width: int,
) -> list[Sample]:
    rng = random.Random(seed)
    out: list[Sample] = []
    if not records:
        return out

    for i in range(num_samples):
        r = records[i % len(records)]
        rgb = _load_rgb(r.image_path)
        meta = _load_json(r.json_path)
        w, h = rgb.size
        mask = _load_or_make_mask(
            r,
            meta=meta,
            w=w,
            h=h,
            line_width=int(line_width),
            mask_suffix=mask_suffix,
            cache_dir=cache_dir,
        )
        if crop == "center":
            box = _center_crop_box(w, h, patch_size)
        else:
            box = _random_crop_box(w, h, patch_size, rng)

        rgb_patch = rgb.crop(box)
        mask_patch = mask.crop(box)
        x = _pil_to_chw_float01(rgb_patch)
        y = _pil_mask_to_hw_float01(mask_patch)
        out.append(Sample(stem=r.stem, image_path=str(r.image_path), box=box, x=x, y=y))
    return out


def _save_grid_figure(
    out_path: Path | None,
    *,
    samples: list[Sample],
    probs: torch.Tensor,  # (N,1,H,W)
    threshold: float,
    dpi: int,
    title: str,
    show: bool,
) -> None:
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("visualization requires matplotlib. Install it (e.g. `uv add matplotlib`).") from e

    n = len(samples)
    ncols = 4
    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 3.2, max(1, n) * 3.2), squeeze=False)
    pred_bin = (probs >= float(threshold)).float()

    for i, s in enumerate(samples):
        rgb = s.x.clamp(0, 1).permute(1, 2, 0).numpy()
        gt = s.y[0].clamp(0, 1).numpy()
        pr_prob = probs[i, 0].clamp(0, 1).cpu().numpy()
        pr = pred_bin[i, 0].clamp(0, 1).cpu().numpy()

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"input\n{s.stem}")
        axes[i, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("GT")
        axes[i, 2].imshow(pr_prob, cmap="magma", vmin=0, vmax=1)
        axes[i, 2].set_title("pred(prob)")
        axes[i, 3].imshow(pr, cmap="gray", vmin=0, vmax=1)
        axes[i, 3].set_title(f"pred@{threshold:g}")
        for j in range(ncols):
            axes[i, j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=int(dpi))

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--checkpoint", type=str, default="runs/stage2/checkpoints/best.pt")
    p.add_argument("--out-dir", type=str, default="runs/stage2/infer")
    p.add_argument("--out-name", type=str, default="pred_vs_gt.png")
    p.add_argument("--max-data-num", type=int, default=None)
    p.add_argument("--stems", type=str, nargs="*", default=None, help="only run on specified stems (filenames without extension)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crop", type=str, choices=["random", "center"], default="center")
    p.add_argument("--cache-dir", type=str, default="runs/stage2/masks_cache")
    p.add_argument("--mask-suffix", type=str, default="_wave_mask.png")
    p.add_argument("--line-width", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--show", action="store_true", help="show figure window (may not work on headless env)")
    p.add_argument("--no-save", action="store_true", help="do not save figure (use with --show)")
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args()

    logger = _setup_logger(args.log_level)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    records = discover_records(Path(args.data_root), max_data_num=args.max_data_num, seed=args.seed)
    if args.stems:
        wanted = set(args.stems)
        records = [r for r in records if r.stem in wanted]
    if not records:
        raise ValueError("no records found (need matching .png and .json).")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    samples = _make_samples(
        records,
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        patch_size=int(args.patch_size),
        crop=str(args.crop),
        cache_dir=cache_dir,
        mask_suffix=str(args.mask_suffix),
        line_width=int(args.line_width),
    )

    model = UNetRes(in_channels=3, base_channels=int(args.base_channels), out_channels=1)
    state = _load_checkpoint_state_dict(ckpt_path)
    model.load_state_dict(state, strict=True)
    model.to(args.device)
    model.eval()

    xs = torch.stack([s.x for s in samples], dim=0)
    probs = torch.zeros((xs.size(0), 1, xs.size(2), xs.size(3)), dtype=torch.float32)

    with torch.no_grad():
        for i0 in range(0, xs.size(0), int(args.batch_size)):
            x = xs[i0 : i0 + int(args.batch_size)].to(args.device, non_blocking=True)
            logits = model(x).detach().cpu()
            probs[i0 : i0 + logits.size(0)] = torch.sigmoid(logits)

    out_dir = Path(args.out_dir)
    out_path = None if args.no_save else (out_dir / str(args.out_name))
    _save_grid_figure(
        out_path,
        samples=samples,
        probs=probs,
        threshold=float(args.threshold),
        dpi=int(args.dpi),
        title="stage2 inference",
        show=bool(args.show),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples.json").write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "num_samples": len(samples),
                "patch_size": int(args.patch_size),
                "crop": str(args.crop),
                "threshold": float(args.threshold),
                "samples": [
                    {
                        "stem": s.stem,
                        "image_path": s.image_path,
                        "box": list(s.box),
                    }
                    for s in samples
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("saved: %s", str(out_path) if out_path is not None else "(no-save)")
    logger.info("saved: %s", str(out_dir / "samples.json"))


if __name__ == "__main__":
    main()
