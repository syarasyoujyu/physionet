from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from stage1.data import (
    GridIntersectionPatchDataset,
    Record,
    _load_json,
    _load_rgb,
    _pil_mask_to_hw_float01,
    _pil_to_chw_float01,
    _random_crop_box,
    discover_records,
)
from stage1.model import UNetRes


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
    logger = logging.getLogger("stage1.inference")
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


def _center_crop_box(w: int, h: int, size: int) -> tuple[int, int, int, int]:
    if w < size or h < size:
        raise ValueError(f"image smaller than patch: {(w, h)} < {size}")
    left = (w - size) // 2
    top = (h - size) // 2
    return left, top, left + size, top + size


def _make_samples(
    records: list[Record],
    *,
    labeler: GridIntersectionPatchDataset,
    num_samples: int,
    seed: int,
    patch_size: int,
    crop: str,
) -> list[Sample]:
    rng = random.Random(seed)
    out: list[Sample] = []
    if not records:
        return out

    for i in range(num_samples):
        r = records[i % len(records)]
        rgb = _load_rgb(r.image_path)
        meta = _load_json(r.json_path)
        # stage1.data の GridIntersectionPatchDataset._load_or_make_label を利用して交点マスクを生成/読込
        label = labeler._load_or_make_label(r, rgb, meta)
        w, h = rgb.size
        if crop == "center":
            box = _center_crop_box(w, h, patch_size)
        else:
            box = _random_crop_box(w, h, patch_size, rng)

        rgb_patch = rgb.crop(box)
        label_patch = label.crop(box)
        x = _pil_to_chw_float01(rgb_patch)
        y = _pil_mask_to_hw_float01(label_patch)
        out.append(Sample(stem=r.stem, image_path=str(r.image_path), box=box, x=x, y=y))
    return out


def _dilate_points(mask01: torch.Tensor, *, radius: int) -> torch.Tensor:
    """
    交点マスクは点が小さく見えにくいので、可視化用に膨張する（学習/推論自体には使わない）。
    mask01: (N,1,H,W) or (1,H,W) values in [0,1]
    """
    if radius <= 0:
        return mask01
    if mask01.ndim == 3:
        mask01 = mask01.unsqueeze(0)
    k = int(2 * radius + 1)
    return F.max_pool2d(mask01, kernel_size=k, stride=1, padding=radius)


def _save_grid_figure(
    out_path: Path | None,
    *,
    samples: list[Sample],
    probs: torch.Tensor,  # (N,1,H,W)
    threshold: float,
    dot_radius: int,
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
    gt = torch.stack([s.y for s in samples], dim=0)
    gt_vis = _dilate_points(gt, radius=int(dot_radius))
    pred_vis = _dilate_points(pred_bin, radius=int(dot_radius))

    for i, s in enumerate(samples):
        rgb = s.x.clamp(0, 1).permute(1, 2, 0).numpy()
        pr_prob = probs[i, 0].clamp(0, 1).cpu().numpy()
        gt_i = gt_vis[i, 0].clamp(0, 1).cpu().numpy()
        pr = pred_vis[i, 0].clamp(0, 1).cpu().numpy()

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"input\n{s.stem}")
        axes[i, 1].imshow(gt_i, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("GT(dilated)")
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
    p.add_argument("--checkpoint", type=str, default="runs/stage1/checkpoints/best.pt")
    p.add_argument("--out-dir", type=str, default="runs/stage1/infer")
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
    p.add_argument("--label-source", type=str, choices=["auto", "file"], default="auto")
    p.add_argument("--mask-suffix", type=str, default="_grid_mask.png")
    p.add_argument("--label-cache-dir", type=str, default="runs/stage1/labels_cache")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--dot-radius", type=int, default=2, help="dilation radius for displaying sparse point masks")
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

    label_cache_dir = None
    if args.label_source == "auto" and args.label_cache_dir:
        label_cache_dir = Path(args.label_cache_dir)

    labeler = GridIntersectionPatchDataset(
        records,
        patch_size=int(args.patch_size),
        samples_per_epoch=1,
        seed=int(args.seed),
        label_source=str(args.label_source),  # type: ignore[arg-type]
        mask_suffix=str(args.mask_suffix),
        label_cache_dir=label_cache_dir,
    )

    samples = _make_samples(
        records,
        labeler=labeler,
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        patch_size=int(args.patch_size),
        crop=str(args.crop),
    )

    model = UNetRes(in_channels=3, base_channels=int(args.base_channels), out_channels=1)
    state = _load_checkpoint_state_dict(ckpt_path)
    model.load_state_dict(state, strict=True)
    model.to(args.device)
    model.eval()

    xs = torch.stack([s.x for s in samples], dim=0)
    ys = torch.stack([s.y for s in samples], dim=0)
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
        dot_radius=int(args.dot_radius),
        dpi=int(args.dpi),
        title="stage1 inference",
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
