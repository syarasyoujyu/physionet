from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage1.data import (
    GridIntersectionPatchDataset,
    discover_records,
    precompute_grid_intersection_labels,
)
from stage1.losses import BCEWithLogitsLoss2D
from stage1.metrics import iou_from_logits
from stage1.model import UNetRes


@dataclass
class TrainConfig:
    data_root: str
    out_dir: str
    max_data_num: int | None
    init_from: str | None
    init_strict: bool
    precompute_cache: bool
    train_fraction: float
    val_fraction: float
    epochs: int
    lr: float
    batch_size: int
    patch_size: int
    samples_per_epoch: int
    val_samples: int
    base_channels: int
    num_workers: int
    seed: int
    device: str
    label_source: str
    mask_suffix: str
    label_cache_dir: str | None
    pos_weight: float | None
    log_level: str
    log_interval: int


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_checkpoint(path: Path, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int, best_iou: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_iou": best_iou,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
        },
        path,
    )


def _worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    if hasattr(ds, "rng"):
        base = int(torch.initial_seed()) % (2**32)
        ds.rng = random.Random(base + worker_id)


def _split_records(records: list, val_fraction: float) -> tuple[list, list]:
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("--val-fraction must be in [0, 1).")
    if val_fraction == 0.0 or len(records) < 2:
        return records, records

    def key(stem: str) -> int:
        h = hashlib.md5(stem.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little")

    sorted_records = sorted(records, key=lambda r: key(getattr(r, "stem", str(r))))
    n_val = max(1, int(round(len(sorted_records) * val_fraction)))
    val = sorted_records[:n_val]
    train = sorted_records[n_val:]
    if not train:
        train = val
    return train, val


def _setup_logging(out_dir: Path, level: str) -> logging.Logger:
    lvl = getattr(logging, level.upper(), None)
    if not isinstance(lvl, int):
        raise ValueError(f"invalid log level: {level!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage1.train")
    logger.setLevel(lvl)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    class _TqdmHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                msg = self.format(record)
                tqdm.write(msg)
            except Exception:
                self.handleError(record)

    th = _TqdmHandler()
    th.setLevel(lvl)
    th.setFormatter(fmt)
    logger.addHandler(th)

    fh = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
    fh.setLevel(lvl)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _load_checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj  # assume raw state_dict
    raise TypeError(f"unsupported checkpoint format: {type(obj)}")


def _init_model_from_checkpoint(logger: logging.Logger, model: torch.nn.Module, *, path: Path, strict: bool) -> None:
    state = _load_checkpoint_state_dict(path)
    if strict:
        model.load_state_dict(state, strict=True)
        logger.info("init_from=%s strict=True loaded_keys=%d", str(path), len(state))
        return

    # strict=False still errors on size mismatch, so filter those out.
    cur = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped_mismatch = 0
    for k, v in state.items():
        if k not in cur:
            continue
        if hasattr(v, "shape") and hasattr(cur[k], "shape") and v.shape != cur[k].shape:
            skipped_mismatch += 1
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    logger.info(
        "init_from=%s strict=False loaded=%d missing=%d unexpected=%d skipped_mismatch=%d",
        str(path),
        len(filtered),
        len(missing),
        len(unexpected),
        skipped_mismatch,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--out-dir", type=str, default="runs/stage1")
    p.add_argument("--max-data-num", type=int, default=None, help="maximum number of records to use (default: all)")
    p.add_argument("--init-from", type=str, default=None, help="initialize model weights from a checkpoint (.pt)")
    p.add_argument("--init-strict", action="store_true", help="strictly require all keys/shapes to match when loading --init-from")
    p.add_argument("--precompute-cache", action="store_true", help="precompute and cache masks before training (shows progress)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--samples-per-epoch", type=int, default=10_000)
    p.add_argument("--val-samples", type=int, default=1_000)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val-fraction", type=float, default=0.2, help="fraction of records used for validation")
    p.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        help="fraction of records used for training (overrides --val-fraction)",
    )
    p.add_argument("--label-source", type=str, choices=["auto", "file", "json"], default="json")
    p.add_argument("--mask-suffix", type=str, default="_grid_mask.png")
    p.add_argument("--label-cache-dir", type=str, default="runs/stage1/labels_cache")
    p.add_argument("--pos-weight", type=float, default=None)
    p.add_argument("--log-level", type=str, default="INFO", help="logging level (DEBUG/INFO/WARNING/ERROR)")
    p.add_argument("--log-interval", type=int, default=50, help="log every N train/val batches")
    args = p.parse_args()

    if args.train_fraction is not None:
        train_fraction = float(args.train_fraction)
        if not (0.0 < train_fraction <= 1.0):
            raise ValueError("--train-fraction must be in (0, 1].")
        val_fraction = 1.0 - train_fraction
    else:
        val_fraction = float(args.val_fraction)
        if not (0.0 <= val_fraction < 1.0):
            raise ValueError("--val-fraction must be in [0, 1).")
        train_fraction = 1.0 - val_fraction

    cfg = TrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        max_data_num=args.max_data_num,
        init_from=args.init_from,
        init_strict=bool(args.init_strict),
        precompute_cache=bool(args.precompute_cache),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        val_samples=args.val_samples,
        base_channels=args.base_channels,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        label_source=args.label_source,
        mask_suffix=args.mask_suffix,
        label_cache_dir=args.label_cache_dir if args.label_source != "file" else None,
        pos_weight=args.pos_weight,
        log_level=args.log_level,
        log_interval=args.log_interval,
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    logger = _setup_logging(out_dir, cfg.log_level)
    logger.info("stage1 training start")
    logger.info("python=%s platform=%s", sys.version.split()[0], platform.platform())
    logger.info(
        "torch=%s cuda_available=%s device=%s",
        getattr(torch, "__version__", "unknown"),
        torch.cuda.is_available(),
        cfg.device,
    )

    _seed_all(cfg.seed)
    logger.info("seed=%d", cfg.seed)

    records = discover_records(Path(cfg.data_root), max_data_num=cfg.max_data_num, seed=cfg.seed)
    logger.info("records=%d (max_data_num=%s)", len(records), cfg.max_data_num)
    if cfg.precompute_cache:
        stats = precompute_grid_intersection_labels(
            records,
            label_source=cfg.label_source,  # type: ignore[arg-type]
            mask_suffix=cfg.mask_suffix,
            label_cache_dir=None if cfg.label_cache_dir is None else Path(cfg.label_cache_dir),
            progress=True,
        )
        logger.info("precompute_cache done: %s", stats)
    train_records, val_records = _split_records(records, cfg.val_fraction)
    logger.info(
        "train_records=%d val_records=%d train_fraction=%.4f val_fraction=%.4f",
        len(train_records),
        len(val_records),
        cfg.train_fraction,
        cfg.val_fraction,
    )

    train_ds = GridIntersectionPatchDataset(
        train_records,
        patch_size=cfg.patch_size,
        samples_per_epoch=cfg.samples_per_epoch,
        seed=cfg.seed,
        label_source=cfg.label_source,  # type: ignore[arg-type]
        mask_suffix=cfg.mask_suffix,
        label_cache_dir=None if cfg.label_cache_dir is None else Path(cfg.label_cache_dir),
    )
    val_ds = GridIntersectionPatchDataset(
        val_records,
        patch_size=cfg.patch_size,
        samples_per_epoch=cfg.val_samples,
        seed=cfg.seed + 1,
        label_source=cfg.label_source,  # type: ignore[arg-type]
        mask_suffix=cfg.mask_suffix,
        label_cache_dir=None if cfg.label_cache_dir is None else Path(cfg.label_cache_dir),
    )
    logger.info(
        "dataset patch_size=%d samples_per_epoch=%d val_samples=%d label_source=%s",
        cfg.patch_size,
        cfg.samples_per_epoch,
        cfg.val_samples,
        cfg.label_source,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn if cfg.num_workers > 0 else None,
    )

    model = UNetRes(in_channels=3, base_channels=cfg.base_channels, out_channels=1)
    if cfg.init_from is not None:
        _init_model_from_checkpoint(logger, model, path=Path(cfg.init_from), strict=cfg.init_strict)
    model = model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = BCEWithLogitsLoss2D(pos_weight=cfg.pos_weight).to(cfg.device)
    logger.info(
        "model=UNetRes base_channels=%d batch_size=%d lr=%g pos_weight=%s num_workers=%d",
        cfg.base_channels,
        cfg.batch_size,
        cfg.lr,
        cfg.pos_weight,
        cfg.num_workers,
    )

    best_iou = -1.0
    progress_disable = not (hasattr(sys.stderr, "isatty") and sys.stderr.isatty())
    epoch_pbar = tqdm(
        range(1, cfg.epochs + 1),
        desc="[stage1] epochs",
        dynamic_ncols=True,
        disable=progress_disable,
    )
    for epoch in epoch_pbar:
        model.train()
        t0 = time()
        train_loss = 0.0
        n_train_batches = 0
        for x, y in train_loader:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * x.size(0)
            n_train_batches += 1
            if cfg.log_interval > 0 and (n_train_batches % cfg.log_interval) == 0:
                epoch_pbar.set_postfix(loss=f"{float(loss.item()):.4f}", lr=f"{opt.param_groups[0].get('lr', cfg.lr):g}")
        train_loss /= float(cfg.samples_per_epoch)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        n_seen = 0
        for x, y in val_loader:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            with torch.no_grad():
                logits = model(x)
                loss = loss_fn(logits, y)
            bs = x.size(0)
            val_loss += float(loss.item()) * bs
            val_iou += iou_from_logits(logits, y) * bs
            n_seen += bs
        val_loss /= max(1, n_seen)
        val_iou /= max(1, n_seen)

        dt = time() - t0
        epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_iou=f"{val_iou:.4f}",
            best_iou=f"{best_iou:.4f}" if best_iou >= 0 else "n/a",
        )
        tqdm.write(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_iou": val_iou,
                    "seconds": dt,
                },
                ensure_ascii=False,
            )
        )

        _save_checkpoint(out_dir / "checkpoints" / "last.pt", model, opt, epoch, best_iou)
        if val_iou > best_iou:
            best_iou = val_iou
            _save_checkpoint(out_dir / "checkpoints" / "best.pt", model, opt, epoch, best_iou)


if __name__ == "__main__":
    main()
