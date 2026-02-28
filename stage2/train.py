from __future__ import annotations

import argparse
import json
import random
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage2.data import WaveformPatchDataset, discover_records
from stage2.losses import IoULoss
from stage2.metrics import iou_from_logits
from stage2.model import UNetRes


@dataclass
class TrainConfig:
    data_root: str
    out_dir: str
    max_data_num: int | None
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
    cache_dir: str | None
    line_width: int
    positive_sample_prob: float
    val_fraction: float


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--out-dir", type=str, default="runs/stage2")
    p.add_argument("--max-data-num", type=int, default=None, help="maximum number of records to use (default: all)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--samples-per-epoch", type=int, default=10_000)
    p.add_argument("--val-samples", type=int, default=1_000)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--cache-dir", type=str, default="runs/stage2/masks_cache")
    p.add_argument("--line-width", type=int, default=2)
    p.add_argument("--positive-sample-prob", type=float, default=0.7)
    args = p.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        max_data_num=args.max_data_num,
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
        cache_dir=args.cache_dir,
        line_width=args.line_width,
        positive_sample_prob=args.positive_sample_prob,
        val_fraction=args.val_fraction,
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    _seed_all(cfg.seed)

    records = discover_records(Path(cfg.data_root), max_data_num=cfg.max_data_num, seed=cfg.seed)
    train_records, val_records = _split_records(records, cfg.val_fraction)
    train_ds = WaveformPatchDataset(
        train_records,
        patch_size=cfg.patch_size,
        samples_per_epoch=cfg.samples_per_epoch,
        seed=cfg.seed,
        cache_dir=None if cfg.cache_dir is None else Path(cfg.cache_dir),
        line_width=cfg.line_width,
        positive_sample_prob=cfg.positive_sample_prob,
    )
    val_ds = WaveformPatchDataset(
        val_records,
        patch_size=cfg.patch_size,
        samples_per_epoch=cfg.val_samples,
        seed=cfg.seed + 1,
        cache_dir=None if cfg.cache_dir is None else Path(cfg.cache_dir),
        line_width=cfg.line_width,
        positive_sample_prob=cfg.positive_sample_prob,
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

    model = UNetRes(in_channels=3, base_channels=cfg.base_channels, out_channels=1).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = IoULoss().to(cfg.device)

    best_iou = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"[stage2] train epoch {epoch}/{cfg.epochs}", leave=False):
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * x.size(0)
        train_loss /= float(cfg.samples_per_epoch)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        n_seen = 0
        for x, y in tqdm(val_loader, desc=f"[stage2] val epoch {epoch}/{cfg.epochs}", leave=False):
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
        print(
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
