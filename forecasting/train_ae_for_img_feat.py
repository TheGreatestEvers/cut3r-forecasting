import os
import sys
from pathlib import Path
from typing import Optional, Literal, Union

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Subset
import math
import random

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# AMP
from torch.cuda.amp import autocast, GradScaler

# make sure we can import your local package and the CUT3R submodule
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO_ROOT)

from forecasting.models.patch_ae import PatchAE, ae_loss
from cut3r.src.dust3r.model import ARCroco3DStereo
from forecasting.waymo_seq_dataset import WaymoDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_call(obj):
    """Support both .train_dataloader (attr) and .train_dataloader() (method)."""
    return obj() if callable(obj) else obj


def _get_loader(dm, split: Literal["train", "val", "test"]):
    name = f"{split}_dataloader"
    if not hasattr(dm, name):
        raise AttributeError(f"DataModule has no '{name}'")
    return _maybe_call(getattr(dm, name))


@torch.no_grad()
def _encode_with_cut3r(cut3r, views):
    # CUT3R is frozen; keep it under no_grad + eval for speed/stability
    (img_feats, _, _), _ = cut3r._forward_encoder(views=views)
    return img_feats


def validate_epoch(ae: nn.Module,
                   cut3r: nn.Module,
                   val_loader,
                   amp: bool = True) -> float:
    ae.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        views = batch["views"]  # list length B; CUT3R handles internals
        with torch.no_grad():
            img_feats = _encode_with_cut3r(cut3r, views)

        with torch.no_grad(), autocast(enabled=amp):
            recon = ae(img_feats)
            loss = ae_loss(img_feats, recon)

        bs = recon.shape[0] if hasattr(recon, "shape") else 1
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)

def make_quick_val_loader(val_loader, max_batches=16):
    """Wrap a val loader to yield only up to `max_batches` per call."""
    def _iter():
        count = 0
        for batch in val_loader:
            yield batch
            count += 1
            if count >= max_batches:
                break
    return _iter

def validate(ae, cut3r, val_loader_iter, amp=True):
    ae.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader_iter():
            views = batch["views"]
            # features from frozen CUT3R
            (img_feats, _, _), _ = cut3r._forward_encoder(views=views)
            with torch.cuda.amp.autocast(enabled=amp):
                recon = ae(img_feats)
                loss = ae_loss(img_feats, recon)
            bs = recon.shape[0] if hasattr(recon, "shape") else 1
            total += loss.item() * bs
            n += bs
    return total / max(1, n)

def train_and_val_ae(
    data_module,
    ae,
    cut3r,
    num_epochs=100,             # many epochs are fine if each epoch is short
    steps_per_epoch=5000,       # cap the work per epoch
    validate_every_n_steps=1000,# quick val cadence
    quick_val_max_batches=16,   # how many batches per quick val
    full_val_every_n_epochs=3,  # run full val occasionally
    lr=1e-3,
    amp=True,
    patience=5,                 # early stopping on full-val
):
    # --- setup ---
    if hasattr(data_module, "setup"):
        try:
            data_module.setup("fit")
        except TypeError:
            data_module.setup()

    train_loader = _get_loader(data_module, "train")
    val_loader_full = _get_loader(data_module, "val")

    # Optional: make a smaller *fixed* proxy val dataset for quick checks
    # If your val loader exposes `.dataset`, you can do:
    if hasattr(val_loader_full, "dataset"):
        val_indices = list(range(len(val_loader_full.dataset)))
        random.Random(0).shuffle(val_indices)
        small_n = min(len(val_indices), quick_val_max_batches * val_loader_full.batch_size * 4)
        val_subset = Subset(val_loader_full.dataset, val_indices[:small_n])
        from torch.utils.data import DataLoader
        val_loader_small = DataLoader(
            val_subset,
            batch_size=val_loader_full.batch_size,
            num_workers=getattr(val_loader_full, "num_workers", 4),
            pin_memory=getattr(val_loader_full, "pin_memory", True),
            persistent_workers=getattr(val_loader_full, "persistent_workers", True),
            shuffle=False,
        )
    else:
        val_loader_small = val_loader_full  # fallback

    quick_val_iter = make_quick_val_loader(val_loader_small, max_batches=quick_val_max_batches)
    full_val_iter  = make_quick_val_loader(val_loader_full,  max_batches=math.inf)  # consume all

    # freeze CUT3R
    cut3r.eval()
    for p in cut3r.parameters():
        p.requires_grad = False
    ae = ae.to(device)

    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    writer = SummaryWriter("runs/patch_ae_bigdata")
    best_full_val = float("inf")
    epochs_since_improve = 0
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        ae.train()
        running, seen = 0.0, 0

        for step, batch in enumerate(train_loader, start=1):
            # --- forward ---
            views = batch["views"]
            with torch.no_grad():
                (img_feats, _, _), _ = cut3r._forward_encoder(views=views)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                recon = ae(img_feats)
                loss = ae_loss(img_feats, recon)

            # --- backward ---
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = recon.shape[0] if hasattr(recon, "shape") else 1
            running += loss.item() * bs
            seen += bs
            global_step += 1

            # quick validation
            if global_step % validate_every_n_steps == 0:
                qval = validate(ae, cut3r, quick_val_iter, amp=amp)
                writer.add_scalar("val/quick_loss", qval, global_step)

            writer.add_scalar("train/step_loss", loss.item(), global_step)

            # cap work per epoch
            if step >= steps_per_epoch:
                break

        train_epoch_loss = running / max(1, seen)
        writer.add_scalar("train/epoch_loss", train_epoch_loss, epoch)
        print(f"[Epoch {epoch}] steps={step} train_loss={train_epoch_loss:.4f}")

        # occasional full validation
        if epoch % full_val_every_n_epochs == 0:
            full_val = validate(ae, cut3r, full_val_iter, amp=amp)
            writer.add_scalar("val/full_loss", full_val, epoch)
            print(f"  full_val_loss={full_val:.4f}")

            lr_sched.step(full_val)

            # early stopping + checkpoint
            if full_val < best_full_val - 1e-6:
                best_full_val = full_val
                epochs_since_improve = 0
                torch.save({"epoch": epoch,
                            "model_state": ae.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "val_loss": full_val},
                           "checkpoints/patchae_best.pt")
                print("  → Saved best checkpoint")
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print("Early stopping triggered.")
                    break

    writer.close()


if __name__ == "__main__":
    # Init data module
    waymo_data_module = WaymoDataModule(
        root="/mnt/datasets/waymo/",
        sequence_length=32,
        stride=4,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Init AE
    ae = PatchAE(
        in_dim=768,
        latent_dim=64,
        hidden=512,
        dropout=0.2,
    )

    # Init CUT3R (frozen)
    cut3r_weights_path = "cut3r/src/cut3r_512_dpt_4_64.pth"
    cut3r = ARCroco3DStereo.from_pretrained(cut3r_weights_path).eval().to(device)
    for p in cut3r.parameters():
        p.requires_grad = False

    train_and_val_ae(
        waymo_data_module,
        ae,
        cut3r,
        num_epochs=20,
        lr=1e-3,              # Good starting point for AE with Adam
        weight_decay=0.0,      # try AdamW + 1e-4 if you see overfitting
        use_adamw=False,
        scheduler="plateau",   # or "cosine"
        amp=True,              # mixed precision on GPUs
        grad_clip_norm=1.0,
        log_dir="/mnt/home/ae_training_logs/runs",
        ckpt_dir="/mnt/home/ae_training_checkpoints",
    )
