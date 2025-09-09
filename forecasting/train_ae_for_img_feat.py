import os
import sys
import math
import random
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

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
def _encode_with_cut3r(cut3r, views, amp=True):
    # inference_mode avoids autograd book-keeping and is faster than no_grad
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=amp and torch.cuda.is_available()):
        (img_feats, _, _), _ = cut3r._forward_encoder(views=views)
    return img_feats


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


def validate(ae: nn.Module, cut3r: nn.Module, val_loader_iter, amp: bool = True) -> float:
    ae.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader_iter():
            views = batch["views"]
            img_feats = _encode_with_cut3r(cut3r, views)
            with autocast(enabled=amp):
                recon = ae(img_feats)
                loss = ae_loss(img_feats, recon)
            bs = recon.shape[0] if hasattr(recon, "shape") else 1
            total += loss.item() * bs
            n += bs
    return total / max(1, n)


def train_and_val_ae(
    data_module,
    ae: nn.Module,
    cut3r: nn.Module,
    *,
    num_epochs: int = 100,             # many epochs are fine if each epoch is short
    steps_per_epoch: int = 5000,       # cap the work per epoch
    validate_every_n_steps: int = 1000,# quick val cadence
    quick_val_max_batches: int = 16,   # how many batches per quick val
    full_val_every_n_epochs: int = 3,  # run full val occasionally
    lr: float = 1e-3,
    amp: bool = True,
    patience: int = 5,                 # early stopping on full-val
    weight_decay: float = 0.0,
    use_adamw: bool = False,
    scheduler: Literal["plateau", "cosine", "none"] = "plateau",
    grad_clip_norm: float = 1.0,
    log_dir: str = "runs/patch_ae",
    ckpt_dir: str = "checkpoints",
    seed: int = 0,
):
    # --- setup (seeding, dirs, logging) ---
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    run_name = "patch_ae"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    if hasattr(data_module, "setup"):
        try:
            data_module.setup("fit")
        except TypeError:
            data_module.setup()

    train_loader = _get_loader(data_module, "train")
    val_loader_full = _get_loader(data_module, "val")

    # Optional: make a smaller *fixed* proxy val dataset for quick checks
    if hasattr(val_loader_full, "dataset"):
        val_indices = list(range(len(val_loader_full.dataset)))
        random.Random(seed).shuffle(val_indices)
        bs_guess = getattr(val_loader_full, "batch_size", 1) or 1
        small_n = min(len(val_indices), quick_val_max_batches * bs_guess * 4)
        val_subset = Subset(val_loader_full.dataset, val_indices[:small_n])
        val_loader_small = DataLoader(
            val_subset,
            batch_size=bs_guess,
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

    # optimizer
    if use_adamw:
        optimizer = optim.AdamW(ae.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

    # schedulers
    if scheduler == "plateau":
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    elif scheduler == "cosine":
        lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    else:
        lr_sched = None

    amp_enabled = bool(amp and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    best_full_val = float("inf")
    epochs_since_improve = 0
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        ae.train()
        running, seen = 0.0, 0
        steps_this_epoch = 0

        for step, batch in enumerate(train_loader, start=1):
            views = batch["views"]

            # features from frozen CUT3R
            with torch.no_grad():
                img_feats = _encode_with_cut3r(cut3r, views, amp=amp)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                recon = ae(img_feats)
                loss = ae_loss(img_feats, recon)

            # backward + step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(ae.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            bs = recon.shape[0] if hasattr(recon, "shape") else 1
            running += loss.item() * bs
            seen += bs
            global_step += 1
            steps_this_epoch += 1

            # quick validation
            if validate_every_n_steps and (global_step % validate_every_n_steps == 0):
                qval = validate(ae, cut3r, quick_val_iter, amp=amp_enabled)
                writer.add_scalar("val/quick_loss", qval, global_step)

            # logs
            writer.add_scalar("train/step_loss", loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            # cap work per epoch
            if steps_this_epoch >= steps_per_epoch:
                break

        train_epoch_loss = running / max(1, seen)
        writer.add_scalar("train/epoch_loss", train_epoch_loss, epoch)
        print(f"[Epoch {epoch}] steps={steps_this_epoch} train_loss={train_epoch_loss:.4f}")

        # occasional full validation
        if epoch % full_val_every_n_epochs == 0:
            full_val = validate(ae, cut3r, full_val_iter, amp=amp_enabled)
            writer.add_scalar("val/full_loss", full_val, epoch)
            print(f"  full_val_loss={full_val:.4f}")

            # step schedulers
            if isinstance(lr_sched, optim.lr_scheduler.ReduceLROnPlateau):
                lr_sched.step(full_val)
            elif lr_sched is not None:
                lr_sched.step()

            # early stopping + checkpoint
            if full_val < best_full_val - 1e-6:
                best_full_val = full_val
                epochs_since_improve = 0
                ckpt_path = os.path.join(ckpt_dir, "patchae_best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": ae.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": full_val,
                    },
                    ckpt_path,
                )
                print(f"  → Saved best checkpoint to {ckpt_path}")
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
        lr=1e-3,               # Good starting point for AE with Adam/AdamW
        weight_decay=0.0,      # try 1e-4 with AdamW if you see overfitting
        use_adamw=False,
        scheduler="plateau",   # or "cosine" or "none"
        amp=True,              # mixed precision (auto-disabled on CPU)
        grad_clip_norm=1.0,
        log_dir="/mnt/home/ae_training_logs/runs",
        ckpt_dir="/mnt/home/ae_training_checkpoints",
    )
