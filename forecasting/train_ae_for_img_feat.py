import os
import sys
from pathlib import Path
from typing import Optional, Literal, Union

import torch
import torch.nn as nn
import torch.optim as optim

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


def train_and_val_ae(
    data_module: WaymoDataModule,
    ae: PatchAE,
    cut3r: ARCroco3DStereo,
    num_epochs: int = 12,
    lr: float = 1e-3,
    weight_decay: float = 0.0,           # set to 1e-4 for AdamW if desired
    use_adamw: bool = False,
    scheduler: Optional[Literal["cosine", "plateau"]] = "plateau",
    amp: bool = True,
    grad_clip_norm: Optional[float] = 1.0,
    log_dir: Union[str, os.PathLike] = "runs/patch_ae",
    ckpt_dir: Union[str, os.PathLike] = "checkpoints",
):
    # ---- setup ----
    # Some DataModules need explicit setup:
    if hasattr(data_module, "setup"):
        try:
            data_module.setup("fit")
        except TypeError:
            data_module.setup()

    train_loader = _get_loader(data_module, "train")
    val_loader = _get_loader(data_module, "val")

    # Make training deterministic-ish
    torch.backends.cudnn.benchmark = True  # speed for fixed-size inputs

    # Freeze CUT3R
    cut3r.eval()
    for p in cut3r.parameters():
        p.requires_grad = False

    ae = ae.to(device)

    # Optimizer & (optional) scheduler
    if use_adamw:
        optimizer = optim.AdamW(ae.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(ae.parameters(), lr=lr)

    if scheduler == "cosine":
        lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.1)
    elif scheduler == "plateau":
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    else:
        lr_sched = None

    scaler = GradScaler(enabled=amp)

    # Logging & checkpoints
    writer = SummaryWriter(log_dir)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    # ---- training loop ----
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        ae.train()
        running_loss = 0.0
        num_samples = 0

        for batch in train_loader:
            views = batch["views"]  # CUT3R consumes this; no manual .to(device) typically needed
            # If your 'views' tensors require device placement, handle it here.

            # 1) Encode features with CUT3R (frozen)
            with torch.no_grad():
                img_feats = _encode_with_cut3r(cut3r, views)

            # 2) Forward + loss (AMP)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                recon = ae(img_feats)
                loss = ae_loss(img_feats, recon)

            # 3) Backprop
            scaler.scale(loss).backward()

            # 4) Optional grad clipping
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ae.parameters(), grad_clip_norm)

            # 5) Step
            scaler.step(optimizer)
            scaler.update()

            # Logging (per step)
            bs = recon.shape[0] if hasattr(recon, "shape") else 1
            running_loss += loss.item() * bs
            num_samples += bs
            global_step += 1

            if global_step % 50 == 0:
                writer.add_scalar("train/step_loss", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        # Epoch metrics
        train_loss = running_loss / max(num_samples, 1)

        # ---- validation ----
        val_loss = validate_epoch(ae, cut3r, val_loader, amp=amp)

        # Schedulers
        if isinstance(lr_sched, optim.lr_scheduler.ReduceLROnPlateau):
            lr_sched.step(val_loss)
        elif lr_sched is not None:
            lr_sched.step()

        # Log epoch metrics
        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("train/lr_epoch_end", optimizer.param_groups[0]["lr"], epoch)

        print(f"[Epoch {epoch:03d}/{num_epochs}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.3e}")

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = ckpt_dir / f"patchae_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": ae.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": {
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "use_adamw": use_adamw,
                        "scheduler": scheduler,
                        "amp": amp,
                        "grad_clip_norm": grad_clip_norm,
                    },
                },
                ckpt_path,
            )
            print(f"  → Saved best checkpoint to: {ckpt_path}")

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
        log_dir="runs/patch_ae",
        ckpt_dir="checkpoints",
    )
