# --- Distributed helpers ---
import os, time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from waymo_data_helpers import WaymoFrameDataset

def ddp_setup():
    """Initialize torch.distributed from torchrun env; returns (rank, world_size, local_rank, device)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        rank, world_size, local_rank = 0, 1, 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return rank, world_size, local_rank, device

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# --- Speed toggles (harmless on older GPUs) ---
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try: torch.set_float32_matmul_precision("high")
except Exception: pass

# Your existing helper from earlier:
def collate_views(batch):
    # batch: list of {"view": dict, "uid": str}
    views = [b["view"] for b in batch]
    uids  = [b["uid"]  for b in batch]
    return {"views": views, "uids": uids}

@torch.no_grad()
def precompute_cut3r_features_distributed(
    cut3r,
    root: str,
    split: str,
    out_dir: str,
    *,
    batch_size: int = 32,
    num_workers: int = 8,
    resize_to=(384, 640),
    amp: bool = True,
    skip_existing: bool = True,
    channels_last: bool = True,
):
    """
    Run with torchrun to use multiple GPUs:
      torchrun --standalone --nproc_per_node=4 train.py --do_precompute

    Saves: <out_dir>/<split>/<segment>/<camera>/<timestamp>.pt (fp16, [C,H,W])
    """
    rank, world_size, local_rank, device = ddp_setup()
    t0 = time.time()

    # Build the per-frame dataset once per rank
    ds = WaymoFrameDataset(root, split, camera_name=None, resize_to=resize_to)

    # Shard across ranks so there's no overlap
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_views,
        drop_last=False,
    )

    cut3r.eval().to(device)

    # Optional: channels_last can speed up conv nets
    def to_dev_img(t: torch.Tensor) -> torch.Tensor:
        if channels_last:
            return t.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        return t.to(device, non_blocking=True)

    out_root = Path(out_dir) / split

    def uid_to_path(uid: str) -> Path:
        seg, cam, ts = uid.split("|")
        return out_root / seg / cam / f"{ts}.pt"

    processed_local = 0
    skipped_local = 0

    for batch in dl:
        views = batch["views"]
        uids  = batch["uids"]

        # Fast resume: drop items that already exist on disk
        if skip_existing:
            keep = []
            keep_idx = []
            for i, uid in enumerate(uids):
                p = uid_to_path(uid)
                if p.exists():
                    skipped_local += 1
                    continue
                keep.append(views[i]); keep_idx.append(i)
            if not keep:
                continue
            views = keep
            uids  = [uids[i] for i in keep_idx]

        # Move inputs to device
        for v in views:
            # v["img"] is [1,3,H,W] float in [0,1]; if you change the dataset to return uint8,
            # you'd convert to float here (and divide by 255) to reduce H2D bandwidth.
            v["img"] = to_dev_img(v["img"])

        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            (img_feats, _, _), _ = cut3r._forward_encoder(views=views)   # [B, C, H, W]

        img_feats = img_feats.half().cpu().contiguous()  # fp16 saves space and IO

        # Atomic saves to tolerate preemption
        for i, uid in enumerate(uids):
            p = uid_to_path(uid)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".pt.tmp")
            torch.save(img_feats[i], tmp)
            os.replace(tmp, p)  # atomic rename on POSIX
            processed_local += 1

        # (Optional) light progress on rank 0
        if rank == 0 and processed_local % (batch_size * 20) == 0:
            print(f"[precompute {split}] rank0 processed≈{processed_local} (skipped≈{skipped_local})")

    # Gather counts for a small summary
    if dist.is_available() and dist.is_initialized():
        tens = torch.tensor([processed_local, skipped_local], device=device, dtype=torch.long)
        dist.all_reduce(tens, op=dist.ReduceOp.SUM)
        processed_total = int(tens[0].item()); skipped_total = int(tens[1].item())
    else:
        processed_total, skipped_total = processed_local, skipped_local

    if rank == 0:
        dt = time.time() - t0
        print(f"[precompute {split}] wrote {processed_total} features, skipped {skipped_total}, in {dt:.1f}s to {out_root}")

    ddp_cleanup()


if __name__ == "__main__":
    import argparse, os
    from pathlib import Path

    # Change this if your weights live elsewhere
    CUT3R_WEIGHTS = "cut3r/src/cut3r_512_dpt_4_64.pth"

    parser = argparse.ArgumentParser("Precompute CUT3R features (multi-GPU via torchrun)")
    parser.add_argument("--data-root", required=True, help="Waymo root dir")
    parser.add_argument("--out-dir",   required=True, help="Directory to write features")
    parser.add_argument("--split",     default="training", choices=["training", "validation", "testing"])
    parser.add_argument("--bs",        type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--workers",   type=int, default=min(16, (os.cpu_count() or 8)),
                        help="DataLoader workers per process")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Build CUT3R and go
    from cut3r.src.dust3r.model import ARCroco3DStereo
    cut3r = ARCroco3DStereo.from_pretrained(CUT3R_WEIGHTS)
    precompute_cut3r_features_distributed(
        cut3r=cut3r,
        root=args.data_root,
        split=args.split,
        out_dir=args.out_dir,
        batch_size=args.bs,
        num_workers=args.workers,
        resize_to=(384, 640),   # edit here if you need a different HxW
        amp=True,               # on by default; edit to False if you must
        channels_last=True,     # usually faster for conv nets
        skip_existing=True,     # resume-friendly; delete files to recompute
    )
