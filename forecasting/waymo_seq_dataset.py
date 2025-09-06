import os
import glob
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from PIL import Image
import io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Map Waymo camera enums -> strings (if your parquet uses ints)
CAMERA_MAP = {
    0: "UNKNOWN",
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}

# Column names (adapt if your parquet differs)
COL_SEG   = "key.segment_context_name"
COL_TS    = "key.frame_timestamp_micros"
COL_CAM   = "key.camera_name"                 # int or str
COL_IMG   = "[CameraImageComponent].image"    # raw bytes (usually JPEG)
COL_POSE= "[CameraImageComponent].pose.transform"
NEEDED_FOR_INDEX = [COL_SEG, COL_TS, COL_CAM]
NEEDED_FOR_ITEM  = [COL_IMG, COL_TS, COL_SEG, COL_CAM]

def _list_parquets(root: str, split: str) -> List[str]:
    # e.g., root/waymo/training/camera_image/*.parquet or root/training/camera_image/*.parquet
    candidates = [
        os.path.join(root, split, "camera_image", "*.parquet"),
        os.path.join(root, "waymo", split, "camera_image", "*.parquet"),
    ]
    files = []
    for pat in candidates:
        files.extend(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"No parquet files found for split='{split}' under {root}. "
                                f"Tried patterns: {candidates}")
    return sorted(files)

def _to_cam_str(series: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(series):
        return series.map(CAMERA_MAP)
    return series.astype(str)

def _to_pose4x4(raw):
    M = np.array(raw, dtype=np.float32)
    if M.ndim == 1 and M.size == 16:
        M = M.reshape(4, 4)
    if M.shape != (4, 4):
        # fallback if something is off
        M = np.eye(4, dtype=np.float32)
    return M

def _decode_to_chw_tensor(img_bytes, resize_to=None):
    """bytes -> torch.FloatTensor [1,3,H,W] with values in [0,1]"""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    true_H, true_W = img.height, img.width
    if resize_to is not None:
        img = img.resize(resize_to[::-1], Image.BILINEAR)  # resize_to=(H,W)
    arr = np.array(img, dtype=np.uint8)          # H x W x 3
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # [3,H,W]
    tensor = tensor.unsqueeze(0)                 # [1,3,H,W]
    return tensor, (true_H, true_W)

def collate_views_and_tensor(batch):
    """
    batch: List[ sample ], where sample = List[ view_dict ] of length T
      view_dict has:
        - "img": torch.Tensor [1,3,H,W] or [3,H,W]
        - "camera_pose": torch.Tensor [1,4,4] or [4,4]
        - (other keys left inside `views`)
    Returns:
      {
        "views": batch-as-is (List[List[Dict]]),
        "images": torch.FloatTensor [B,T,3,H,W],
        "camera_poses": torch.FloatTensor [B,T,4,4],
      }
    """
    B = len(batch)
    assert B > 0
    T = len(batch[0])  # fixed sequence_length

    images_btchw = []
    poses_bt44 = []

    for sample in batch:  # sample: List[view] length T
        imgs_tchw = []
        poses_t44 = []
        for v in sample:
            img = v["img"]
            # make sure img is [3,H,W]
            if img.dim() == 4 and img.size(0) == 1:
                img = img.squeeze(0)
            elif img.dim() == 3:
                pass
            else:
                raise ValueError(f"Unexpected img shape: {tuple(img.shape)}")

            pose = v["camera_pose"]
            # make sure pose is [4,4]
            if pose.dim() == 3 and pose.size(0) == 1:
                pose = pose.squeeze(0)
            elif pose.dim() == 2:
                pass
            else:
                raise ValueError(f"Unexpected pose shape: {tuple(pose.shape)}")

            imgs_tchw.append(img)     # [3,H,W]
            poses_t44.append(pose)    # [4,4]

        imgs_tchw = torch.stack(imgs_tchw, dim=0)   # [T,3,H,W]
        poses_t44 = torch.stack(poses_t44, dim=0)   # [T,4,4]
        images_btchw.append(imgs_tchw)
        poses_bt44.append(poses_t44)

    images = torch.stack(images_btchw, dim=0)       # [B,T,3,H,W]
    camera_poses = torch.stack(poses_bt44, dim=0)   # [B,T,4,4]

    return {
        "views": batch,              # List[List[Dict]] (unchanged)
        "images": images,            # Tensor [B,T,3,H,W]
        "camera_poses": camera_poses # Tensor [B,T,4,4]
    }

class WaymoSequenceDataset(Dataset):
    """
    Builds fixed-length windows of frames per (segment, camera) from Waymo camera_image parquet files.
    Each __getitem__ returns:
      - images: FloatTensor [T, C, H, W]  (after transform)
      - meta: dict with segment_id, camera_name, timestamps [T]
    """
    def __init__(
        self,
        root: str,
        split: str,                      # "training" | "validation" | "testing"
        sequence_length: int = 8,
        stride: int = 1,                 # frame step within a sequence window
        camera_name: str = "FRONT",      # filter to one camera; set to None for all cameras
        transform = None,                # torchvision transform applied per frame (PIL -> tensor)
        drop_last: bool = True,          # drop incomplete tails per (segment, camera)
        decode_threads: int = 0,         # reserved for future parallel decode
        window_overlap: bool = False
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.camera_name = camera_name
        self.drop_last = drop_last
        self.decode_threads = decode_threads
        self.window_overlap = window_overlap

        # Sensible default: turn PIL -> Tensor in [0,1]
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),                 # HWC uint8 -> CHW float in [0,1]
        ])

        # 1) discover parquet files
        self.files = _list_parquets(root, split)

        # 2) build the index of windows without loading images
        #    index entries: (file_idx, segment_id, camera_name, start_row, length, row_ids[T])
        self._windows: List[Tuple[int, str, str, int, int, List[int]]] = []
        self._build_index()

    def _scan_file_for_index(self, file_idx: int, path: str) -> None:
        # Load only light columns for indexing
        table = pq.read_table(path, columns=NEEDED_FOR_INDEX)
        df = table.to_pandas()

        # 2) Remember original row numbers before any sort
        df = df.reset_index().rename(columns={"index": "_orig_row"})

        cam_str = _to_cam_str(df[COL_CAM])
        df = df.assign(_cam=cam_str)

        # Optional camera filter
        if self.camera_name is not None:
            df = df[df["_cam"] == self.camera_name]

        # Guard: remove rows with missing timestamps/segments
        df = df.dropna(subset=[COL_SEG, COL_TS])

        # Sort within file by (segment, camera, timestamp)
        df = df.sort_values([COL_SEG, "_cam", COL_TS]).reset_index(drop=True)

        # Group by (segment, camera) and build windows using ORIGINAL row ids
        for (seg, cam), grp in df.groupby([COL_SEG, "_cam"], sort=False):
            n = len(grp)
            T, s = self.sequence_length, self.stride
            step = getattr(self, "window_start_step", 1)  # default overlap=1
            max_start = n - (T - 1) * s
            for start in range(0, max(0, max_start), step):
                idxs = [start + k * s for k in range(T)]
                if idxs[-1] >= n:
                    continue
                # Grab original row numbers for this window
                row_ids = grp["_orig_row"].iloc[idxs].tolist()
                self._windows.append((file_idx, seg, cam, start, T, row_ids))

    def _build_index(self) -> None:
        for fi, path in enumerate(self.files):
            self._scan_file_for_index(fi, path)

        if len(self._windows) == 0:
            raise RuntimeError("No sequence windows found. "
                               "Check camera filter, sequence_length/stride, and parquet columns.")

    def __len__(self) -> int:
        return len(self._windows)

    def _read_rows(self, file_idx: int, row_ids: List[int]) -> pd.DataFrame:
        path = self.files[file_idx]
        # Read minimal columns needed for __getitem__
        table = pq.read_table(path, columns=NEEDED_FOR_ITEM)
        df = table.to_pandas()
        # Fancy-index select the rows by integer positions
        # (they refer to the order of rows in this parquet file)
        return df.iloc[row_ids].reset_index(drop=True)

    def _decode_image(self, raw: bytes) -> Image.Image:
        # raw is expected to be encoded bytes (JPEG/PNG)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_idx, seg, cam, _start, T, row_ids = self._windows[idx]
        path = self.files[file_idx]

        # We need image bytes, timestamps (optional), and pose (if available)
        need_cols = [COL_IMG, COL_TS, COL_SEG, COL_CAM]
        if hasattr(self, "has_pose") and self.has_pose:
            need_cols.append(COL_POSE)
        else:
            # try to read pose; if it fails once, remember it
            try:
                pq.read_table(path, columns=[COL_POSE], memory_map=True)
                self.has_pose = True
                need_cols.append(COL_POSE)
            except Exception:
                self.has_pose = False

        table = pq.read_table(path, columns=need_cols, memory_map=True)
        sub = table.take(pa.array(row_ids)).to_pandas()

        views = []
        for i, row in sub.iterrows():
            # --- image ---
            img4d, (true_H, true_W) = _decode_to_chw_tensor(
                row[COL_IMG],
                resize_to=getattr(self, "resize_to", None)  # e.g., (H,W) or None
            )  # [1,3,H,W]

            B, C, H, W = img4d.shape  # B=1
            # --- pose ---
            if self.has_pose and (COL_POSE in row) and (row[COL_POSE] is not None):
                pose44 = _to_pose4x4(row[COL_POSE])
            else:
                pose44 = np.eye(4, dtype=np.float32)

            # --- build view dict in your model's format ---
            view = {
                "img": img4d,  # [1,3,H,W] float in [0,1]
                "ray_map": torch.full((B, 6, H, W), torch.nan),  # [1,6,H,W]
                "true_shape": torch.from_numpy(np.array([true_H, true_W], dtype=np.int32)),
                "idx": i,                          # position within this sequence
                "instance": str(i),                # or f"{seg}_{cam}_{int(row[COL_TS])}"
                "camera_pose": torch.from_numpy(pose44).unsqueeze(0),  # [1,4,4]
                "img_mask": torch.tensor(True).unsqueeze(0),    # [1]
                "ray_mask": torch.tensor(False).unsqueeze(0),   # [1]
                "update": torch.tensor(True).unsqueeze(0),      # [1]
                "reset": torch.tensor(False).unsqueeze(0),      # [1]
            }
            views.append(view)

        return views

def waymo_dataloader(
    root: str,
    split: str,
    batch_size: int = 4,
    sequence_length: int = 8,
    stride: int = 1,
    camera_name: str = "FRONT",
    transform=None,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    ds = WaymoSequenceDataset(
        root=root,
        split=split,
        sequence_length=sequence_length,
        stride=stride,
        camera_name=camera_name,
        transform=transform,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,   # drop partial batch at epoch end
        collate_fn=collate_views_and_tensor
    )

# (Optional) A simple Lightning-style DataModule shim
class WaymoDataModule:
    def __init__(
        self,
        root: str,
        sequence_length: int = 8,
        stride: int = 1,
        camera_name: str = "FRONT",
        transform=None,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        self.root = root
        self.sequence_length = sequence_length
        self.stride = stride
        self.camera_name = camera_name
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return waymo_dataloader(
            root=self.root, split="training",
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            stride=self.stride,
            camera_name=self.camera_name,
            transform=self.transform,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return waymo_dataloader(
            root=self.root, split="validation",
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            stride=self.stride,
            camera_name=self.camera_name,
            transform=self.transform,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return waymo_dataloader(
            root=self.root, split="testing",
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            stride=self.stride,
            camera_name=self.camera_name,
            transform=self.transform,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
