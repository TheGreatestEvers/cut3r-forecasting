import os, io, glob, json, math, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

CAMERA_MAP = {
    0: "UNKNOWN", 1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT",
}
COL_SEG = "key.segment_context_name"
COL_TS  = "key.frame_timestamp_micros"
COL_CAM = "key.camera_name"
COL_IMG = "[CameraImageComponent].image"

def _list_parquets(root: str, split: str) -> List[str]:
    pats = [os.path.join(root, split, "camera_image", "*.parquet"),
            os.path.join(root, "waymo", split, "camera_image", "*.parquet")]
    files = []
    for p in pats: files.extend(glob.glob(p))
    if not files:
        raise FileNotFoundError(f"No parquet files for split='{split}' under {root}")
    return sorted(files)

def _to_cam_str(s: pd.Series) -> pd.Series:
    return s.map(CAMERA_MAP) if pd.api.types.is_integer_dtype(s) else s.astype(str)

def _decode_rgb(img_bytes, resize_to=None) -> torch.Tensor:
    with Image.open(io.BytesIO(img_bytes)) as im:
        im = im.convert("RGB")
        if resize_to is not None:
            # If you're on Pillow>=10, prefer: Image.Resampling.BILINEAR
            im = im.resize((resize_to[1], resize_to[0]), Image.BILINEAR)

        # Make a writable, C-contiguous array
        arr = np.array(im, dtype=np.uint8, copy=True)

    t = torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
    return t  # [3,H,W]

class WaymoFrameDataset(Dataset):
    """
    Each item is a single frame (image) with a stable UID (seg|cam|ts).
    """
    def __init__(self, root: str, split: str, camera_name: Optional[str] = "FRONT",
                 resize_to: Optional[Tuple[int,int]] = (384,640)):
        self.root = root; self.split = split; self.camera_name = camera_name; self.resize_to = resize_to
        self.files = _list_parquets(root, split)
        self.index: List[Tuple[int,int]] = []   # (file_idx, row_id)
        self.meta:  List[Tuple[str,str,int]] = []  # (seg, cam, ts)
        self._build_index()

    def _build_index(self):
        for fi, path in enumerate(self.files):
            tbl = pq.read_table(path, columns=[COL_SEG, COL_TS, COL_CAM])
            df  = tbl.to_pandas().reset_index(drop=True)
            df["_cam"] = _to_cam_str(df[COL_CAM])
            if self.camera_name is not None:
                df = df[df["_cam"] == self.camera_name]
            df = df.dropna(subset=[COL_SEG, COL_TS])

            # Bring original row ids into a column, then unpack directly
            for rid, seg, cam, ts in df[[COL_SEG, "_cam", COL_TS]] \
                                    .reset_index() \
                                    .itertuples(index=False, name=None):
                self.index.append((fi, int(rid)))        # row id in the parquet file
                self.meta.append((seg, cam, int(ts)))    # seg|cam|ts

    def __len__(self): return len(self.index)

    def uid_at(self, i: int) -> str:
        seg, cam, ts = self.meta[i]
        return f"{seg}|{cam}|{ts}"

    def __getitem__(self, i: int) -> Dict[str, Any]:
        fi, rid = self.index[i]
        path = self.files[fi]
        # read just needed column; this still reads the whole column into memory once per call,
        # but it's fine for a one-time precompute. For tighter IO, map row_ids to row-groups.
        tbl = pq.read_table(path, columns=[COL_IMG, COL_SEG, COL_CAM, COL_TS], memory_map=True)
        row = tbl.take(pa.array([rid])).to_pandas().iloc[0]
        img = _decode_rgb(row[COL_IMG], resize_to=self.resize_to)  # [3,H,W]
        #view = {
        #    "img": img.unsqueeze(0),        # [1,3,H,W] to match CUT3R's expected shape
        #    "camera_pose": torch.eye(4).unsqueeze(0),  # if unused by encoder, keep dummy
        #    "idx": 0, "instance": self.uid_at(i),
        #}
        view = {
                "img": img,
                "ray_map": torch.full(
                    (
                        img.shape[0],
                        6,
                        img.shape[-2],
                        img.shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": 2,
                "idx": 1,
                "instance": str(1),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
        return {"view": view, "uid": self.uid_at(i)}

def collate_views(batch):
    # batch: list of {"view": dict, "uid": str}
    views = [b["view"] for b in batch]
    uids  = [b["uid"]  for b in batch]
    return {"views": views, "uids": uids}

class WaymoFeatureWindowDataset(Dataset):
    """
    Yields windows of precomputed features for (segment,camera) sequences.
    Each item:
      {
        "img_feats": Tensor [T, C, H, W] (fp16),
        "uids": List[str],  # seg|cam|ts for each frame
        "meta": {"segment": str, "camera": str}
      }
    """
    def __init__(self, root: str, split: str, sequence_length: int = 8, stride: int = 1,
                 camera_name: Optional[str] = "FRONT", feat_dir: str = "", resize_to=(384,640)):
        self.root = root; self.split = split
        self.sequence_length = sequence_length; self.stride = stride
        self.camera_name = camera_name
        self.resize_to = resize_to
        self.files = _list_parquets(root, split)
        self.feat_dir = Path(feat_dir) / split
        self.windows: List[Tuple[int, List[int], str, str]] = []  # (file_idx, row_ids[T], seg, cam)
        self._build_index()

    def _build_index(self):
        for fi, path in enumerate(self.files):
            tbl = pq.read_table(path, columns=[COL_SEG, COL_TS, COL_CAM])
            df  = tbl.to_pandas().reset_index().rename(columns={"index":"_orig_row"})
            df["_cam"] = _to_cam_str(df[COL_CAM])
            if self.camera_name is not None:
                df = df[df["_cam"] == self.camera_name]
            df = df.dropna(subset=[COL_SEG, COL_TS]).sort_values([COL_SEG, "_cam", COL_TS]).reset_index(drop=True)
            for (seg, cam), grp in df.groupby([COL_SEG, "_cam"], sort=False):
                n = len(grp); T = self.sequence_length; s = self.stride
                step = 1  # full overlap (best for diversity)
                max_start = n - (T - 1)*s
                for start in range(0, max(0, max_start), step):
                    idxs = [start + k*s for k in range(T)]
                    if idxs[-1] >= n: continue
                    row_ids = grp.iloc[idxs]["_orig_row"].tolist()
                    self.windows.append((fi, row_ids, seg, cam))
        if not self.windows:
            raise RuntimeError("No windows found — check filters.")

    def __len__(self): return len(self.windows)

    def _uid(self, seg: str, cam: str, ts: int) -> str:
        return f"{seg}|{cam}|{ts}"

    def _feat_path(self, uid: str) -> Path:
        seg, cam, ts = uid.split("|")
        return self.feat_dir / seg / cam / f"{ts}.pt"

    def __getitem__(self, i: int) -> Dict[str, Any]:
        fi, row_ids, seg, cam = self.windows[i]
        path = self.files[fi]
        tbl = pq.read_table(path, columns=[COL_TS], memory_map=True)
        ts_list = tbl.take(pa.array(row_ids)).to_pandas()[COL_TS].astype(int).tolist()
        uids = [self._uid(seg, cam, ts) for ts in ts_list]

        feats = []
        for uid in uids:
            p = self._feat_path(uid)
            if not p.exists():
                raise FileNotFoundError(f"Missing feature file {p}. Did you run precompute for {self.split}?")
            feats.append(torch.load(p, map_location="cpu"))  # [C,H,W] fp16
        feats = torch.stack(feats, dim=0)  # [T,C,H,W]
        return {"img_feats": feats, "uids": uids, "meta": {"segment": seg, "camera": cam}}

def collate_feature_windows(batch):
    # stack to [B,T,C,H,W]
    feats = torch.stack([b["img_feats"] for b in batch], dim=0)
    uids  = [b["uids"] for b in batch]
    return {"img_feats": feats, "uids": uids}

