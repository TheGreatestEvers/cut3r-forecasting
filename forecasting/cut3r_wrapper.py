# forecasting/cut3r_wrapper.py

import os, sys, torch
import torch.nn as nn
from typing import List, Optional

# add submodule to path
SUBMODULE = os.path.join(os.path.dirname(__file__), "..", "cut3r")
sys.path.append(os.path.abspath(SUBMODULE))

# ⚠️ Adjust this import path to your submodule layout
from cut3r.src.dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig
from transformers.file_utils import ModelOutput


class FeatureSink:
    """
    Minimal interface for anything that wants features.
    Implement .push(sample_meta: dict, feats: dict) to consume features.
    """
    def push(self, sample_meta: dict, feats: dict):
        pass


@dataclass
class EncoderOnlyOutput(ModelOutput):
    """
    Minimal output when running encoding only.
    Per-view lists aligned with the input 'views' order.
    """
    feats: List[torch.Tensor]              # each [B_i, P, D] encoder tokens for that view (B_i = num True img_mask in batch)
    pos:   List[torch.Tensor]              # each [B_i, P, 2] 2D positions for that view
    shapes: List[torch.Tensor]             # each [B_i, 2] true (H,W)
    views: Optional[List[Any]] = None      # passthrough reference if you want it


class ARCroco3DEncodeOnly(ARCroco3DStereo):
    """
    A lightweight wrapper that *only* runs the image encoder path and stops.
    - Skips ray-map encoding, decoder, state, heads.
    - Taps encoder tokens and optionally sends them to a FeatureSink.
    - Returns EncoderOnlyOutput.
    """
    def __init__(
        self,
        config: ARCroco3DStereoConfig,
        feature_sink: FeatureSink | None = None,
        detach: bool = True,
        to_cpu_for_sink: bool = False,   # keep False for speed; set True if your sink is CPU-only
        cast_half_for_sink: bool = False,
    ):
        super().__init__(config)
        self._feature_sink = feature_sink
        self._detach = detach
        self._to_cpu_for_sink = to_cpu_for_sink
        self._cast_half_for_sink = cast_half_for_sink

    # ---- tiny helper to send to sink without side effects
    def _push(self, meta: dict, feats: dict):
        sink = self._feature_sink
        if sink is None:
            return
        safe = {}
        for k, v in feats.items():
            if torch.is_tensor(v):
                if self._detach: v = v.detach()
                if self._cast_half_for_sink: v = v.half()
                if self._to_cpu_for_sink: v = v.cpu()
            safe[k] = v
        try:
            sink.push(meta, safe)
        except Exception as e:
            print(f"[ARCroco3DEncodeOnly] sink.push failed: {e}")

    # ---- the only method you need to override
    @torch.no_grad()
    def forward(self, views, **kwargs) -> EncoderOnlyOutput:
        """
        Encode ONLY images for each view that has img_mask==True.
        Returns encoder tokens + positions per view, in input order.
        """
        device = views[0]["img"].device
        B = views[0]["img"].shape[0]

        feats_per_view: List[torch.Tensor] = []
        pos_per_view:   List[torch.Tensor] = []
        shapes_per_view: List[torch.Tensor] = []

        for i, view in enumerate(views):
            img: torch.Tensor = view["img"]          # [B,3,H,W]
            img_mask: torch.Tensor = view["img_mask"]# [B] bool
            # pick only the batch items that are True for this view
            if img_mask is None:
                select = torch.ones(B, dtype=torch.bool, device=device)
            else:
                select = img_mask.to(torch.bool)

            if not torch.any(select):
                # nothing to encode for this view
                feats_per_view.append(torch.empty(0))
                pos_per_view.append(torch.empty(0))
                shapes_per_view.append(torch.empty(0))
                continue

            imgs_sel = img[select]                                  # [B_i,3,H,W]
            if "true_shape" in view:
                shape_sel = view["true_shape"][select]              # [B_i,2]
            else:
                H, W = img.shape[-2:]
                shape_sel = torch.tensor([H, W], device=device)[None].repeat(imgs_sel.size(0), 1)

            # use the *original* encoder implementation
            outs, pos, _ = super()._encode_image(imgs_sel, shape_sel)
            x = outs[-1]  # [B_i, P, D]
            feats_per_view.append(x)
            pos_per_view.append(pos)
            shapes_per_view.append(shape_sel)

            # optional: send to sink (e.g., online AE trainer)
            self._push(
                meta={"tap": "encoder_img", "view_idx": i, "batch_size": int(x.shape[0])},
                feats={"f_img_enc": x, "pos_img": pos, "true_shape": shape_sel},
            )

        return EncoderOnlyOutput(
            feats=feats_per_view,
            pos=pos_per_view,
            shapes=shapes_per_view,
            views=views,
        )



class ARCroco3DEncodeTap(ARCroco3DStereo):
    """
    Exact same interface as ARCroco3DStereo, but intercepts encoder image tokens
    *inside* _encode_image and forwards them to a user-provided sink.

    - no decoder features
    - no state features
    """
    def __init__(
        self,
        config: ARCroco3DStereoConfig,
        feature_sink: FeatureSink | None = None,
        detach: bool = True,     # detach() before sending to sink
        to_cpu: bool = True,     # move to cpu before sending to sink
        cast_half: bool = False, # optionally cast to fp16 before sink
    ):
        super().__init__(config)
        self._feature_sink = feature_sink
        self._detach = detach
        self._to_cpu = to_cpu
        self._cast_half = cast_half

    def _maybe_push(self, meta: dict, feats: dict):
        if self._feature_sink is None:
            return
        safe = {}
        for k, v in feats.items():
            if torch.is_tensor(v):
                if self._detach:
                    v = v.detach()
                if self._cast_half:
                    v = v.half()
                if self._to_cpu:
                    v = v.cpu()
            safe[k] = v
        try:
            self._feature_sink.push(meta, safe)
        except Exception as e:
            # never break inference
            print(f"[ARCroco3DEncodeTap] sink.push failed: {e}")

    # -----------------------------
    # OVERRIDE: encoder image path
    # -----------------------------
    def _encode_image(self, image: torch.Tensor, true_shape: torch.Tensor):
        """
        IDENTICAL to CUT3R's _encode_image, except we tap the *post-encoder* tokens.
        """
        # --- begin: original CUT3R logic ---
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                # NOTE: if you don't need checkpoint here, you can skip for speed
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm(x)
        # --- end: original CUT3R logic ---

        # TAP: post-encoder tokens (what you asked for)
        # x:   [B, P, D]  — ViT patch tokens after enc_blocks + LayerNorm
        # pos: [B, P, 2]  — per-token 2D positions used by CUT3R
        self._maybe_push(
            meta={"tap": "encoder_img", "batch": image.shape[0]},
            feats={"f_img_enc": x, "pos_img": pos, "true_shape": true_shape},
        )

        # return exactly what base class returns
        return [x], pos, None
