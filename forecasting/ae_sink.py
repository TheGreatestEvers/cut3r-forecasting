# forecasting/ae_sink_fast.py
import torch
from torch.nn.utils import clip_grad_norm_
from forecasting.models.patch_ae import ae_loss

class AESinkFast:
    """
    Efficient online AE trainer:
      - no train()/eval() flips
      - accumulates batches and steps periodically
      - optional patch subsampling
    """
    def __init__(
        self,
        ae,                       # nn.Module already set to .train() outside
        optimizer,
        device="cuda",
        cosine_weight=0.1,
        use_amp=True,
        grad_clip=1.0,
        # efficiency controls:
        step_every_n_views=4,     # accumulate features from N views, then step
        max_patches=None,         # sample K patches per frame if set (e.g., 96)
        max_accum_batches=8,      # safety cap on how many views to hold
        keep_on_gpu=True,         # avoid CPU<->GPU copies if you have VRAM
    ):
        self.ae = ae.to(device)
        self.opt = optimizer
        self.device = device
        self.cosine_weight = cosine_weight
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.use_amp = use_amp
        self.grad_clip = grad_clip

        self.step_every_n_views = max(1, step_every_n_views)
        self.max_patches = max_patches
        self.max_accum_batches = max_accum_batches
        self.keep_on_gpu = keep_on_gpu

        self._buf = []     # holds tensors [B,P,D] (on device)
        self._counter = 0

    def _maybe_step(self):
        if len(self._buf) == 0:
            return
        x = torch.cat(self._buf, dim=0)  # [B_total, P, D]
        self._buf.clear()

        self.opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            z, xh = self.ae(x)
            loss = ae_loss(x, xh, cosine_weight=self.cosine_weight)

        self.scaler.scale(loss).backward()
        if self.grad_clip and self.grad_clip > 0:
            self.scaler.unscale_(self.opt)
            clip_grad_norm_(self.ae.parameters(), self.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()

    def push(self, meta: dict, feats: dict):
        x = feats.get("f_img_enc", None)
        if x is None:
            return

        # If wrapper sent CPU tensors, move once here. Prefer sending GPU from wrapper if you can.
        if x.device.type == "cpu":
            x = x.to(self.device, non_blocking=True)

        # optional patch subsample
        if self.max_patches is not None and x.size(1) > self.max_patches:
            idx = torch.randperm(x.size(1), device=x.device)[: self.max_patches]
            x = x[:, idx]

        # accumulate
        self._buf.append(x if self.keep_on_gpu else x.cpu())
        self._counter += 1

        # step periodically or if buffer is large
        if (self._counter % self.step_every_n_views) == 0 or len(self._buf) >= self.max_accum_batches:
            self._maybe_step()
