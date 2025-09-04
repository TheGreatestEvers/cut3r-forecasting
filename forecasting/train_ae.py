# setup_encode_only_with_ae.py
import os
import sys
import torch
import torch.optim as optim

# make sure we can import your local package and the CUT3R submodule
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO_ROOT)

# --- imports from your repo (as we discussed) ---
from cut3r_wrapper import ARCroco3DEncodeOnly
from forecasting.models.patch_ae import PatchAE
from ae_sink import AESinkFast  # the fast accumulating sink

# --- import CUT3R config from the submodule ---
# adjust the path below to match your submodule’s module layout if needed
from cut3r.src.dust3r.model import ARCroco3DStereoConfig


def build_cfg(
    img_size=(224, 224),
    enc_embed_dim=768,
    enc_depth=24,
    enc_num_heads=16,
    dec_embed_dim=768,
    dec_depth=12,
    dec_num_heads=12,
):
    """Minimal config; heads/decoder params are irrelevant since we won’t use them."""
    return ARCroco3DStereoConfig(
        img_size=img_size,
        enc_embed_dim=enc_embed_dim,
        enc_depth=enc_depth,
        enc_num_heads=enc_num_heads,
        dec_embed_dim=dec_embed_dim,
        dec_depth=dec_depth,
        dec_num_heads=dec_num_heads,
        head_type="linear",
        output_mode="pts3d",
        # keep defaults for everything else
    )


def build_model_and_ae(
    cut3r_ckpt_path: str | None = None,
    latent_dim: int = 128,
    ae_hidden: int = 512,
    step_every_n_views: int = 4,
    max_patches: int | None = 96,
    use_amp: bool = True,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    cosine_weight: float = 0.1,
    cast_half_for_sink: bool = True,
    to_cpu_for_sink: bool = False,
):
    """
    Returns:
        model  : ARCroco3DEncodeOnly (frozen, eval)
        ae     : PatchAE (train mode)
        sink   : AESinkFast (online trainer)
        device : 'cuda' or 'cpu'
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Build CUT3R config & wrapper
    cfg = build_cfg(enc_embed_dim=768)  # adjust if your encoder dim differs
    model = ARCroco3DEncodeOnly(
        cfg,
        feature_sink=None,         # assigned after AE is created (below)
        detach=True,               # detach from CUT3R graph
        to_cpu_for_sink=to_cpu_for_sink,
        cast_half_for_sink=cast_half_for_sink,
    )

    # 2) Load CUT3R weights (strict=False to be robust to minor key diffs)
    if cut3r_ckpt_path and os.path.isfile(cut3r_ckpt_path):
        state = torch.load(cut3r_ckpt_path, map_location="cpu")
        sd = state.get("model", state)
        _ = model.load_state_dict(sd, strict=False)

    # freeze CUT3R and set eval
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # 3) Build AE + optimizer
    ae = PatchAE(in_dim=cfg.croco_kwargs.get("enc_embed_dim", 768), latent_dim=latent_dim, hidden=ae_hidden).to(device)
    ae.train()
    ae_opt = optim.AdamW(ae.parameters(), lr=lr, weight_decay=weight_decay)

    # 4) Build fast sink (accumulate, step periodically)
    sink = AESinkFast(
        ae=ae,
        optimizer=ae_opt,
        device=device,
        cosine_weight=cosine_weight,
        use_amp=use_amp,
        grad_clip=1.0,
        step_every_n_views=step_every_n_views,
        max_patches=max_patches,
        max_accum_batches=8,
        keep_on_gpu=not to_cpu_for_sink,
    )

    # 5) Attach sink to the model
    model._feature_sink = sink

    return model, ae, sink, device


if __name__ == "__main__":
    # example usage
    ckpt = os.path.join(REPO_ROOT, "third_party", "cut3r", "weights", "cut3r_ckpt.pth")  # <-- set your path or None
    model, ae, sink, device = build_model_and_ae(
        cut3r_ckpt_path=ckpt,
        latent_dim=128,
        step_every_n_views=4,
        max_patches=96,
        use_amp=True,
        cast_half_for_sink=True,
        to_cpu_for_sink=False,
    )

    # now call your normal inference code:
    # outputs = model(views)
    # the wrapper will stop after encoding and push encoder tokens to `sink`,
    # which trains the AE online (no feature files).
