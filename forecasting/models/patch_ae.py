# forecasting/models/patch_ae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchAE(nn.Module):
    """Per-patch MLP autoencoder: [B,P,D] -> [B,P,d] -> [B,P,D]."""
    def __init__(self, in_dim=768, latent_dim=128, hidden=512, dropout=0.0):
        super().__init__()
        self.enc = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )
        self.dec = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x):  # x: [B,P,D]
        return self.enc(x)

    def decode(self, z):  # z: [B,P,d]
        return self.dec(z)

    def forward(self, x):
        z = self.encode(x)
        xh = self.decode(z)
        return z, xh


def ae_loss(x, xh, cosine_weight: float = 0.1):
    """MSE + a bit of cosine to stabilize direction."""
    mse = F.mse_loss(xh, x)
    if cosine_weight <= 0:
        return mse
    cos = 1 - F.cosine_similarity(
        xh.reshape(-1, xh.size(-1)),
        x.reshape(-1, x.size(-1)),
        dim=1,
    ).mean()
    return mse + cosine_weight * cos
