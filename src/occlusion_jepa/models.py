"""Encoder CNN, Predictor GRU (rollout autorégressif) et Decoder conv-transpose."""

import copy

import torch
import torch.nn as nn

from .config import Config


class Encoder(nn.Module):
    """Frame (1, 64, 64) -> embedding (embed_dim,). GroupNorm (pas de BatchNorm,
    interaction délicate avec la copie EMA)."""

    def __init__(self, cfg: Config):
        super().__init__()
        layers, in_ch = [], 1
        for ch in cfg.enc_channels:  # 64 -> 32 -> 16 -> 8 -> 4
            layers += [
                nn.Conv2d(in_ch, ch, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, ch),
                nn.SiLU(),
            ]
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        final_size = cfg.img_size // 2 ** len(cfg.enc_channels)
        self.fc = nn.Linear(in_ch * final_size ** 2, cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 1, S, S) ou (B, T, 1, S, S) -> (B, D) ou (B, T, D)."""
        seq = x.dim() == 5
        if seq:
            B, T = x.shape[:2]
            x = x.flatten(0, 1)
        z = self.fc(self.conv(x).flatten(1))
        return z.view(B, T, -1) if seq else z


class Predictor(nn.Module):
    """GRU : les C embeddings de contexte initialisent l'état caché, puis rollout
    autorégressif de H pas (le ẑ produit est réinjecté en entrée du pas suivant)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.gru = nn.GRU(cfg.embed_dim, cfg.gru_hidden,
                          num_layers=cfg.gru_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(cfg.gru_hidden, cfg.gru_hidden),
            nn.SiLU(),
            nn.Linear(cfg.gru_hidden, cfg.embed_dim),
        )

    def forward(self, z_context: torch.Tensor, horizon: int) -> torch.Tensor:
        """z_context : (B, C, D) -> ẑ prédits (B, H, D)."""
        out, h = self.gru(z_context)          # digestion du contexte
        z = self.head(out[:, -1])             # ẑ_{C} (premier pas prédit)
        preds = [z]
        for _ in range(horizon - 1):
            out, h = self.gru(z.unsqueeze(1), h)
            z = self.head(out[:, -1])
            preds.append(z)
        return torch.stack(preds, dim=1)


class Decoder(nn.Module):
    """Embedding (D,) -> frame (1, 64, 64). Utilisé uniquement en probe post-hoc."""

    def __init__(self, cfg: Config):
        super().__init__()
        chs = tuple(reversed(cfg.enc_channels))  # (256, 128, 64, 32)
        self.init_size = cfg.img_size // 2 ** len(chs)
        self.fc = nn.Linear(cfg.embed_dim, chs[0] * self.init_size ** 2)
        layers = []
        for i, ch in enumerate(chs):
            out_ch = chs[i + 1] if i + 1 < len(chs) else 32
            layers += [
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
            ]
        layers += [nn.Conv2d(32, 1, kernel_size=3, padding=1)]
        self.deconv = nn.Sequential(*layers)
        self.chs = chs

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, self.chs[0], self.init_size, self.init_size)
        return torch.sigmoid(self.deconv(x))


def make_ema_encoder(encoder: Encoder) -> Encoder:
    ema = copy.deepcopy(encoder)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update(online: Encoder, ema: Encoder, momentum: float) -> None:
    for p_o, p_e in zip(online.parameters(), ema.parameters()):
        p_e.mul_(momentum).add_(p_o, alpha=1 - momentum)
    for b_o, b_e in zip(online.buffers(), ema.buffers()):
        b_e.copy_(b_o)
