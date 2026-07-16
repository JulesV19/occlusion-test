"""Loss JEPA : invariance (MSE vs cible EMA stop-grad) + VICReg variance/covariance."""

import torch
import torch.nn.functional as F

from .config import Config


def variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Hinge sur l'écart-type par dimension : max(0, 1 - std_d). z : (N, D)."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(1.0 - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Somme des covariances hors-diagonale au carré, normalisée par D. z : (N, D)."""
    N, D = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (N - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / D


def jepa_loss(cfg: Config, z_pred: torch.Tensor, z_target: torch.Tensor,
              z_online: torch.Tensor) -> tuple:
    """
    z_pred   : (B, H, D) — embeddings prédits par rollout
    z_target : (B, H, D) — embeddings EMA des frames cibles (déjà détachés)
    z_online : (B, T, D) — embeddings de l'encodeur online (VICReg dessus)

    Retourne (loss_totale, dict de logs).
    """
    inv = F.mse_loss(z_pred, z_target.detach())
    z_flat = z_online.flatten(0, 1)
    var = variance_loss(z_flat)
    cov = covariance_loss(z_flat)
    total = cfg.lambda_inv * inv + cfg.lambda_var * var + cfg.lambda_cov * cov
    logs = {
        "loss": total.item(),
        "inv": inv.item(),
        "var": var.item(),
        "cov": cov.item(),
        "z_std": z_flat.std(dim=0).mean().item(),  # ≈1 attendu, ↓ = collapse
    }
    return total, logs
