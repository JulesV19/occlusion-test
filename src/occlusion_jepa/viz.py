"""Visualisations : sanity-check données, courbes de loss, PCA 3D (plotly),
métriques latentes du rollout, grilles de rollouts décodés."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from .config import Config
from .data import sample_batch
from .probes import encode_dataset, rollout


# --------------------------------------------------------- sanity check data

def plot_sequence(cfg: Config, seed: int = 0):
    """Grille d'une séquence générée (contexte | horizon), flags d'occlusion."""
    batch = sample_batch(cfg, 1, seed=seed)
    frames = batch["frames"][0, :, 0].numpy()
    occ = batch["occluded"][0].numpy()
    T = frames.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(1.3 * T, 1.8))
    for t, ax in enumerate(axes):
        ax.imshow(frames[t], cmap="gray", vmin=0, vmax=1)
        title = f"t={t}"
        if t < cfg.context_len:
            title += "\nctx"
        if occ[t]:
            title += "\nOCC"
            for s in ax.spines.values():
                s.set_edgecolor("red"), s.set_linewidth(2)
        ax.set_title(title, fontsize=7)
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_history(history: list):
    """Courbes des termes de loss + std des embeddings (détection de collapse)."""
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    for ax, key in zip(axes, ["inv", "var", "cov", "z_std"]):
        ax.plot(steps, [h[key] for h in history])
        ax.set_title(key), ax.set_xlabel("step"), ax.grid(alpha=0.3)
        if key == "inv":
            ax.set_yscale("log")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- PCA 3D

def pca3d_figures(cfg: Config, encoder, predictor, device: str,
                  n_seqs: int = 200, seed: int = 999):
    """Deux figures plotly :
    1. Nuage PCA 3D de tous les embeddings, coloré par x du disque,
       frames occluses en losanges noirs.
    2. Un épisode avec traversée : trajectoire latente réelle vs rollout prédit.
    """
    data = encode_dataset(cfg, encoder, n_seqs, device, seed=seed)
    z = data["z"].flatten(0, 1).numpy()
    pos = data["positions"].flatten(0, 1).numpy()
    occ = data["occluded"].flatten(0, 1).numpy()

    pca = PCA(n_components=3).fit(z)
    p = pca.transform(z)
    evr = pca.explained_variance_ratio_

    fig_cloud = go.Figure()
    fig_cloud.add_trace(go.Scatter3d(
        x=p[~occ, 0], y=p[~occ, 1], z=p[~occ, 2], mode="markers",
        marker=dict(size=2.5, color=pos[~occ, 0], colorscale="Turbo",
                    colorbar=dict(title="x disque"), opacity=0.7),
        name="visible"))
    fig_cloud.add_trace(go.Scatter3d(
        x=p[occ, 0], y=p[occ, 1], z=p[occ, 2], mode="markers",
        marker=dict(size=3.5, color="black", symbol="diamond", opacity=0.9),
        name="occlus"))
    fig_cloud.update_layout(
        title=f"PCA 3D des embeddings — var. expliquée {evr.sum():.0%} "
              f"({', '.join(f'{v:.0%}' for v in evr)})",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        legend=dict(x=0), margin=dict(l=0, r=0, t=40, b=0))

    # --- épisode avec traversée pendant l'horizon ---
    C = cfg.context_len
    crossing = data["occluded"][:, C:].any(dim=1)
    idx = int(torch.nonzero(crossing)[0]) if crossing.any() else 0
    z_ep = data["z"][idx].numpy()
    occ_ep = data["occluded"][idx].numpy()
    z_hat = rollout(cfg, encoder, predictor,
                    data["frames"][idx:idx + 1], device)[0].numpy()
    p_ep, p_hat = pca.transform(z_ep), pca.transform(z_hat)

    fig_ep = go.Figure()
    fig_ep.add_trace(go.Scatter3d(
        x=p_ep[:, 0], y=p_ep[:, 1], z=p_ep[:, 2],
        mode="lines+markers+text", text=[f"t{t}" for t in range(len(p_ep))],
        textfont=dict(size=8),
        marker=dict(size=4, color=["black" if o else "royalblue" for o in occ_ep]),
        line=dict(color="royalblue", width=3), name="réel (noir = occlus)"))
    fig_ep.add_trace(go.Scatter3d(
        x=p_hat[:, 0], y=p_hat[:, 1], z=p_hat[:, 2],
        mode="lines+markers+text", text=[f"t{C + t}" for t in range(len(p_hat))],
        textfont=dict(size=8),
        marker=dict(size=4, color="orange"),
        line=dict(color="orange", width=3, dash="dash"), name="rollout prédit"))
    fig_ep.update_layout(
        title="Trajectoire latente d'un épisode traversant la barre (même PCA)",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        legend=dict(x=0), margin=dict(l=0, r=0, t=40, b=0))
    return fig_cloud, fig_ep


# ----------------------------------------------------- métriques latentes

@torch.no_grad()
def latent_metrics(cfg: Config, ema_encoder, encoder, predictor, device: str,
                   n_seqs: int = 256, seed: int = 777) -> dict:
    """MSE / cosine entre ẑ prédit et z̄ EMA réel, par pas de rollout et
    ventilé visible vs occlus."""
    batch = sample_batch(cfg, n_seqs, seed=seed)
    frames = batch["frames"].to(device)
    C = cfg.context_len
    z_hat = rollout(cfg, encoder, predictor, batch["frames"], device)  # (N, H, D)
    z_bar = ema_encoder(frames[:, C:]).cpu()                           # (N, H, D)
    occ = batch["occluded"][:, C:]                                     # (N, H)

    mse = (z_hat - z_bar).pow(2).mean(dim=-1)          # (N, H)
    cos = F.cosine_similarity(z_hat, z_bar, dim=-1)    # (N, H)
    per_step = {
        "mse": mse.mean(dim=0).tolist(),
        "cosine": cos.mean(dim=0).tolist(),
        "frac_occlus": occ.float().mean(dim=0).tolist(),
    }
    split = {}
    for name, mask in [("visible", ~occ), ("occlus", occ)]:
        if mask.sum() > 0:
            split[name] = {"mse": float(mse[mask].mean()),
                           "cosine": float(cos[mask].mean()),
                           "n": int(mask.sum())}
    return {"per_step": per_step, "split": split}


def plot_latent_metrics(metrics: dict):
    per_step = metrics["per_step"]
    H = len(per_step["mse"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, key, label in [(axes[0], "mse", "MSE(ẑ, z̄)"),
                           (axes[1], "cosine", "cosine(ẑ, z̄)")]:
        ax.plot(range(H), per_step[key], marker="o")
        ax2 = ax.twinx()
        ax2.bar(range(H), per_step["frac_occlus"], alpha=0.15, color="red")
        ax2.set_ylim(0, 1), ax2.set_ylabel("frac. occlus", color="red")
        ax.set_title(label), ax.set_xlabel("pas de rollout"), ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------ rollouts décodés

@torch.no_grad()
def plot_decoded_rollouts(cfg: Config, encoder, predictor, decoder_obs,
                          decoder_oracle, device: str, n_examples: int = 3,
                          seed: int = 31415):
    """Pour des séquences traversant la barre : frames réelles vs frames décodées
    depuis les ẑ prédits (decoder-obs et decoder-oracle)."""
    batch = sample_batch(cfg, 64, seed=seed)
    C, T = cfg.context_len, cfg.seq_len
    crossing = batch["occluded"][:, C:].any(dim=1)
    idxs = torch.nonzero(crossing)[:n_examples, 0]

    rows = []
    for i in idxs:
        i = int(i)
        z_hat = rollout(cfg, encoder, predictor,
                        batch["frames"][i:i + 1], device)[0].to(device)
        dec_obs = decoder_obs(z_hat)[:, 0].cpu().numpy()
        dec_ora = decoder_oracle(z_hat)[:, 0].cpu().numpy()
        rows.append((batch["frames"][i, :, 0].numpy(),
                     batch["occluded"][i].numpy(), dec_obs, dec_ora))

    n = len(rows)
    fig, axes = plt.subplots(3 * n, T, figsize=(1.1 * T, 3.4 * n))
    axes = axes.reshape(3 * n, T)
    for k, (real, occ, dec_obs, dec_ora) in enumerate(rows):
        for t in range(T):
            for j, (img, label) in enumerate([
                    (real[t], "réel"),
                    (dec_obs[t - C] if t >= C else None, "ẑ→obs"),
                    (dec_ora[t - C] if t >= C else None, "ẑ→oracle")]):
                ax = axes[3 * k + j, t]
                ax.set_xticks([]), ax.set_yticks([])
                if img is None:
                    ax.axis("off")
                    continue
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                if occ[t]:
                    for s in ax.spines.values():
                        s.set_edgecolor("red"), s.set_linewidth(2)
                if t == 0 or (t == C and j > 0):
                    ax.set_ylabel(label, fontsize=8)
    fig.suptitle("Rollouts décodés (bord rouge = disque totalement occlus)",
                 fontsize=10)
    fig.tight_layout()
    return fig
