"""Probes post-hoc (encodeur gelé) : décodeurs de visualisation + probe linéaire.

- Decoder-obs    : z -> frame observée (avec barre)
- Decoder-oracle : z -> frame SANS barre (rendue depuis la position ground-truth).
  Test le plus direct : si le disque décodé apparaît à la bonne position pendant
  l'occlusion, l'information de position a survécu en latent.
- Probe linéaire : régression z -> (x, y), R²/MAE ventilés visible vs occlus.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

from .config import Config
from .data import make_sequence, render_frames, sample_batch
from .models import Decoder, Encoder


# ------------------------------------------------------------- helpers éval

@torch.no_grad()
def encode_dataset(cfg: Config, encoder: Encoder, n_seqs: int, device: str,
                   seed: int = 1234) -> dict:
    """Génère n_seqs séquences d'éval et les encode. Tenseurs sur CPU."""
    encoder.eval()
    batch = sample_batch(cfg, n_seqs, seed=seed)
    z = encoder(batch["frames"].to(device)).cpu()  # (N, T, D)
    return {**batch, "z": z}


@torch.no_grad()
def rollout(cfg: Config, encoder: Encoder, predictor,
            frames: torch.Tensor, device: str) -> torch.Tensor:
    """frames (B, T, 1, S, S) -> ẑ prédits (B, H, D) à partir du contexte. CPU."""
    encoder.eval()
    predictor.eval()
    z_ctx = encoder(frames[:, :cfg.context_len].to(device))
    return predictor(z_ctx, cfg.horizon).cpu()


# ------------------------------------------------------------- décodeurs

def train_decoder(cfg: Config, encoder: Encoder, device: str,
                  oracle: bool = False, seed: int = 42) -> Decoder:
    """Entraîne un décodeur z -> frame, encodeur gelé.

    oracle=False : cible = frame observée (avec barre)
    oracle=True  : cible = frame sans barre, rendue depuis la position ground-truth
    """
    encoder.eval()
    decoder = Decoder(cfg).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=cfg.probe_lr)
    rng = np.random.default_rng(seed)
    # ~8 frames par séquence suffisent pour remplir un batch de frames
    seqs_per_batch = max(1, cfg.probe_batch_size // cfg.seq_len)

    name = "decoder-oracle" if oracle else "decoder-obs"
    for _ in tqdm(range(cfg.probe_steps), desc=name):
        seqs = [make_sequence(cfg, rng) for _ in range(seqs_per_batch)]
        frames = torch.stack([s["frames"] for s in seqs])          # (B, T, 1, S, S)
        if oracle:
            targets = torch.stack([
                torch.from_numpy(render_frames(cfg, s["positions"].numpy(),
                                               with_bar=False))
                for s in seqs
            ])
        else:
            targets = frames
        frames = frames.to(device).flatten(0, 1)
        targets = targets.to(device).flatten(0, 1)

        with torch.no_grad():
            z = encoder(frames)
        loss = F.mse_loss(decoder(z), targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    decoder.eval()
    return decoder


# ------------------------------------------------------------- probe linéaire

def fit_linear_probe(z: torch.Tensor, positions: torch.Tensor) -> LinearRegression:
    """Régression linéaire z (N, D) -> (x, y) (N, 2)."""
    probe = LinearRegression()
    probe.fit(z.numpy(), positions.numpy())
    return probe


def _split_metrics(probe, z, positions, occluded) -> dict:
    """MAE (px) et R² ventilés visible / occlus."""
    pred = probe.predict(z.numpy())
    err = np.abs(pred - positions.numpy()).mean(axis=1)  # MAE par frame
    occ = occluded.numpy()
    out = {}
    for name, mask in [("visible", ~occ), ("occlus", occ)]:
        if mask.sum() == 0:
            continue
        ss_res = ((pred[mask] - positions.numpy()[mask]) ** 2).sum()
        ss_tot = ((positions.numpy()[mask]
                   - positions.numpy()[mask].mean(axis=0)) ** 2).sum()
        out[name] = {"mae_px": float(err[mask].mean()),
                     "r2": float(1 - ss_res / ss_tot),
                     "n": int(mask.sum())}
    return out


def probe_report(cfg: Config, encoder: Encoder, predictor,
                 device: str, n_train: int = 512, n_eval: int = 256) -> dict:
    """Probe linéaire entraînée sur embeddings réels, évaluée sur :
    - embeddings réels held-out (visible vs occlus)
    - embeddings PRÉDITS ẑ du rollout (visible vs occlus) — le test clé.
    """
    train = encode_dataset(cfg, encoder, n_train, device, seed=1234)
    eval_ = encode_dataset(cfg, encoder, n_eval, device, seed=5678)

    probe = fit_linear_probe(train["z"].flatten(0, 1),
                             train["positions"].flatten(0, 1))

    report = {"z_reel": _split_metrics(
        probe,
        eval_["z"].flatten(0, 1),
        eval_["positions"].flatten(0, 1),
        eval_["occluded"].flatten(0, 1),
    )}

    z_pred = rollout(cfg, encoder, predictor, eval_["frames"], device)
    C = cfg.context_len
    report["z_predit"] = _split_metrics(
        probe,
        z_pred.flatten(0, 1),
        eval_["positions"][:, C:].flatten(0, 1),
        eval_["occluded"][:, C:].flatten(0, 1),
    )
    return report
