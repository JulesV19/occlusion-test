"""Boucle d'entraînement JEPA : rollout multi-step + cible EMA + VICReg.

Dataset infini généré à la volée : une "epoch" = cfg.steps_per_epoch steps.
À la fin de chaque epoch : éval sur un set de validation figé, sauvegarde de
last.pt (toujours) et best.pt (quand val_inv s'améliore) — téléchargeables en
cours d'entraînement.
"""

import math
import os
from collections import defaultdict

import torch
from tqdm.auto import tqdm

from .config import Config
from .data import make_loader, sample_batch
from .losses import jepa_loss
from .models import Encoder, ema_update, make_ema_encoder, make_predictor

VAL_SEED = 987_654


def cosine_warmup_lr(step: int, cfg: Config) -> float:
    """Facteur multiplicatif : warmup linéaire puis décroissance cosine -> 0."""
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def ema_momentum(step: int, cfg: Config) -> float:
    """Schedule cosine : momentum_start -> momentum_end."""
    progress = step / max(1, cfg.steps)
    return cfg.ema_momentum_end - (cfg.ema_momentum_end - cfg.ema_momentum_start) \
        * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(cfg: Config, encoder, ema_encoder, predictor, val_batch,
             device: str) -> dict:
    """Loss de prédiction sur le set de validation figé, globale + frames
    occluses + cosine (métrique d'échelle plus stable que la MSE, la cible
    EMA dérivant au fil du training)."""
    encoder.eval(), predictor.eval()
    frames = val_batch["frames"].to(device)
    occ = val_batch["occluded"][:, cfg.context_len:].to(device)
    z_pred = predictor(encoder(frames[:, :cfg.context_len]), cfg.horizon)
    z_tgt = ema_encoder(frames[:, cfg.context_len:])
    err = (z_pred - z_tgt).pow(2).mean(dim=-1)                        # (N, H)
    cos = torch.nn.functional.cosine_similarity(z_pred, z_tgt, dim=-1)
    out = {"val_inv": err.mean().item(), "val_cos": cos.mean().item()}
    if occ.any():
        out["val_inv_occ"] = err[occ].mean().item()
    encoder.train(), predictor.train()
    return out


def train_jepa(cfg: Config, device: str | None = None, ckpt_dir: str = "."):
    """Entraîne le JEPA. Retourne (encoder, ema_encoder, predictor, history).

    history : un dict par epoch (moyennes train + métriques val).
    Checkpoints : {ckpt_dir}/last.pt à chaque epoch, {ckpt_dir}/best.pt sur
    amélioration de val_inv.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    os.makedirs(ckpt_dir, exist_ok=True)

    encoder = Encoder(cfg).to(device)
    ema_encoder = make_ema_encoder(encoder)
    predictor = make_predictor(cfg).to(device)

    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_warmup_lr(s, cfg))

    loader = iter(make_loader(cfg))
    val_batch = sample_batch(cfg, cfg.val_seqs, seed=VAL_SEED)

    history, best_val, step = [], math.inf, 0
    pbar = tqdm(total=cfg.steps, desc="train JEPA")
    for epoch in range(cfg.epochs):
        acc = defaultdict(float)
        for _ in range(cfg.steps_per_epoch):
            batch = next(loader)
            frames = batch["frames"].to(device, non_blocking=True)  # (B,T,1,S,S)

            z_online = encoder(frames)                              # (B, T, D)
            z_pred = predictor(z_online[:, :cfg.context_len], cfg.horizon)
            with torch.no_grad():
                z_target = ema_encoder(frames[:, cfg.context_len:])  # (B, H, D)

            loss, logs = jepa_loss(cfg, z_pred, z_target, z_online)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            ema_update(encoder, ema_encoder, ema_momentum(step, cfg))

            for k, v in logs.items():
                acc[k] += v
            step += 1
            pbar.update(1)
            if step % cfg.log_every == 0:
                pbar.set_postfix(inv=f"{logs['inv']:.4f}",
                                 z_std=f"{logs['z_std']:.2f}",
                                 best=f"{best_val:.4f}")

        rec = {k: v / cfg.steps_per_epoch for k, v in acc.items()}
        rec.update(evaluate(cfg, encoder, ema_encoder, predictor, val_batch, device))
        rec.update(epoch=epoch, step=step)
        history.append(rec)

        extra = {"epoch": epoch, "val_inv": rec["val_inv"]}
        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), cfg, encoder,
                        ema_encoder, predictor, history, extra=extra)
        is_best = rec["val_inv"] < best_val
        if is_best:
            best_val = rec["val_inv"]
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), cfg, encoder,
                            ema_encoder, predictor, history, extra=extra)
        pbar.write(
            f"epoch {epoch + 1:>3}/{cfg.epochs}  "
            f"inv {rec['inv']:.4f}  val_inv {rec['val_inv']:.4f}  "
            f"val_occ {rec.get('val_inv_occ', float('nan')):.4f}  "
            f"val_cos {rec['val_cos']:.3f}  z_std {rec['z_std']:.2f}"
            + ("  ← best.pt" if is_best else ""))
    pbar.close()

    return encoder, ema_encoder, predictor, history


def save_checkpoint(path: str, cfg: Config, encoder, ema_encoder, predictor,
                    history, extra: dict | None = None) -> None:
    torch.save({
        "cfg": vars(cfg),
        "encoder": encoder.state_dict(),
        "ema_encoder": ema_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "history": history,
        **(extra or {}),
    }, path)


def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["cfg"])
    if "predictor_type" not in ckpt["cfg"]:
        # checkpoints antérieurs au switch markov/gru : on infère depuis les poids
        cfg.predictor_type = ("gru" if any(k.startswith("gru.")
                                           for k in ckpt["predictor"]) else "markov")
    encoder = Encoder(cfg).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    ema_encoder = Encoder(cfg).to(device)
    ema_encoder.load_state_dict(ckpt["ema_encoder"])
    predictor = make_predictor(cfg).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    return cfg, encoder, ema_encoder, predictor, ckpt["history"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--ckpt-dir", type=str, default=".")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.steps_per_epoch is not None:
        cfg.steps_per_epoch = args.steps_per_epoch
    cfg.warmup_steps = min(cfg.warmup_steps, cfg.steps // 10)
    train_jepa(cfg, ckpt_dir=args.ckpt_dir)
    print(f"checkpoints -> {args.ckpt_dir}/best.pt, {args.ckpt_dir}/last.pt")
