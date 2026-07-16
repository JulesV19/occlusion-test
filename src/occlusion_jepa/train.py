"""Boucle d'entraînement JEPA : rollout multi-step + cible EMA + VICReg."""

import math

import torch
from tqdm.auto import tqdm

from .config import Config
from .data import make_loader
from .losses import jepa_loss
from .models import Encoder, Predictor, ema_update, make_ema_encoder


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


def train_jepa(cfg: Config, device: str | None = None):
    """Entraîne le JEPA. Retourne (encoder, ema_encoder, predictor, history)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    encoder = Encoder(cfg).to(device)
    ema_encoder = make_ema_encoder(encoder)
    predictor = Predictor(cfg).to(device)

    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_warmup_lr(s, cfg))

    loader = iter(make_loader(cfg))
    history = []
    pbar = tqdm(range(cfg.steps), desc="train JEPA")
    for step in pbar:
        batch = next(loader)
        frames = batch["frames"].to(device, non_blocking=True)  # (B, T, 1, S, S)

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

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            logs["step"] = step
            history.append(logs)
            pbar.set_postfix(inv=f"{logs['inv']:.4f}", var=f"{logs['var']:.3f}",
                             cov=f"{logs['cov']:.3f}", z_std=f"{logs['z_std']:.2f}")

    return encoder, ema_encoder, predictor, history


def save_checkpoint(path: str, cfg: Config, encoder, ema_encoder, predictor,
                    history) -> None:
    torch.save({
        "cfg": vars(cfg),
        "encoder": encoder.state_dict(),
        "ema_encoder": ema_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "history": history,
    }, path)


def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["cfg"])
    encoder = Encoder(cfg).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    ema_encoder = Encoder(cfg).to(device)
    ema_encoder.load_state_dict(ckpt["ema_encoder"])
    predictor = Predictor(cfg).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    return cfg, encoder, ema_encoder, predictor, ckpt["history"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out", type=str, default="jepa_ckpt.pt")
    args = parser.parse_args()

    cfg = Config()
    if args.steps is not None:
        cfg.steps = args.steps
        cfg.warmup_steps = min(cfg.warmup_steps, args.steps // 10)
    encoder, ema_encoder, predictor, history = train_jepa(cfg)
    save_checkpoint(args.out, cfg, encoder, ema_encoder, predictor, history)
    print(f"checkpoint -> {args.out}")
