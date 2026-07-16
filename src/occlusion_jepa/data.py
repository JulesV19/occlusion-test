"""Génération à la volée de séquences : disque rebondissant + barre occlusive
dont la position est tirée aléatoirement PAR ÉPISODE (fixe au sein d'un épisode).

Chaque sample est un dict de tenseurs :
    frames    (T, 1, S, S) float32 [0, 1]  — frames observées (avec barre)
    positions (T, 2)       float32         — centre (x, y) ground-truth du disque
    occluded  (T,)         bool            — disque totalement caché par la barre
    bar_x     ()           float32         — bord gauche de la barre de l'épisode
avec T = context_len + horizon.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .config import Config


# ---------------------------------------------------------------- simulation

def sample_bar_x(cfg: Config, rng: np.random.Generator) -> float:
    """Bord gauche de la barre, uniforme dans cfg.bar_x_range."""
    return float(rng.uniform(*cfg.bar_x_range))


def simulate_trajectory(cfg: Config, rng: np.random.Generator,
                        ensure_crossing: bool, bar_x: float) -> np.ndarray:
    """Trajectoire (T, 2) : vitesse constante, rebonds élastiques sur les bords.

    Si ensure_crossing, la position/vitesse initiales sont choisies pour que le
    disque traverse le centre de la barre autour d'une frame tirée dans l'horizon.
    """
    T = cfg.seq_len
    r = cfg.disk_radius
    lo, hi = r, cfg.img_size - 1 - r  # bornes du centre du disque

    vx = rng.uniform(*cfg.speed_x_range)
    vy = rng.uniform(*cfg.speed_y_range) * rng.choice([-1.0, 1.0])

    if ensure_crossing:
        bar_center = bar_x + cfg.bar_width / 2
        # frame de traversée du centre de la barre, tirée dans l'horizon puis
        # plafonnée pour que le point de départ tienne dans l'écran — on préserve
        # vx (et donc la durée d'occlusion calibrée) plutôt que de le ralentir
        direction = rng.choice([-1.0, 1.0])
        max_dist = (bar_center - lo - 1) if direction > 0 else (hi - bar_center - 1)
        t_c = rng.integers(cfg.context_len, T - 1)
        t_c = max(1, min(t_c, int(max_dist / vx)))
        x = bar_center - direction * vx * t_c
        vx *= direction
    else:
        x = rng.uniform(lo, hi)
        vx *= rng.choice([-1.0, 1.0])
    y = rng.uniform(lo, hi)

    positions = np.empty((T, 2), dtype=np.float32)
    for t in range(T):
        positions[t] = (x, y)
        x += vx
        y += vy
        if x < lo or x > hi:  # rebond élastique
            x = np.clip(x, lo, hi) * 2 - x
            vx = -vx
        if y < lo or y > hi:
            y = np.clip(y, lo, hi) * 2 - y
            vy = -vy
    return positions


# ----------------------------------------------------------------- rendu

def render_frames(cfg: Config, positions: np.ndarray, bar_x: float | None = None,
                  with_bar: bool = True) -> np.ndarray:
    """Rendu (T, 1, S, S) float32 : disque blanc anti-aliasé, barre par-dessus."""
    S = cfg.img_size
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    dist = np.sqrt((xx[None] - positions[:, 0, None, None]) ** 2
                   + (yy[None] - positions[:, 1, None, None]) ** 2)
    frames = np.clip(cfg.disk_radius + 0.5 - dist, 0.0, 1.0)  # (T, S, S)
    if with_bar:
        assert bar_x is not None, "bar_x requis pour rendre la barre"
        x0 = int(round(bar_x))
        x1 = int(round(bar_x + cfg.bar_width))
        frames[:, :, x0:x1] = cfg.bar_color
    return frames[:, None]


def occlusion_flags(cfg: Config, positions: np.ndarray,
                    bar_x: float) -> np.ndarray:
    """(T,) bool : True quand le disque est totalement derrière la barre."""
    x = positions[:, 0]
    return (x >= bar_x + cfg.disk_radius) & \
           (x <= bar_x + cfg.bar_width - cfg.disk_radius)


# ----------------------------------------------------------------- dataset

def make_sequence(cfg: Config, rng: np.random.Generator) -> dict:
    bar_x = sample_bar_x(cfg, rng)
    ensure_crossing = rng.random() < cfg.crossing_frac
    positions = simulate_trajectory(cfg, rng, ensure_crossing, bar_x)
    return {
        "frames": torch.from_numpy(render_frames(cfg, positions, bar_x)),
        "positions": torch.from_numpy(positions),
        "occluded": torch.from_numpy(occlusion_flags(cfg, positions, bar_x)),
        "bar_x": torch.tensor(bar_x, dtype=torch.float32),
    }


class BouncingDiskDataset(IterableDataset):
    """Dataset infini, généré à la volée, seed distincte par worker."""

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = np.random.default_rng(self.seed + 1000 * worker_id)
        while True:
            yield make_sequence(self.cfg, rng)


def make_loader(cfg: Config) -> DataLoader:
    return DataLoader(
        BouncingDiskDataset(cfg, seed=cfg.seed),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )


def sample_batch(cfg: Config, n: int, seed: int = 1234) -> dict:
    """Batch d'éval reproductible (sans DataLoader), sur CPU."""
    rng = np.random.default_rng(seed)
    seqs = [make_sequence(cfg, rng) for _ in range(n)]
    return {k: torch.stack([s[k] for s in seqs]) for k in seqs[0]}
