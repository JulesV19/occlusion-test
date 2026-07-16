from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Données ---
    img_size: int = 64
    disk_radius: float = 5.0
    bar_width: float = 15.0          # occlusion totale ≈ (bar_width - 2r) / |vx| ≈ 3 frames
    bar_color: float = 0.5
    bar_margin: float = 12.0         # marge min barre-bords ; position tirée par épisode
    speed_x_range: tuple = (2.5, 4.5)  # |vx| en px/frame
    speed_y_range: tuple = (0.5, 3.0)  # |vy| en px/frame
    context_len: int = 4             # C frames de contexte
    horizon: int = 8                 # H pas de rollout
    crossing_frac: float = 0.5       # fraction de séquences garanties avec traversée de barre

    # --- Modèles ---
    embed_dim: int = 128
    enc_channels: tuple = (32, 64, 128, 256)
    gru_hidden: int = 256
    gru_layers: int = 1

    # --- Loss (coefficients canoniques VICReg) ---
    lambda_inv: float = 25.0
    lambda_var: float = 25.0
    lambda_cov: float = 1.0

    # --- Entraînement ---
    batch_size: int = 128
    epochs: int = 60
    steps_per_epoch: int = 500       # dataset infini : une "epoch" = ce nb de steps
    val_seqs: int = 256              # séquences de validation figées (seed fixe)
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1_000
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0
    num_workers: int = 2
    log_every: int = 100
    seed: int = 0

    # --- Probes (post-hoc) ---
    probe_steps: int = 3_000
    probe_lr: float = 1e-3
    probe_batch_size: int = 128

    @property
    def steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def seq_len(self) -> int:
        return self.context_len + self.horizon

    @property
    def bar_x_range(self) -> tuple:
        """Plage du bord gauche de la barre (tirée uniformément par épisode)."""
        return (self.bar_margin, self.img_size - self.bar_width - self.bar_margin)
