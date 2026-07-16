"""Visualiseur local de la qualité des données synthétiques.

Usage :
    python scripts/data_viewer.py                 # viewer interactif animé
    python scripts/data_viewer.py --stats         # histogrammes de calibration
    python scripts/data_viewer.py --gif seq.gif   # exporte une séquence en GIF

Touches (mode interactif) :
    espace  pause / lecture
    ←  →    frame par frame (met en pause)
    n       nouvelle séquence
    q       quitter
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from occlusion_jepa import Config
from occlusion_jepa.data import (make_sequence, occlusion_flags, sample_bar_x,
                                 simulate_trajectory)


# ------------------------------------------------------------------ viewer

class SequenceViewer:
    def __init__(self, cfg: Config, seed: int, interval_ms: int = 120):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.paused = False
        self.t = 0
        self._new_sequence(draw=False)

        self.fig, (self.ax_img, self.ax_traj) = plt.subplots(
            1, 2, figsize=(9, 4.5), width_ratios=[1, 1])
        self.fig.canvas.manager.set_window_title("Occlusion JEPA — data viewer")

        # panneau frame
        self.im = self.ax_img.imshow(self.frames[0], cmap="gray", vmin=0, vmax=1)
        self.ax_img.set_xticks([]), self.ax_img.set_yticks([])

        # panneau trajectoire (repère image : y vers le bas)
        S = cfg.img_size
        self.bar_rect = Rectangle(
            (self.bar_x, 0), cfg.bar_width, S, color="gray", alpha=0.35,
            label="barre")
        self.ax_traj.add_patch(self.bar_rect)
        self.traj_line, = self.ax_traj.plot([], [], "-", color="royalblue", lw=1.5)
        self.traj_dot, = self.ax_traj.plot([], [], "o", color="royalblue", ms=8)
        self.ax_traj.set_xlim(0, S), self.ax_traj.set_ylim(S, 0)
        self.ax_traj.set_aspect("equal")
        self.ax_traj.set_title("trajectoire ground-truth", fontsize=9)
        self.ax_traj.legend(loc="upper right", fontsize=7)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.anim = animation.FuncAnimation(
            self.fig, self._tick, interval=interval_ms, cache_frame_data=False)
        self.fig.tight_layout()

    def _new_sequence(self, draw: bool = True):
        seq = make_sequence(self.cfg, self.rng)
        self.frames = seq["frames"][:, 0].numpy()
        self.positions = seq["positions"].numpy()
        self.occ = seq["occluded"].numpy()
        self.bar_x = seq["bar_x"].item()
        self.t = 0
        if draw:
            self.bar_rect.set_x(self.bar_x)
            self._draw()

    def _draw(self):
        t = self.t
        self.im.set_data(self.frames[t])
        color = "red" if self.occ[t] else "black"
        for s in self.ax_img.spines.values():
            s.set_edgecolor(color), s.set_linewidth(3 if self.occ[t] else 1)
        x, y = self.positions[t]
        n_occ = int(self.occ.sum())
        self.ax_img.set_title(
            f"t={t}/{len(self.frames) - 1}   pos=({x:.1f}, {y:.1f})"
            f"   {'OCCLUS' if self.occ[t] else 'visible'}"
            f"   ({n_occ} frames occluses/seq)",
            fontsize=9, color=color)
        self.traj_line.set_data(self.positions[:t + 1, 0], self.positions[:t + 1, 1])
        self.traj_dot.set_data([x], [y])
        self.traj_dot.set_color("red" if self.occ[t] else "royalblue")
        self.fig.canvas.draw_idle()

    def _tick(self, _):
        if not self.paused:
            self.t = (self.t + 1) % len(self.frames)
            self._draw()

    def _on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "n":
            self._new_sequence()
        elif event.key in ("left", "right"):
            self.paused = True
            step = 1 if event.key == "right" else -1
            self.t = (self.t + step) % len(self.frames)
            self._draw()
        elif event.key == "q":
            plt.close(self.fig)


# ------------------------------------------------------------------ stats

def occlusion_durations(cfg: Config, n_seqs: int, seed: int) -> np.ndarray:
    """Durées des épisodes d'occlusion COMPLETS sur des trajectoires longues
    non forcées (crossing naturel) — mesure la calibration géométrie/vitesse
    sans l'artefact de plafonnement de vx du sampling forcé."""
    long_cfg = replace(cfg, context_len=0, horizon=200, crossing_frac=0.0)
    rng = np.random.default_rng(seed)
    durations = []
    for _ in range(n_seqs):
        bar_x = sample_bar_x(long_cfg, rng)
        pos = simulate_trajectory(long_cfg, rng, ensure_crossing=False, bar_x=bar_x)
        occ = occlusion_flags(long_cfg, pos, bar_x).astype(int)
        d = np.diff(np.concatenate([[0], occ, [0]]))
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if s > 0 and e < len(occ):  # épisodes complets uniquement
                durations.append(e - s)
    return np.array(durations)


def show_stats(cfg: Config, seed: int, n_seqs: int = 200):
    durations = occlusion_durations(cfg, n_seqs, seed)

    # côté training : fraction de séquences (T = C + H) contenant de l'occlusion
    rng = np.random.default_rng(seed + 1)
    seqs = [make_sequence(cfg, rng) for _ in range(500)]
    frac_occ_seq = np.mean([s["occluded"].any().item() for s in seqs])
    frac_occ_horizon = np.mean(
        [s["occluded"][cfg.context_len:].any().item() for s in seqs])
    occ_frames = np.concatenate([s["occluded"].numpy() for s in seqs])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].hist(durations, bins=np.arange(0.5, durations.max() + 1.5),
                 color="royalblue", edgecolor="white")
    axes[0].axvline(durations.mean(), color="red", ls="--",
                    label=f"moyenne = {durations.mean():.2f}")
    axes[0].set_title("Durée d'occlusion totale (crossings naturels)", fontsize=9)
    axes[0].set_xlabel("frames"), axes[0].legend(fontsize=8)

    vx = np.random.default_rng(seed).uniform(*cfg.speed_x_range, 5000)
    theo = (cfg.bar_width - 2 * cfg.disk_radius) / vx
    axes[1].hist(theo, bins=30, color="seagreen", edgecolor="white")
    axes[1].set_title("Durée théorique (bar_w − 2r)/|vx|", fontsize=9)
    axes[1].set_xlabel("frames")

    axes[2].bar(["seq avec\nocclusion", "occlusion dans\nl'horizon",
                 "frames\noccluses"],
                [frac_occ_seq, frac_occ_horizon, occ_frames.mean()],
                color=["royalblue", "orange", "gray"])
    axes[2].set_ylim(0, 1)
    axes[2].set_title(f"Fractions (config training, T={cfg.seq_len})", fontsize=9)
    for i, v in enumerate([frac_occ_seq, frac_occ_horizon, occ_frames.mean()]):
        axes[2].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)

    print(f"durées d'occlusion (naturelles) : moyenne {durations.mean():.2f}, "
          f"médiane {np.median(durations):.0f}, "
          f"min/max {durations.min()}/{durations.max()} frames (n={len(durations)})")
    print(f"séquences training avec occlusion : {frac_occ_seq:.0%} "
          f"(dans l'horizon : {frac_occ_horizon:.0%})")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ gif

def export_gif(cfg: Config, seed: int, path: str, interval_ms: int = 120):
    rng = np.random.default_rng(seed)
    seq = make_sequence(cfg, rng)
    frames, occ = seq["frames"][:, 0].numpy(), seq["occluded"].numpy()

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    im = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]), ax.set_yticks([])

    def update(t):
        im.set_data(frames[t])
        color = "red" if occ[t] else "black"
        for s in ax.spines.values():
            s.set_edgecolor(color), s.set_linewidth(3 if occ[t] else 1)
        ax.set_title(f"t={t}" + ("  OCCLUS" if occ[t] else ""), color=color,
                     fontsize=9)
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=interval_ms)
    anim.save(path, writer="pillow")
    plt.close(fig)
    print(f"GIF -> {path}")


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true",
                        help="histogrammes de calibration au lieu du viewer")
    parser.add_argument("--gif", type=str, default=None,
                        help="exporte une séquence en GIF vers ce chemin")
    parser.add_argument("--frames", type=int, default=40,
                        help="frames par séquence dans le viewer (défaut 40)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config()
    if args.stats:
        show_stats(cfg, seed=args.seed)
        plt.show()
    elif args.gif:
        view_cfg = replace(cfg, horizon=args.frames - cfg.context_len)
        export_gif(view_cfg, args.seed, args.gif)
    else:
        # séquences plus longues que le training pour voir plusieurs rebonds
        view_cfg = replace(cfg, horizon=args.frames - cfg.context_len)
        viewer = SequenceViewer(view_cfg, seed=args.seed)
        print(__doc__)
        plt.show()


if __name__ == "__main__":
    main()
