#!/usr/bin/env python3
"""
seir_viewer_2.py — Visualiseur SEIR style spaghetti (référence prof)
730 jours, plusieurs seeds superposés, animation de la simulation.

Usage :
    # Générer les runs multi-seeds d'abord :
    for seed in $(seq 42 51); do ./seir_multiseed $seed; done

    # Lancer le viewer animé (seed principale + spaghetti)
    python3 seir_viewer_2.py

    # Avec fichiers personnalisés
    python3 seir_viewer_2.py --main counts_seed_42.csv \
        --frames frames_seed_42.bin \
        --seeds counts_seed_43.csv counts_seed_44.csv ...

    # Sauvegarder en MP4
    python3 seir_viewer_2.py --save anim_730.mp4
"""

import argparse
import struct
import glob
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap

# ── Thème identique au viewer principal ─────────────────────────────────────
BG        = "#0b0c10"
BG_PANEL  = "#13141a"
BG_CARD   = "#1a1b24"
COL_GRID  = "#252636"
COL_TEXT  = "#d4d6e0"
COL_MUTED = "#6b6e82"
COL_ACCENT= "#4e9af1"

SEIR_RGBA = np.array([
    [0.04, 0.04, 0.07, 1.00],
    [0.20, 0.47, 0.82, 1.00],
    [0.94, 0.73, 0.08, 1.00],
    [0.86, 0.12, 0.12, 1.00],
    [0.10, 0.72, 0.30, 1.00],
])
CMAP_SEIR = ListedColormap(SEIR_RGBA)

SEIR_COLORS = {
    "S": "#3878c8",
    "E": "#f0bb14",
    "I": "#dc1f1f",
    "R": "#1ab84c",
}

matplotlib.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG_PANEL,
    "axes.edgecolor":    COL_GRID,
    "axes.labelcolor":   COL_MUTED,
    "axes.titlecolor":   COL_TEXT,
    "xtick.color":       COL_MUTED,
    "ytick.color":       COL_MUTED,
    "text.color":        COL_TEXT,
    "legend.facecolor":  BG_CARD,
    "legend.edgecolor":  COL_GRID,
    "legend.labelcolor": COL_TEXT,
    "grid.color":        COL_GRID,
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         8.5,
})

FRAME_EVERY = 5

# ── I/O ──────────────────────────────────────────────────────────────────────

def load_frames(path):
    with open(path, "rb") as f:
        W, H = struct.unpack("ii", f.read(8))
        raw  = f.read()
    n = len(raw) // (W * H)
    return W, H, np.frombuffer(raw[:n*W*H], dtype=np.uint8).reshape(n, H, W)

def load_csv(path):
    df = pd.read_csv(path)
    N  = df[["S","E","I","R"]].iloc[0].sum()
    df["N"] = N
    return df

# ── Figure ───────────────────────────────────────────────────────────────────

class SEIRViewer2:
    """
    Layout :
      ┌─────────────────┬────────────────────────────┐
      │  Grille animée  │  S / E / I / R  spaghetti  │
      │   300×300       │  (plusieurs seeds)          │
      └─────────────────┴────────────────────────────┘
      │              Contrôles                        │
    """

    def __init__(self, main_csv, frames_path, seed_csvs):
        self.df_main  = load_csv(main_csv)
        self.N        = int(self.df_main["N"].iloc[0])
        self.W, self.H, self.frames = load_frames(frames_path)
        self.nf       = len(self.frames)
        self.df_seeds = [load_csv(f) for f in seed_csvs]
        self.fi       = 0
        self.playing  = True
        self.speed    = 1
        self._build()
        self._static()
        self._update(0)

    def _build(self):
        self.fig = plt.figure(figsize=(16, 8.5), facecolor=BG)
        self.fig.canvas.manager.set_window_title(
            "SEIR Viewer 2 — 730 jours — Style référence prof")

        self.fig.text(0.5, 0.975,
            "Simulation SEIR multi-agents — 730 jours (2 ans)",
            ha="center", va="top", fontsize=12, fontweight="bold",
            color=COL_TEXT, family="monospace")
        self.fig.text(0.5, 0.955,
            f"Grille {self.W}×{self.H} · {self.N:,} agents · "
            f"{len(self.df_seeds)+1} runs superposés · voisinage Moore 9 cellules",
            ha="center", va="top", fontsize=8, color=COL_MUTED, family="monospace")

        outer = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[1, 0.065],
            hspace=0.07,
            left=0.03, right=0.98, top=0.94, bottom=0.04,
        )
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0],
            width_ratios=[1.1, 1], wspace=0.08,
        )

        # ── Grille spatiale ──
        self.ax_grid = self.fig.add_subplot(inner[0])
        self.ax_grid.set_facecolor(BG)
        self.ax_grid.set_title("Distribution spatiale des agents",
                               fontsize=9, pad=5, color=COL_TEXT)
        self.ax_grid.set_xlabel("x  (cellule)", fontsize=8)
        self.ax_grid.set_ylabel("y  (cellule)", fontsize=8)
        self.ax_grid.tick_params(labelsize=7)

        # ── 4 panneaux spaghetti ──
        right = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=inner[1],
            hspace=0.50, wspace=0.38,
        )
        self.ax = {}
        labels  = {"S": "Susceptibles (S)", "E": "Exposés (E)",
                   "I": "Infectieux (I)",   "R": "Rétablis (R)"}
        pos     = {"S": (0,0), "E": (0,1), "I": (1,0), "R": (1,1)}
        for key, (r, c) in pos.items():
            ax = self.fig.add_subplot(right[r, c])
            ax.set_title(labels[key], fontsize=8.5, pad=3, color=COL_TEXT)
            ax.set_xlabel("Jours", fontsize=7.5)
            ax.set_ylabel("Individus", fontsize=7.5)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, int(self.df_main["step"].max()))
            ax.set_ylim(0, self.N * 1.06)
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            ax.tick_params(labelsize=7)
            self.ax[key] = ax

        # ── Zone de contrôles ──
        self.ax_ctrl = self.fig.add_subplot(outer[1])
        self.ax_ctrl.set_facecolor(BG)
        self.ax_ctrl.axis("off")

    def _static(self):
        # Image grille
        self.im = self.ax_grid.imshow(
            self.frames[0], cmap=CMAP_SEIR, vmin=0, vmax=4,
            interpolation="nearest", aspect="equal", origin="upper",
        )
        handles = [
            mpatches.Patch(facecolor=SEIR_RGBA[i+1], edgecolor="none",
                           label=l)
            for i, l in enumerate(["Susceptible (S)", "Expose (E)",
                                    "Infectieux (I)", "Retabli (R)"])
        ]
        self.ax_grid.legend(handles=handles, loc="lower right",
                            fontsize=7, framealpha=0.88)
        self.lbl_day = self.ax_grid.text(
            0.015, 0.985, "Jour   0",
            transform=self.ax_grid.transAxes, fontsize=11,
            fontweight="bold", color="white", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=BG_CARD,
                      edgecolor=COL_GRID, alpha=0.88),
        )

        # Spaghetti : tous les seeds en fond (alpha faible)
        self.spaghetti = {key: [] for key in "SEIR"}
        for df in self.df_seeds:
            for key in "SEIR":
                ln, = self.ax[key].plot(
                    [], [], lw=0.9, color=SEIR_COLORS[key],
                    alpha=0.30, zorder=1,
                )
                self.spaghetti[key].append(ln)

        # Courbe principale (seed 42) en avant-plan
        self.main_lines = {}
        for key in "SEIR":
            ln, = self.ax[key].plot(
                [], [], lw=2.0, color=SEIR_COLORS[key],
                alpha=1.0, zorder=5, label="seed 42",
            )
            self.main_lines[key] = ln

        # Curseurs verticaux
        self.cursors = {}
        for key in "SEIR":
            cur = self.ax[key].axvline(
                0, color="#ffffff", lw=0.8, ls="--", alpha=0.5, zorder=10)
            self.cursors[key] = cur

        # Annotation pic I
        peak_day = int(self.df_main.loc[self.df_main["I"].idxmax(), "step"])
        peak_I   = int(self.df_main["I"].max())
        self.ax["I"].axvline(peak_day, color=SEIR_COLORS["I"],
                             lw=0.8, ls=":", alpha=0.7)
        self.ax["I"].annotate(
            f"Pic J{peak_day}\n{peak_I:,}",
            xy=(peak_day, peak_I),
            xytext=(peak_day + max(self.df_main["step"].max()*0.05, 10),
                    peak_I * 0.85),
            fontsize=7, color=SEIR_COLORS["I"],
            arrowprops=dict(arrowstyle="->", color=SEIR_COLORS["I"],
                            lw=0.7, connectionstyle="arc3,rad=-0.2"),
        )

        # Légende seeds
        from matplotlib.lines import Line2D
        leg_handles = [
            Line2D([0],[0], color=SEIR_COLORS["S"], lw=2, label="seed 42 (principal)"),
            Line2D([0],[0], color=SEIR_COLORS["S"], lw=0.9, alpha=0.35,
                   label=f"autres seeds ({len(self.df_seeds)})"),
        ]
        self.ax["S"].legend(handles=leg_handles, fontsize=6.5,
                            loc="upper right", framealpha=0.8)

        # Contrôles bas
        self.fig.text(
            0.03, 0.022,
            "Espace : lecture/pause  |  [<] [>] : frame  |  R : debut  |  haut/bas : vitesse",
            fontsize=7, color=COL_MUTED,
        )
        self.fig.text(0.98, 0.022,
            "SEIR Viewer 2 — 730j — M1 CHPS",
            fontsize=7, color=COL_MUTED, ha="right")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _update(self, fi):
        fi   = int(np.clip(fi, 0, self.nf - 1))
        self.fi = fi
        step = fi * FRAME_EVERY

        # Grille
        self.im.set_data(self.frames[fi])
        self.lbl_day.set_text(f"Jour {step:4d}")

        # Sous-ensemble du CSV jusqu'au step courant
        sub_main = self.df_main[self.df_main["step"] <= step]

        # Courbe principale
        for key in "SEIR":
            self.main_lines[key].set_data(
                sub_main["step"].values, sub_main[key].values)
            self.cursors[key].set_xdata([step, step])

        # Spaghetti seeds
        for i, df in enumerate(self.df_seeds):
            sub = df[df["step"] <= step]
            for key in "SEIR":
                self.spaghetti[key][i].set_data(
                    sub["step"].values, sub[key].values)

        self.fig.canvas.draw_idle()

    def _animate_step(self, _):
        if self.playing:
            self.fi = (self.fi + self.speed) % self.nf
        self._update(self.fi)

    def _on_key(self, event):
        if event.key == " ":
            self.playing = not self.playing
        elif event.key == "right":
            self._update(min(self.fi + 1, self.nf - 1))
        elif event.key == "left":
            self._update(max(self.fi - 1, 0))
        elif event.key in ("r", "R"):
            self.fi = 0; self._update(0)
        elif event.key == "up":
            self.speed = min(self.speed + 1, 8)
        elif event.key == "down":
            self.speed = max(self.speed - 1, 1)

    def run(self, interval_ms=80):
        self.anim = animation.FuncAnimation(
            self.fig, self._animate_step,
            frames=None, interval=interval_ms, cache_frame_data=False,
        )
        plt.show()

    def save(self, output, fps=20):
        print(f"[Export] → {output}  ({self.nf} frames, {fps} fps)")
        def _f(fi): self._update(fi)
        ani = animation.FuncAnimation(
            self.fig, _f, frames=self.nf,
            interval=50, cache_frame_data=False,
        )
        writer = animation.FFMpegWriter(
            fps=fps, bitrate=2400,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "fast"],
        )
        ani.save(output, writer=writer, dpi=130,
                 savefig_kwargs={"facecolor": BG})
        print(f"[Export] Termine → {output}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SEIR Viewer 2 — spaghetti 730 jours",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--main",   default="counts_seed_42.csv",
                        help="CSV de la simulation principale (seed 42)")
    parser.add_argument("--frames", default="frames_seed_42.bin",
                        help="fichier bin de la simulation principale")
    parser.add_argument("--seeds",  nargs="*", default=None,
                        help="autres CSV à superposer (spaghetti)")
    parser.add_argument("--save",   default=None, metavar="OUT.mp4",
                        help="sauvegarder en MP4 (nécessite ffmpeg)")
    parser.add_argument("--fps",    default=20, type=int)
    parser.add_argument("--speed",  default=80, type=int,
                        help="intervalle entre frames en ms (défaut 80)")
    args = parser.parse_args()

    # Auto-détection des seeds si non spécifiées
    seed_csvs = args.seeds
    if seed_csvs is None:
        import glob as g
        all_seeds = sorted(g.glob("counts_seed_*.csv"))
        seed_csvs = [f for f in all_seeds if f != args.main]
        if not seed_csvs:
            # Fallback : utiliser le CSV principal comme seule courbe
            seed_csvs = []
        print(f"[Viewer2] {len(seed_csvs)} seeds supplémentaires détectés")

    for path in (args.main, args.frames):
        if not os.path.exists(path):
            print(f"[Erreur] Fichier introuvable : {path}", file=sys.stderr)
            print("Générez d'abord les données :", file=sys.stderr)
            print("  for seed in $(seq 42 51); do ./seir_multiseed $seed; done",
                  file=sys.stderr)
            sys.exit(1)

    viewer = SEIRViewer2(args.main, args.frames, seed_csvs)

    if args.save:
        viewer.save(args.save, fps=args.fps)
    else:
        viewer.run(interval_ms=args.speed)

if __name__ == "__main__":
    main()
