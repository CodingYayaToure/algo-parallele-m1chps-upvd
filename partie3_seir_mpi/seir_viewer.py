#!/usr/bin/env python3
"""
seir_viewer.py — Visualiseur scientifique interactif pour la simulation SEIR
Auteur : M1 CHPS — Algorithmique Parallèle

Usage :
    python3 seir_viewer.py                              # données par défaut
    python3 seir_viewer.py --bin frames_seq.bin --csv counts_seq.csv
    python3 seir_viewer.py --bin frames_seq.bin --csv counts_seq.csv --save anim.mp4
    python3 seir_viewer.py --compare counts_seq.csv counts_mpi_2.csv \
                           --labels "Séquentiel" "MPI P=2"
"""

import argparse
import struct
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
from matplotlib.widgets import Button, Slider

# ── Thème global ────────────────────────────────────────────────────────────
BG        = "#0b0c10"
BG_PANEL  = "#13141a"
BG_CARD   = "#1a1b24"
COL_GRID  = "#252636"
COL_TEXT  = "#d4d6e0"
COL_MUTED = "#6b6e82"
COL_ACCENT= "#4e9af1"

# Palette SEIR  (0=vide, 1=S, 2=E, 3=I, 4=R)
SEIR_RGBA = np.array([
    [0.04, 0.04, 0.07, 1.00],   # 0  vide   — noir profond
    [0.20, 0.47, 0.82, 1.00],   # 1  S      — bleu acier
    [0.94, 0.73, 0.08, 1.00],   # 2  E      — ambre
    [0.86, 0.12, 0.12, 1.00],   # 3  I      — rouge vif
    [0.10, 0.72, 0.30, 1.00],   # 4  R      — vert émeraude
])
CMAP_SEIR   = ListedColormap(SEIR_RGBA)
SEIR_LABELS = ["Vide", "Susceptible (S)", "Exposé (E)", "Infectieux (I)", "Rétabli (R)"]
SEIR_KEYS   = ["S", "E", "I", "R"]
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

# ── I/O ────────────────────────────────────────────────────────────────────

def load_frames(path: str):
    """Charge un fichier binaire de frames. Retourne (W, H, array [N,H,W])."""
    with open(path, "rb") as f:
        W, H = struct.unpack("ii", f.read(8))
        raw  = f.read()
    n  = len(raw) // (W * H)
    return W, H, np.frombuffer(raw[: n * W * H], dtype=np.uint8).reshape(n, H, W)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    N  = df[["S", "E", "I", "R"]].iloc[0].sum()
    df["N"]          = N
    df["attack_rate"] = (N - df["S"]) / N * 100.0
    df["incidence"]   = df["I"].diff().clip(lower=0).fillna(0).astype(int)
    return df


# ── Métriques épidémiologiques ─────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, step: int):
    """Retourne un dict de métriques pour le step courant."""
    row   = df[df["step"] <= step].iloc[-1]
    N     = int(row["N"])
    S, E, I, R = int(row["S"]), int(row["E"]), int(row["I"]), int(row["R"])

    peak_idx = df["I"].idxmax()
    peak_step= int(df.loc[peak_idx, "step"])
    peak_I   = int(df.loc[peak_idx, "I"])

    # Taux d'attaque cumulé
    attack   = (N - S) / N * 100.0

    # Détermination de la phase
    if step < peak_step and I > 0:
        if step < peak_step * 0.5:
            phase, phase_col = "▲ Croissance exponentielle", "#f0bb14"
        else:
            phase, phase_col = "▲ Approche du pic",          "#f07014"
    elif step == peak_step:
        phase, phase_col = "◆ Pic épidémique",               "#dc1f1f"
    elif I < N * 0.001:
        phase, phase_col = "● Phase endémique",              "#1ab84c"
    else:
        phase, phase_col = "▼ Déclin",                       "#3878c8"

    # Estimation Reff très approximative (sliding window 7j)
    if step >= 7:
        sub   = df[(df["step"] >= step - 7) & (df["step"] <= step)]
        dS    = float(sub["S"].iloc[0] - sub["S"].iloc[-1])
        Smid  = float(sub["S"].mean())
        Imid  = float(sub["I"].mean()) + 1e-9
        reff  = (dS / 7.0) / (Imid * Smid / N + 1e-9) * 7.0
        reff  = min(max(reff, 0.0), 10.0)
    else:
        reff = 0.0

    return dict(S=S, E=E, I=I, R=R, N=N,
                attack=attack, phase=phase, phase_col=phase_col,
                peak_step=peak_step, peak_I=peak_I, reff=reff, step=step)


# ── Construction de la figure ───────────────────────────────────────────────

class SEIRViewer:
    """
    Interface scientifique complète :
      - Grille spatiale animée (panneau gauche, grand)
      - Série temporelle SEIR + incidence (panneau droit haut)
      - Métriques épidémiologiques en temps réel (panneau droit bas)
      - Barre de contrôle : Play/Pause, vitesse, curseur de step
    """

    FRAME_EVERY = 5    # doit correspondre à la constante FRAME_EVERY du code C

    def __init__(self, frames_path: str, csv_path: str):
        self.W, self.H, self.frames = load_frames(frames_path)
        self.df  = load_csv(csv_path)
        self.N   = int(self.df["N"].iloc[0])
        self.nf  = len(self.frames)
        self.fi  = 0        # index frame courant
        self.playing = True
        self.speed   = 1    # frames/update (1=lent, 4=rapide)
        self._build_figure()
        self._build_controls()
        self._draw_static_elements()
        self._update(0)

    # ── Layout ──────────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 8.5), facecolor=BG)
        self.fig.canvas.manager.set_window_title(
            "SEIR Multi-Agent Viewer — M1 CHPS Algorithmique Parallèle"
        )

        # Titre principal
        self.fig.text(
            0.5, 0.975,
            "Simulation SEIR — propagation épidémique multi-agents",
            ha="center", va="top", fontsize=12, fontweight="bold",
            color=COL_TEXT, family="monospace",
        )
        self.fig.text(
            0.5, 0.955,
            f"Grille {self.W}×{self.H} — {self.N:,} agents — modèle SEIR stochastique",
            ha="center", va="top", fontsize=8, color=COL_MUTED, family="monospace",
        )

        # GridSpec principal : [grille | droit] / contrôles
        outer = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[1, 0.07],
            hspace=0.08,
            left=0.03, right=0.98, top=0.94, bottom=0.04,
        )
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0],
            width_ratios=[1.15, 1],
            wspace=0.08,
        )

        # Panneau gauche — grille spatiale
        self.ax_grid = self.fig.add_subplot(inner[0])
        self.ax_grid.set_facecolor(BG)
        self.ax_grid.set_title("Distribution spatiale des agents",
                               fontsize=9, pad=5, color=COL_TEXT)
        self.ax_grid.set_xlabel("x  (cellule)", fontsize=8)
        self.ax_grid.set_ylabel("y  (cellule)", fontsize=8)
        self.ax_grid.tick_params(labelsize=7)

        # Panneau droit — sous-division verticale
        right = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=inner[1],
            height_ratios=[2.2, 1.0, 1.3],
            hspace=0.45,
        )

        # Série temporelle SEIR
        self.ax_ts = self.fig.add_subplot(right[0])
        self.ax_ts.set_title("Dynamique SEIR", fontsize=9, pad=4)
        self.ax_ts.set_xlabel("Jours", fontsize=8)
        self.ax_ts.set_ylabel("Individus", fontsize=8)
        self.ax_ts.set_xlim(0, self.df["step"].max())
        self.ax_ts.set_ylim(0, self.N * 1.06)
        self.ax_ts.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        self.ax_ts.grid(True, alpha=0.35)

        # Incidence journalière
        self.ax_inc = self.fig.add_subplot(right[1])
        self.ax_inc.set_title("Nouveaux cas infectieux / jour", fontsize=8.5, pad=4)
        self.ax_inc.set_xlabel("Jours", fontsize=8)
        self.ax_inc.set_ylabel("ΔI", fontsize=8)
        self.ax_inc.set_xlim(0, self.df["step"].max())
        self.ax_inc.grid(True, alpha=0.35)

        # Tableau de bord métriques
        self.ax_dash = self.fig.add_subplot(right[2])
        self.ax_dash.set_facecolor(BG_CARD)
        self.ax_dash.axis("off")

        # Zone de contrôles (bas)
        self.ax_ctrl = self.fig.add_subplot(outer[1])
        self.ax_ctrl.set_facecolor(BG)
        self.ax_ctrl.axis("off")

    # ── Éléments statiques ──────────────────────────────────────────────────

    def _draw_static_elements(self):
        ax = self.ax_grid

        # Image de fond
        self.im = ax.imshow(
            self.frames[0], cmap=CMAP_SEIR, vmin=0, vmax=4,
            interpolation="nearest", aspect="equal", origin="upper",
        )

        # Légende colorée
        handles = [
            mpatches.Patch(facecolor=SEIR_RGBA[i+1], edgecolor="none",
                           label=SEIR_LABELS[i+1])
            for i in range(4)
        ]
        ax.legend(
            handles=handles, loc="lower right", fontsize=7.5,
            framealpha=0.88, borderpad=0.6, labelspacing=0.35,
        )

        # Étiquette de jour
        self.lbl_day = ax.text(
            0.015, 0.985, "Jour 0",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            color="white", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=BG_CARD,
                      edgecolor=COL_GRID, alpha=0.88),
        )

        # ── Séries temporelles ──
        ax_ts = self.ax_ts
        steps = self.df["step"].values
        self.ts_lines = {}
        for key in SEIR_KEYS:
            line, = ax_ts.plot([], [], lw=1.6, color=SEIR_COLORS[key], label=key)
            self.ts_lines[key] = line

        # Remplissage sous I
        self.ts_fill = ax_ts.fill_between([], [], alpha=0.18, color=SEIR_COLORS["I"])

        # Ligne curseur temporelle
        self.ts_cursor = ax_ts.axvline(0, color="#ffffff", lw=0.9,
                                       ls="--", alpha=0.6, zorder=10)
        ax_ts.legend(loc="upper right", fontsize=8, ncol=2,
                     framealpha=0.85, borderpad=0.5)

        # Annotation pic
        peak_step = int(self.df["I"].idxmax())
        peak_I    = int(self.df["I"].max())
        peak_day  = int(self.df.loc[self.df["I"].idxmax(), "step"])
        ax_ts.axvline(peak_day, color=SEIR_COLORS["I"], lw=0.8,
                      ls=":", alpha=0.7, zorder=5)
        ax_ts.annotate(
            f"Pic J{peak_day}\n{peak_I:,}",
            xy=(peak_day, peak_I),
            xytext=(peak_day + max(self.df["step"].max() * 0.05, 5), peak_I * 0.92),
            fontsize=7, color=SEIR_COLORS["I"],
            arrowprops=dict(arrowstyle="->", color=SEIR_COLORS["I"],
                            lw=0.8, connectionstyle="arc3,rad=-0.2"),
        )

        # ── Incidence ──
        ax_inc = self.ax_inc
        self.inc_bars = ax_inc.bar(
            self.df["step"], self.df["incidence"],
            width=1.0, color=SEIR_COLORS["I"], alpha=0.55,
            linewidth=0, zorder=2,
        )
        self.inc_cursor = ax_inc.axvline(0, color="#ffffff", lw=0.9,
                                         ls="--", alpha=0.6, zorder=10)
        ax_inc.set_ylim(0, self.df["incidence"].max() * 1.15)

        # ── Dashboard ──
        self._draw_dashboard_frame()

    def _draw_dashboard_frame(self):
        """Dessine le cadre du tableau de bord (éléments qui ne changent pas)."""
        ax = self.ax_dash
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        # Titre du tableau
        ax.text(0.5, 0.97, "MÉTRIQUES ÉPIDÉMIOLOGIQUES",
                ha="center", va="top", fontsize=8, fontweight="bold",
                color=COL_ACCENT, transform=ax.transAxes)

        # Séparateur
        ax.axhline(0.89, color=COL_GRID, lw=0.8, xmin=0.02, xmax=0.98)

        # Libellés fixes (les valeurs seront des Text mobiles)
        labels_left = [
            (0.03, 0.81, "Susceptibles (S)"),
            (0.03, 0.67, "Exposés      (E)"),
            (0.03, 0.53, "Infectieux   (I)"),
            (0.03, 0.39, "Rétablis     (R)"),
        ]
        for x, y, txt in labels_left:
            ax.text(x, y, txt, fontsize=8, color=COL_MUTED,
                    transform=ax.transAxes, va="center")

        # Séparateur central
        ax.axhline(0.28, color=COL_GRID, lw=0.8, xmin=0.02, xmax=0.98)

        labels_bot = [
            (0.03, 0.21, "Taux d'attaque"),
            (0.03, 0.12, "R effectif (estim.)"),
            (0.55, 0.21, "Pic infectieux"),
            (0.55, 0.12, "Phase"),
        ]
        for x, y, txt in labels_bot:
            ax.text(x, y, txt, fontsize=7.5, color=COL_MUTED,
                    transform=ax.transAxes, va="center")

        # Textes dynamiques — initialisés vides, mis à jour dans _update
        colors = [SEIR_COLORS[k] for k in SEIR_KEYS]
        self.dash_vals = {}
        ys    = [0.81, 0.67, 0.53, 0.39]
        for i, key in enumerate(SEIR_KEYS):
            self.dash_vals[key] = ax.text(
                0.55, ys[i], "—", fontsize=9, fontweight="bold",
                color=colors[i], transform=ax.transAxes, va="center", ha="left",
            )
            self.dash_vals[f"{key}_pct"] = ax.text(
                0.85, ys[i], "—", fontsize=7.5,
                color=COL_MUTED, transform=ax.transAxes, va="center", ha="left",
            )

        self.dash_vals["attack"] = ax.text(
            0.03, 0.12 - 0.01, "—", fontsize=9, fontweight="bold",
            color=COL_ACCENT, transform=ax.transAxes, va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, edgecolor="none"),
            zorder=5,
        )
        # On va surcharger attack à y=0.21
        self.dash_vals["attack"].set_position((0.03 + 0.32, 0.21))

        self.dash_vals["reff"] = ax.text(
            0.03 + 0.32, 0.12, "—", fontsize=9, fontweight="bold",
            color=COL_ACCENT, transform=ax.transAxes, va="center",
        )
        self.dash_vals["peak"] = ax.text(
            0.55 + 0.27, 0.21, "—", fontsize=8.5, fontweight="bold",
            color=SEIR_COLORS["I"], transform=ax.transAxes, va="center",
        )
        self.dash_vals["phase"] = ax.text(
            0.55 + 0.12, 0.12, "—", fontsize=7.5, fontweight="bold",
            color="#ffffff", transform=ax.transAxes, va="center",
        )

        # Barres de proportion SEIR (mini progress bars)
        self.dash_bars = {}
        bar_y_map = {"S": 0.78, "E": 0.64, "I": 0.50, "R": 0.36}
        for key, by in bar_y_map.items():
            # fond gris
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.55, by), 0.41, 0.05,
                boxstyle="round,pad=0.005",
                facecolor=COL_GRID, edgecolor="none",
                transform=ax.transAxes, zorder=1,
            ))
            # barre colorée (largeur dynamique)
            bar = mpatches.FancyBboxPatch(
                (0.55, by), 0.001, 0.05,
                boxstyle="round,pad=0.005",
                facecolor=SEIR_COLORS[key], edgecolor="none",
                transform=ax.transAxes, zorder=2, alpha=0.75,
            )
            ax.add_patch(bar)
            self.dash_bars[key] = bar

    # ── Contrôles interactifs ───────────────────────────────────────────────

    def _build_controls(self):
        fig = self.fig

        # Bouton Play/Pause
        ax_btn = fig.add_axes([0.30, 0.012, 0.06, 0.032])
        ax_btn.set_facecolor(BG_CARD)
        self.btn_play = Button(ax_btn, "|| Pause",
                               color=BG_CARD, hovercolor="#2a2b38")
        self.btn_play.label.set_color(COL_TEXT)
        self.btn_play.label.set_fontsize(8)
        self.btn_play.label.set_family("sans-serif")
        self.btn_play.on_clicked(self._toggle_play)

        # Slider de step
        ax_sl = fig.add_axes([0.38, 0.015, 0.30, 0.020])
        ax_sl.set_facecolor(BG)
        self.slider_step = Slider(
            ax_sl, "Jour", 0, self.nf - 1,
            valinit=0, valstep=1,
            color=COL_ACCENT, track_color=COL_GRID,
        )
        self.slider_step.label.set_color(COL_MUTED)
        self.slider_step.label.set_fontsize(7.5)
        self.slider_step.valtext.set_color(COL_MUTED)
        self.slider_step.valtext.set_fontsize(7.5)
        self.slider_step.on_changed(self._on_slider)

        # Slider de vitesse
        ax_spd = fig.add_axes([0.71, 0.015, 0.12, 0.020])
        ax_spd.set_facecolor(BG)
        self.slider_speed = Slider(
            ax_spd, "Vitesse", 1, 6,
            valinit=1, valstep=1,
            color="#5b5fc7", track_color=COL_GRID,
        )
        self.slider_speed.label.set_color(COL_MUTED)
        self.slider_speed.label.set_fontsize(7.5)
        self.slider_speed.valtext.set_color(COL_MUTED)
        self.slider_speed.valtext.set_fontsize(7.5)
        self.slider_speed.on_changed(lambda v: setattr(self, "speed", int(v)))

        # Bouton restart
        ax_rst = fig.add_axes([0.85, 0.012, 0.06, 0.032])
        ax_rst.set_facecolor(BG_CARD)
        self.btn_rst = Button(ax_rst, "|< Debut",
                              color=BG_CARD, hovercolor="#2a2b38")
        self.btn_rst.label.set_color(COL_TEXT)
        self.btn_rst.label.set_fontsize(8)
        self.btn_rst.label.set_family("sans-serif")
        self.btn_rst.on_clicked(self._restart)

        # Info clavier
        fig.text(0.03, 0.025, "Espace : lecture/pause  |  [<] [>] : frame  |  R : recommencer  |  haut/bas : vitesse",
                 fontsize=7, color=COL_MUTED)

        fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ── Mise à jour ─────────────────────────────────────────────────────────

    def _update(self, fi: int):
        fi = int(np.clip(fi, 0, self.nf - 1))
        self.fi = fi

        step    = fi * self.FRAME_EVERY
        metrics = compute_metrics(self.df, step)

        # ── Grille ──
        self.im.set_data(self.frames[fi])
        self.lbl_day.set_text(f"Jour {step:4d}")

        # ── Séries temporelles ──
        sub = self.df[self.df["step"] <= step]
        for key in SEIR_KEYS:
            self.ts_lines[key].set_data(sub["step"].values, sub[key].values)

        # Remplissage I (recréer à chaque frame pour simplicité)
        self.ts_fill.remove()
        self.ts_fill = self.ax_ts.fill_between(
            sub["step"].values, sub["I"].values,
            alpha=0.18, color=SEIR_COLORS["I"], zorder=1,
        )
        self.ts_cursor.set_xdata([step, step])
        self.inc_cursor.set_xdata([step, step])

        # ── Dashboard ──
        total = metrics["N"]
        for key in SEIR_KEYS:
            count = metrics[key]
            pct   = count / total * 100.0
            self.dash_vals[key].set_text(f"{count:>6,}")
            self.dash_vals[f"{key}_pct"].set_text(f"{pct:5.1f} %")
            # barre de proportion
            bar   = self.dash_bars[key]
            w     = pct / 100.0 * 0.41
            bar.set_width(max(w, 0.002))

        self.dash_vals["attack"].set_text(f"{metrics['attack']:.1f} %")
        reff = metrics["reff"]
        reff_col = "#dc1f1f" if reff > 1.0 else "#1ab84c"
        self.dash_vals["reff"].set_text(f"{reff:.2f}")
        self.dash_vals["reff"].set_color(reff_col)
        self.dash_vals["peak"].set_text(
            f"J{metrics['peak_step']}  {metrics['peak_I']:,}")
        self.dash_vals["phase"].set_text(metrics["phase"])
        self.dash_vals["phase"].set_color(metrics["phase_col"])

        # Slider sync sans déclencher le callback
        self.slider_step.eventson = False
        self.slider_step.set_val(fi)
        self.slider_step.eventson = True

        self.fig.canvas.draw_idle()

    def _animate_step(self, _frame):
        """Callback appelé par FuncAnimation."""
        if self.playing:
            self.fi = (self.fi + self.speed) % self.nf
        self._update(self.fi)

    # ── Callbacks contrôles ─────────────────────────────────────────────────

    def _toggle_play(self, _event=None):
        self.playing = not self.playing
        self.btn_play.label.set_text("|| Pause" if self.playing else ">  Lecture")
        self.fig.canvas.draw_idle()

    def _restart(self, _event=None):
        self.fi = 0
        self._update(0)

    def _on_slider(self, val):
        self._update(int(val))

    def _on_key(self, event):
        if event.key == " ":
            self._toggle_play()
        elif event.key == "right":
            self._update(min(self.fi + 1, self.nf - 1))
        elif event.key == "left":
            self._update(max(self.fi - 1, 0))
        elif event.key in ("r", "R"):
            self._restart()
        elif event.key == "up":
            self.speed = min(self.speed + 1, 6)
        elif event.key == "down":
            self.speed = max(self.speed - 1, 1)

    # ── Point d'entrée principal ─────────────────────────────────────────────

    def run(self, interval_ms: int = 80):
        self.anim = animation.FuncAnimation(
            self.fig, self._animate_step,
            frames=None,          # infini
            interval=interval_ms,
            cache_frame_data=False,
        )
        plt.show()

    def save(self, output: str, fps: int = 20):
        """Exporte en MP4 via ffmpeg."""
        print(f"[Export] → {output}  ({self.nf} frames, {fps} fps)")

        def _save_frame(fi):
            self._update(fi)

        ani = animation.FuncAnimation(
            self.fig, _save_frame,
            frames=self.nf, interval=50, cache_frame_data=False,
        )
        writer = animation.FFMpegWriter(
            fps=fps, bitrate=2400,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "fast"],
        )
        ani.save(output, writer=writer, dpi=130,
                 savefig_kwargs={"facecolor": BG})
        print(f"[Export] Terminé → {output}")


# ── Mode comparaison ─────────────────────────────────────────────────────────

def compare_mode(csv_files, labels, compartment="I", output=None):
    """
    Compare plusieurs runs sur tous les compartiments (S, E, I, R) + taux d'attaque.
    Si compartment != 'all', affiche seulement ce compartiment + taux d'attaque.
    Avec compartment='all' (défaut), affiche les 4 compartiments en grille 2×2.
    """
    palette = ["#4e9af1", "#f0bb14", "#1ab84c", "#dc1f1f", "#b05cf0", "#f07014"]
    labels  = labels or [os.path.basename(f) for f in csv_files]
    dfs     = [load_csv(f) for f in csv_files]

    # ── mode "tous les compartiments" ────────────────────────────────────
    if compartment.lower() == "all":
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), facecolor=BG)
        fig.suptitle("Comparaison des runs — S / E / I / R",
                     fontsize=11, fontweight="bold", color=COL_TEXT, y=0.99)

        comps   = ["S", "E", "I", "R"]
        titles  = ["Susceptibles (S)", "Exposés (E)",
                   "Infectieux (I)",   "Rétablis (R)"]
        colors_seir = {
            "S": "#3878c8", "E": "#f0bb14",
            "I": "#dc1f1f", "R": "#1ab84c",
        }

        for idx, (key, title) in enumerate(zip(comps, titles)):
            ax = axes[idx // 2][idx % 2]
            ax.set_facecolor(BG_PANEL)
            ax.grid(True, alpha=0.35)
            for sp in ax.spines.values():
                sp.set_edgecolor(COL_GRID)
            for i, (df, lbl) in enumerate(zip(dfs, labels)):
                # premier run en couleur SEIR, autres en nuances
                col = colors_seir[key] if i == 0 else palette[i]
                lw  = 2.0 if i == 0 else 1.4
                ls  = "-"  if i == 0 else ["--", ":", "-."][min(i-1, 2)]
                ax.plot(df["step"], df[key], lw=lw, ls=ls, color=col, label=lbl)
            ax.set_title(title, color=COL_TEXT, fontsize=9, pad=4)
            ax.set_xlabel("Jours", fontsize=8)
            ax.set_ylabel("Individus", fontsize=8)
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            ax.legend(fontsize=7.5, framealpha=0.8)

        # taux d'attaque
        ax_att = axes[1][2]
        ax_att.set_facecolor(BG_PANEL)
        ax_att.grid(True, alpha=0.35)
        for sp in ax_att.spines.values():
            sp.set_edgecolor(COL_GRID)
        for i, (df, lbl) in enumerate(zip(dfs, labels)):
            ax_att.plot(df["step"], df["attack_rate"],
                        lw=1.6, color=palette[i], label=lbl)
        ax_att.set_title("Taux d'attaque cumulé (%)", color=COL_TEXT, fontsize=9)
        ax_att.set_xlabel("Jours", fontsize=8)
        ax_att.set_ylabel("%", fontsize=8)
        ax_att.legend(fontsize=7.5)

        # panneau vide → légende globale
        axes[0][2].axis("off")
        handles = [
            plt.Line2D([0], [0], color=palette[i], lw=2, label=lbl)
            for i, lbl in enumerate(labels)
        ]
        axes[0][2].legend(handles=handles, loc="center", fontsize=9,
                          facecolor=BG_CARD, edgecolor=COL_GRID,
                          labelcolor=COL_TEXT, framealpha=0.9,
                          title="Runs comparés", title_fontsize=8)

    # ── mode "un seul compartiment" ──────────────────────────────────────
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
        fig.suptitle(f"Comparaison des runs — compartiment {compartment}",
                     fontsize=11, fontweight="bold", color=COL_TEXT, y=0.97)
        ax1, ax2 = axes
        for ax in axes:
            ax.set_facecolor(BG_PANEL)
            ax.grid(True, alpha=0.35)
            for sp in ax.spines.values():
                sp.set_edgecolor(COL_GRID)
        for i, (df, lbl) in enumerate(zip(dfs, labels)):
            col = palette[i % len(palette)]
            ax1.plot(df["step"], df[compartment], lw=1.7, color=col, label=lbl)
            ax2.plot(df["step"], df["attack_rate"], lw=1.7, color=col, label=lbl,
                     ls="--" if i > 0 else "-")
        ax1.set_title(f"Évolution de {compartment}", color=COL_TEXT, fontsize=9)
        ax1.set_xlabel("Jours"); ax1.set_ylabel(f"Individus ({compartment})")
        ax1.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax1.legend(fontsize=8.5)
        ax2.set_title("Taux d'attaque cumulé (%)", color=COL_TEXT, fontsize=9)
        ax2.set_xlabel("Jours"); ax2.set_ylabel("Taux d'attaque (%)")
        ax2.legend(fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"[Compare] Sauvegardé → {output}")
    else:
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualiseur scientifique SEIR multi-agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--bin",     default="frames_seq.bin", metavar="FILE",
                        help="fichier binaire des frames (.bin)")
    parser.add_argument("--csv",     default="counts_seq.csv", metavar="FILE",
                        help="fichier CSV des comptages SEIR")
    parser.add_argument("--save",    default=None, metavar="OUT.mp4",
                        help="sauvegarder l'animation en MP4 (nécessite ffmpeg)")
    parser.add_argument("--fps",     default=20, type=int,
                        help="images/seconde pour l'export MP4 (défaut : 20)")
    parser.add_argument("--compare", nargs="+", metavar="CSV",
                        help="mode comparaison : liste de fichiers CSV")
    parser.add_argument("--labels",  nargs="+", metavar="LABEL",
                        help="étiquettes pour le mode --compare")
    parser.add_argument("--comp",    default="all",
                        help="compartiment à comparer : S, E, I, R, ou all (défaut)")
    parser.add_argument("--output",  default=None, metavar="FILE",
                        help="fichier de sortie pour --compare (PNG)")
    parser.add_argument("--speed",   default=80, type=int, metavar="MS",
                        help="intervalle entre frames en ms (défaut : 80)")
    args = parser.parse_args()

    if args.compare:
        compare_mode(args.compare, args.labels, args.comp, args.output)
        return

    for path in (args.bin, args.csv):
        if not os.path.exists(path):
            print(f"[Erreur] Fichier introuvable : {path}", file=sys.stderr)
            sys.exit(1)

    viewer = SEIRViewer(args.bin, args.csv)

    if args.save:
        viewer.save(args.save, fps=args.fps)
    else:
        viewer.run(interval_ms=args.speed)


if __name__ == "__main__":
    main()
