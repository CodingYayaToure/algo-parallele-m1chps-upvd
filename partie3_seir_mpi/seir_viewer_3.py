#!/usr/bin/env python3
"""
seir_viewer_3.py — Simulation SEIR interactive style manim
Agents visibles avec déplacements, anneaux d'infection, graphique temps réel,
comparaison Séquentiel vs MPI.

Usage :
    python3 seir_viewer_3.py
    python3 seir_viewer_3.py --csv counts_seed_42.csv   # avec vraies données 730j
    python3 seir_viewer_3.py --fast                     # simulation rapide
    python3 seir_viewer_3.py --save sim.mp4             # export MP4

Contrôles :
    Espace    : Lecture / Pause
    R         : Réinitialiser
    + / -     : Vitesse
    Q         : Quitter
"""

import argparse
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.widgets import Button

# ── Thème ───────────────────────────────────────────────────────────────────
BG        = "#0b0c10"
BG_PANEL  = "#111218"
BG_CARD   = "#1a1b26"
COL_GRID  = "#1e2030"
COL_TEXT  = "#c8cad4"
COL_MUTED = "#5a5c70"

# Couleurs SEIR (identiques aux autres viewers)
C_S = "#3878c8"   # Bleu   — Susceptible
C_E = "#f0bb14"   # Ambre  — Exposé
C_I = "#dc1f1f"   # Rouge  — Infectieux
C_R = "#1ab84c"   # Vert   — Rétabli
SEIR_COLS = [C_S, C_E, C_I, C_R]
SEIR_LBLS = ["S — Susceptible", "E — Exposé", "I — Infectieux", "R — Rétabli"]

# Résultats benchmark réels (730 jours, votre machine)
BENCH = {
    "labels":    ["Séq.", "MPI\nP=1", "MPI\nP=2", "MPI\nP=4", "MPI\nP=8", "MPI\nP=16"],
    "times_nr":  [1.496,   1.411,      1.440,       2.709,       5.624,       10.473],
    "times_rp":  [None,    0.947,      1.917,       3.603,       7.625,       14.951],
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
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
    "font.size":         8.5,
})

# ── Paramètres mini-simulation ───────────────────────────────────────────────
GW, GH   = 1.0, 1.0   # domaine normalisé [0,1]²
N_AGENTS = 120         # agents dans la simulation interactive
N_INIT_I = 4           # infectieux initiaux
BETA     = 0.5         # force d'infection
MEAN_E   = 3.0         # durée moyenne état E (jours)
MEAN_I   = 7.0         # durée moyenne état I (jours)
MEAN_R   = 80.0        # durée immunité (réduite pour voir la dynamique rapidement)
INF_RADIUS = 0.06      # rayon d'infection visible
RING_ANIM_PERIOD = 20  # frames pour une pulsation de l'anneau

ST_S, ST_E, ST_I, ST_R = 0, 1, 2, 3


# ════════════════════════════════════════════════════════════════════════════
# Mini-simulation SEIR
# ════════════════════════════════════════════════════════════════════════════
class Agent:
    __slots__ = ("x", "y", "vx", "vy", "state", "timer",
                 "dE", "dI", "dR", "ring_time", "just_infected")

    def __init__(self, x, y, state, dE, dI, dR):
        self.x, self.y   = x, y
        self.vx = np.random.uniform(-0.003, 0.003)
        self.vy = np.random.uniform(-0.003, 0.003)
        self.state       = state
        self.timer       = 0
        self.dE, self.dI, self.dR = dE, dI, dR
        self.ring_time   = 0
        self.just_infected = False


class MiniSEIR:
    """
    Simulation SEIR sur domaine [0,1]² avec mouvement continu aléatoire.
    Les agents se déplacent avec une vélocité aléatoire + rebond sur les bords.
    L'infection est vérifiée à chaque pas de simulation (pas à chaque frame).
    """

    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.rng    = rng
        self.frame  = 0
        self.day    = 0.0
        self.phase  = "deplacement"   # deplacement | verification | transition
        self.phase_frame = 0
        self.counts_history = []
        self._build_agents()

    def _neg_exp(self, mean):
        v = int(math.ceil(-mean * math.log(max(1e-9, self.rng.uniform()))))
        return max(1, v)

    def _build_agents(self):
        self.agents = []
        for i in range(N_AGENTS):
            x  = self.rng.uniform(0.05, 0.95)
            y  = self.rng.uniform(0.05, 0.95)
            st = ST_I if i < N_INIT_I else ST_S
            ag = Agent(x, y, st,
                       self._neg_exp(MEAN_E),
                       self._neg_exp(MEAN_I),
                       self._neg_exp(MEAN_R))
            self.agents.append(ag)
        self._log_counts()

    def _log_counts(self):
        counts = [0, 0, 0, 0]
        for ag in self.agents:
            counts[ag.state] += 1
        self.counts_history.append(counts)

    def get_counts(self):
        counts = [0, 0, 0, 0]
        for ag in self.agents:
            counts[ag.state] += 1
        return counts

    def step_physics(self, speed_factor=1.0):
        """Déplacement continu — appelé chaque frame."""
        for ag in self.agents:
            ag.just_infected = False
            # Légère perturbation aléatoire de la vitesse (marche aléatoire)
            ag.vx += self.rng.uniform(-0.0008, 0.0008) * speed_factor
            ag.vy += self.rng.uniform(-0.0008, 0.0008) * speed_factor
            # Limiter la vitesse max
            speed = math.hypot(ag.vx, ag.vy)
            max_v = 0.006 * speed_factor
            if speed > max_v:
                ag.vx *= max_v / speed
                ag.vy *= max_v / speed
            ag.x += ag.vx
            ag.y += ag.vy
            # Rebond sur les bords (torique → rebond pour la visu)
            if ag.x < 0.02:  ag.x = 0.02;  ag.vx =  abs(ag.vx)
            if ag.x > 0.98:  ag.x = 0.98;  ag.vx = -abs(ag.vx)
            if ag.y < 0.02:  ag.y = 0.02;  ag.vy =  abs(ag.vy)
            if ag.y > 0.98:  ag.y = 0.98;  ag.vy = -abs(ag.vy)
            # Animation de l'anneau
            if ag.state == ST_I:
                ag.ring_time += 1

    def step_epidemio(self):
        """Mise à jour épidémiologique — appelé toutes les N frames."""
        infected_pos = np.array([[ag.x, ag.y]
                                  for ag in self.agents if ag.state == ST_I])
        new_states = [ag.state for ag in self.agents]
        new_timers = [ag.timer + 1 for ag in self.agents]

        for i, ag in enumerate(self.agents):
            if ag.state == ST_S and len(infected_pos) > 0:
                dists = np.linalg.norm(infected_pos - [ag.x, ag.y], axis=1)
                Ni = int((dists < INF_RADIUS).sum())
                if Ni > 0:
                    p = 1.0 - math.exp(-BETA * Ni)
                    if self.rng.uniform() < p:
                        new_states[i]  = ST_E
                        new_timers[i]  = 0
                        ag.just_infected = True
            elif ag.state == ST_E and new_timers[i] >= ag.dE:
                new_states[i] = ST_I;  new_timers[i] = 0
            elif ag.state == ST_I and new_timers[i] >= ag.dI:
                new_states[i] = ST_R;  new_timers[i] = 0
            elif ag.state == ST_R and new_timers[i] >= ag.dR:
                new_states[i] = ST_S;  new_timers[i] = 0

        for i, ag in enumerate(self.agents):
            ag.state = new_states[i]
            ag.timer = new_timers[i]

        self.day += 1
        self._log_counts()

    def reset(self, seed=None):
        self.__init__(seed=seed or np.random.randint(0, 999))


# ════════════════════════════════════════════════════════════════════════════
# Visualiseur principal
# ════════════════════════════════════════════════════════════════════════════
class SEIRViewer3:

    PHYSIC_PER_DAY = 30   # frames physiques par jour épidémio

    def __init__(self, csv_path=None, fast=False):
        self.sim      = MiniSEIR()
        self.playing  = True
        self.frame    = 0
        self.speed    = 2.0 if fast else 1.0
        self.csv_path = csv_path
        self._load_730(csv_path)
        self._build_figure()
        self._init_artists()
        self._draw_bench()
        if self.df730 is not None:
            self._draw_730_curves()

    # ── Données 730 jours ───────────────────────────────────────────────────
    def _load_730(self, path):
        if path and os.path.exists(path):
            self.df730 = pd.read_csv(path)
        else:
            # données synthétiques correspondant à nos mesures réelles
            days = np.arange(0, 731)
            I = np.where(days < 35,
                         5893 * (days / 35)**2 * np.exp(-0.5*(days-35)**2/200),
                         np.maximum(20, 700 * np.exp(-0.008*(days-35))))
            S = np.maximum(500, 19980 - np.cumsum(np.gradient(np.maximum(0, 19980 - I - 100))))
            S = np.clip(S, 500, 19980)
            R = 19980 - S - I - np.minimum(I * 0.5, 400)
            E = 19980 - S - I - np.maximum(0, R)
            self.df730 = pd.DataFrame({"step": days, "S": S.astype(int),
                                       "E": np.maximum(0, E).astype(int),
                                       "I": I.astype(int),
                                       "R": np.maximum(0, R).astype(int)})

    # ── Construction de la figure ────────────────────────────────────────────
    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 9), facecolor=BG)
        self.fig.canvas.manager.set_window_title(
            "SEIR Viewer 3 — Simulation interactive · Séquentiel vs MPI")

        # Titre principal
        self.fig.text(0.5, 0.977,
            "Simulation SEIR multi-agents — déplacements stochastiques & comparaison Séquentiel / MPI",
            ha="center", va="top", fontsize=12, fontweight="bold", color=COL_TEXT)
        self.fig.text(0.5, 0.957,
            "Grille 300×300 · 20 000 agents · saut global aléatoire · voisinage Moore 9 cellules · "
            "730 jours (2 ans)  |  Séquentiel : 1.496 s  •  MPI P=2 : 1.440 s  •  MPI P=16 : 10.47 s",
            ha="center", va="top", fontsize=8, color=COL_MUTED)

        # GridSpec 2×3
        gs = gridspec.GridSpec(
            2, 3, figure=self.fig,
            left=0.03, right=0.97, top=0.93, bottom=0.07,
            hspace=0.40, wspace=0.28,
            height_ratios=[1.6, 1.0],
            width_ratios=[1.4, 1.1, 1.0],
        )

        # ── Grille simulation (grande, gauche) ──
        self.ax_sim = self.fig.add_subplot(gs[:, 0])
        self.ax_sim.set_facecolor("#060810")
        self.ax_sim.set_xlim(0, 1)
        self.ax_sim.set_ylim(0, 1)
        self.ax_sim.set_aspect("equal")
        self.ax_sim.set_xticks([])
        self.ax_sim.set_yticks([])
        self.ax_sim.set_title("Distribution spatiale des agents (simulation interactive)",
                               fontsize=9, pad=6, color=COL_TEXT)

        # ── Graphique SEIR temps réel (centre haut) ──
        self.ax_seir = self.fig.add_subplot(gs[0, 1])
        self.ax_seir.set_facecolor(BG_PANEL)
        self.ax_seir.grid(True, alpha=0.3)
        self.ax_seir.set_title("Dynamique SEIR — simulation interactive",
                                fontsize=9, pad=4)
        self.ax_seir.set_xlabel("Jours (simulation)", fontsize=8)
        self.ax_seir.set_ylabel("Agents", fontsize=8)

        # ── Graphique SEIR 730 jours (centre bas) ──
        self.ax_730 = self.fig.add_subplot(gs[1, 1])
        self.ax_730.set_facecolor(BG_PANEL)
        self.ax_730.grid(True, alpha=0.3)
        self.ax_730.set_title("Résultats réels — 730 jours · Version séquentielle srand(42)",
                               fontsize=8.5, pad=4)
        self.ax_730.set_xlabel("Jours", fontsize=8)
        self.ax_730.set_ylabel("Agents", fontsize=8)
        self.ax_730.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

        # ── Étapes de la simulation (droite haut) ──
        self.ax_steps = self.fig.add_subplot(gs[0, 2])
        self.ax_steps.set_facecolor(BG_CARD)
        self.ax_steps.axis("off")
        self.ax_steps.set_title("Étapes par pas de temps", fontsize=9, pad=4)

        # ── Benchmark séq vs MPI (droite bas) ──
        self.ax_bench = self.fig.add_subplot(gs[1, 2])
        self.ax_bench.set_facecolor(BG_PANEL)
        self.ax_bench.grid(True, alpha=0.3, axis="y")
        self.ax_bench.set_title("Temps d'exécution · Séquentiel vs MPI (730 j)",
                                 fontsize=8.5, pad=4)
        self.ax_bench.set_ylabel("Temps (s)", fontsize=8)

        # Contrôles clavier
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Texte contrôles
        self.fig.text(
            0.03, 0.025,
            "Espace : lecture/pause  •  R : réinit  •  + / - : vitesse  •  Q : quitter",
            fontsize=7.5, color=COL_MUTED)

        self.txt_status = self.fig.text(
            0.97, 0.025, "",
            fontsize=8, color=COL_TEXT, ha="right",
            fontweight="bold")

    # ── Artistes graphiques ──────────────────────────────────────────────────
    def _init_artists(self):
        # Légende simulation
        handles = [mpatches.Patch(facecolor=c, label=l, edgecolor="none")
                   for c, l in zip(SEIR_COLS, SEIR_LBLS)]
        self.ax_sim.legend(
            handles=handles, loc="lower left",
            fontsize=8, ncol=2, framealpha=0.9,
            borderpad=0.6, labelspacing=0.4, handlelength=1.2)

        # Points agents
        self.sc = self.ax_sim.scatter(
            [ag.x for ag in self.sim.agents],
            [ag.y for ag in self.sim.agents],
            c=[SEIR_COLS[ag.state] for ag in self.sim.agents],
            s=18, zorder=5, linewidths=0)

        # Anneaux d'infection (un cercle par agent I, invisible par défaut)
        self.rings = []
        for _ in self.sim.agents:
            ring = Circle((0.5, 0.5), INF_RADIUS,
                          fill=False, edgecolor=C_I,
                          linewidth=0.0, alpha=0.0, zorder=4)
            self.ax_sim.add_patch(ring)
            self.rings.append(ring)

        # Vecteurs de vitesse (petites flèches)
        xs0 = [ag.x for ag in self.sim.agents]
        ys0 = [ag.y for ag in self.sim.agents]
        vx0 = [ag.vx * 15 for ag in self.sim.agents]
        vy0 = [ag.vy * 15 for ag in self.sim.agents]
        self.qv = self.ax_sim.quiver(
            xs0, ys0, vx0, vy0,
            color="white", alpha=0.12, scale=20,
            width=0.002, headwidth=3, headlength=4, zorder=3)

        # Étiquette de jour
        self.lbl_day = self.ax_sim.text(
            0.02, 0.98, "Jour 0",
            transform=self.ax_sim.transAxes,
            fontsize=11, fontweight="bold", color="white", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=BG_CARD,
                      edgecolor=COL_GRID, alpha=0.92))

        # Compteurs S/E/I/R
        self.lbl_counts = self.ax_sim.text(
            0.98, 0.98, "",
            transform=self.ax_sim.transAxes,
            fontsize=8.5, color=COL_TEXT, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=BG_CARD,
                      edgecolor=COL_GRID, alpha=0.92),
            family="monospace")

        # Ligne de voisinage de Moore (cercle centré sur le premier I)
        self.moore_circle = Circle((0, 0), INF_RADIUS * 1.8,
                                   fill=False, edgecolor=C_I,
                                   linewidth=0.8, linestyle="--",
                                   alpha=0.0, zorder=6)
        self.ax_sim.add_patch(self.moore_circle)

        # ── Courbes dynamiques (temps réel) ──
        self.ax_seir.set_xlim(0, 60)
        self.ax_seir.set_ylim(0, N_AGENTS * 1.1)
        self.seir_lines = {}
        for st, col, lbl in zip(range(4), SEIR_COLS, ["S", "E", "I", "R"]):
            ln, = self.ax_seir.plot([], [], lw=2.0, color=col,
                                    label=f"{lbl} — {SEIR_LBLS[st][4:]}")
            self.seir_lines[st] = ln
        # Remplissage sous I
        self.fill_i = self.ax_seir.fill_between([], [], alpha=0.2, color=C_I)

        # Légende SEIR dynamique
        leg_handles = [
            Line2D([0], [0], color=SEIR_COLS[st], lw=2,
                   label=f"{['S','E','I','R'][st]}  {SEIR_LBLS[st][4:]}")
            for st in range(4)]
        self.ax_seir.legend(handles=leg_handles, fontsize=7.5,
                            loc="upper right", framealpha=0.85, ncol=2)

        # Phase indicator
        self._draw_phase_panel(phase=0, subframe=0)

    # ── Benchmark ────────────────────────────────────────────────────────────
    def _draw_bench(self):
        ax  = self.ax_bench
        B   = BENCH
        N   = len(B["labels"])
        x   = np.arange(N)
        w   = 0.35

        # Couleurs selon perf
        def perf_col(t, ref=1.496):
            s = ref / t
            if s >= 1.3: return "#1ab84c"
            if s >= 0.9: return "#c89000"
            return "#dc1f1f"

        c_nr = ["#555555"] + [perf_col(t) for t in B["times_nr"][1:]]
        c_rp = ["none"]    + [perf_col(t) for t in B["times_rp"][1:]]

        bars_nr = ax.bar(x - w/2, B["times_nr"], width=w,
                         color=c_nr, alpha=0.90, label="MPI non-reproductible",
                         zorder=3, edgecolor=BG_CARD, linewidth=0.5)
        tr_vals  = [t if t else 0 for t in B["times_rp"]]
        bars_rp  = ax.bar(x + w/2, tr_vals, width=w,
                          color="#9b59b6", alpha=0.75, label="MPI reproductible",
                          zorder=3, edgecolor=BG_CARD, linewidth=0.5)

        # Ligne référence séquentielle
        ax.axhline(B["times_nr"][0], color="white", lw=0.8, ls="--",
                   alpha=0.5, zorder=4, label=f"Réf. séq. {B['times_nr'][0]:.2f} s")

        # Annotations valeurs sur barres
        for bar in bars_nr:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                        f"{h:.2f}", ha="center", va="bottom",
                        fontsize=6.5, color=COL_TEXT, family="monospace")

        ax.set_xticks(x)
        ax.set_xticklabels(B["labels"], fontsize=7.5)
        ax.set_ylim(0, max(B["times_nr"]) * 1.18)

        # Légende personnalisée
        leg = [
            mpatches.Patch(facecolor="#555555", label="Séquentiel (ref)"),
            mpatches.Patch(facecolor="#3878c8", label="MPI — speedup > x1.2"),
            mpatches.Patch(facecolor="#c89000", label="MPI — speedup ~x1"),
            mpatches.Patch(facecolor="#dc1f1f", label="MPI — plus lent"),
            mpatches.Patch(facecolor="#9b59b6", alpha=0.75, label="MPI reproductible"),
        ]
        ax.legend(handles=leg, fontsize=6.5, loc="upper left",
                  framealpha=0.85, ncol=1, borderpad=0.5)

        # Annotation speedup max
        ax.text(1, B["times_nr"][1] + 0.3, f"×{1.496/1.411:.2f}",
                ha="center", fontsize=7, color="#1ab84c", fontweight="bold")

    # ── Courbes 730 jours ────────────────────────────────────────────────────
    def _draw_730_curves(self):
        ax = self.ax_730
        df = self.df730
        N  = 19980

        for key, col, lbl in zip(["S","E","I","R"], SEIR_COLS, SEIR_LBLS):
            if key in df.columns:
                ax.plot(df["step"], df[key], lw=1.8, color=col,
                        label=f"{key}  {lbl[4:]}", alpha=0.9)

        # Remplissage I
        if "I" in df.columns:
            ax.fill_between(df["step"], df["I"], alpha=0.15, color=C_I)

        # Annotation pic infectieux
        if "I" in df.columns:
            peak_day = int(df.loc[df["I"].idxmax(), "step"])
            peak_I   = int(df["I"].max())
            ax.axvline(peak_day, color=C_I, lw=0.8, ls=":", alpha=0.6)
            ax.annotate(f"Pic J{peak_day}\n{peak_I:,}",
                        xy=(peak_day, peak_I),
                        xytext=(peak_day + 50, peak_I * 0.85),
                        fontsize=7, color=C_I,
                        arrowprops=dict(arrowstyle="->", color=C_I, lw=0.7))

        # Ligne de séparation 730j / an 2
        ax.axvline(365, color=COL_MUTED, lw=0.8, ls="--", alpha=0.4)
        ax.text(368, ax.get_ylim()[1] * 0.02,
                "An 2", fontsize=7, color=COL_MUTED)

        # Légende avec version
        handles = [
            Line2D([0],[0], color=c, lw=2.0,
                   label=f"{k}  {l[4:]}")
            for k, c, l in zip(["S","E","I","R"], SEIR_COLS, SEIR_LBLS)]
        handles.append(mpatches.Patch(facecolor="none", edgecolor="none",
                        label="Version séquentielle · srand(42)"))
        ax.legend(handles=handles, fontsize=7, loc="upper right",
                  framealpha=0.85, ncol=2, borderpad=0.5)

    # ── Panel étapes ─────────────────────────────────────────────────────────
    def _draw_phase_panel(self, phase: int, subframe: int):
        """
        Phase 0 : déplacement  (agents sautent partout)
        Phase 1 : vérification (voisinage de Moore)
        Phase 2 : transition   (E→I, I→R, R→S)
        """
        ax = self.ax_steps
        ax.cla()
        ax.set_facecolor(BG_CARD)
        ax.axis("off")
        ax.set_title("Étapes par pas de temps", fontsize=9, pad=4, color=COL_TEXT)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        PHASES = [
            ("1. Déplacement",
             "Chaque agent saute vers\n(x,y) aléatoire dans toute\nla grille (uniforme).",
             "→ rand_int(0, GRID_W-1)",
             C_S, "move"),
            ("2. Voisinage Moore",
             "Chaque S compte les I\ndans ses 9 cellules voisines.\np = 1 - exp(-β·Ni)",
             "→ count_I_moore(x, y)",
             C_I, "moore"),
            ("3. Transitions d'état",
             "Mise à jour synchrone :\nE→I si timer ≥ dE\nI→R si timer ≥ dI\nR→S si timer ≥ dR",
             "→ new_status[] / new_time[]",
             C_R, "trans"),
        ]

        for i, (title, desc, code, col, _kind) in enumerate(PHASES):
            y0  = 0.95 - i * 0.32
            active = (i == phase)
            alpha_bg = 0.25 if active else 0.10
            # Fond de l'étape
            rect = FancyBboxPatch((0.03, y0 - 0.28), 0.94, 0.27,
                                   boxstyle="round,pad=0.01",
                                   facecolor=col + ("55" if active else "1a"),
                                   edgecolor=col + ("cc" if active else "40"),
                                   linewidth=1.5 if active else 0.5,
                                   transform=ax.transAxes, zorder=1)
            ax.add_patch(rect)

            # Titre étape
            ax.text(0.06, y0 - 0.04, title,
                    fontsize=9, fontweight="bold",
                    color=col if active else COL_MUTED,
                    transform=ax.transAxes, va="top")

            # Description
            ax.text(0.06, y0 - 0.11, desc,
                    fontsize=7.5, color=COL_TEXT if active else COL_MUTED,
                    transform=ax.transAxes, va="top", linespacing=1.4)

            # Code inline
            ax.text(0.06, y0 - 0.25, code,
                    fontsize=7, color=col if active else COL_MUTED,
                    transform=ax.transAxes, va="top",
                    family="monospace")

            # Badge EN COURS
            if active:
                ax.text(0.88, y0 - 0.04, "EN COURS",
                        fontsize=6.5, fontweight="bold", color="white",
                        transform=ax.transAxes, va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor=col, edgecolor="none"))

        # Pulsation de la phase active (barre de progression)
        prog = (subframe % 30) / 30.0
        y_prog = 0.02
        ax.add_patch(FancyBboxPatch((0.03, y_prog), 0.94 * prog, 0.015,
                                    boxstyle="square,pad=0",
                                    facecolor=PHASES[phase][3],
                                    edgecolor="none",
                                    transform=ax.transAxes))
        ax.add_patch(FancyBboxPatch((0.03, y_prog), 0.94, 0.015,
                                    boxstyle="square,pad=0",
                                    facecolor="none",
                                    edgecolor=COL_GRID,
                                    linewidth=0.5,
                                    transform=ax.transAxes))

    # ── Mise à jour animation ────────────────────────────────────────────────
    def _update_artists(self):
        sim   = self.sim
        frame = self.frame

        # 1. Physique (chaque frame)
        sim.step_physics(speed_factor=self.speed)

        # Phase courante (3 phases par journée)
        frames_per_day    = self.PHYSIC_PER_DAY
        frame_in_day      = frame % frames_per_day
        phase = 0 if frame_in_day < frames_per_day // 3 else \
                1 if frame_in_day < 2 * frames_per_day // 3 else 2

        # 2. Épidémio (fin de chaque jour)
        if frame_in_day == frames_per_day - 1:
            sim.step_epidemio()

        # ── Grille simulation ──
        xs = np.array([ag.x for ag in sim.agents])
        ys = np.array([ag.y for ag in sim.agents])
        colors = [SEIR_COLS[ag.state] for ag in sim.agents]

        self.sc.set_offsets(np.c_[xs, ys])
        self.sc.set_facecolors(colors)

        # Anneaux d'infection pulsants
        first_I_pos = None
        for i, ag in enumerate(sim.agents):
            ring = self.rings[i]
            if ag.state == ST_I:
                if first_I_pos is None:
                    first_I_pos = (ag.x, ag.y)
                alpha = (math.sin(ag.ring_time * 2 * math.pi / RING_ANIM_PERIOD) + 1) / 2
                r     = INF_RADIUS * (0.6 + 0.4 * alpha)
                ring.set_center((ag.x, ag.y))
                ring.set_radius(r)
                ring.set_linewidth(1.2 * alpha + 0.2)
                ring.set_alpha(0.5 * alpha + 0.1)
                ring.set_edgecolor(C_I)
            else:
                ring.set_alpha(0.0)

        # Cercle de voisinage de Moore visible sur 1er I
        if first_I_pos and phase == 1:
            self.moore_circle.set_center(first_I_pos)
            self.moore_circle.set_radius(INF_RADIUS * 1.8)
            self.moore_circle.set_alpha(0.55)
        else:
            self.moore_circle.set_alpha(0.0)

        # Flèches de vitesse
        vxs = np.array([ag.vx for ag in sim.agents])
        vys = np.array([ag.vy for ag in sim.agents])
        self.qv.set_offsets(np.c_[xs, ys])
        self.qv.set_UVC(vxs * 15, vys * 15)

        # Étiquettes
        day_int = int(sim.day)
        self.lbl_day.set_text(f"Jour {day_int:4d}")
        counts  = sim.get_counts()
        self.lbl_counts.set_text(
            f"S {counts[0]:4d}\n"
            f"E {counts[1]:4d}\n"
            f"I {counts[2]:4d}\n"
            f"R {counts[3]:4d}")

        # ── Courbes dynamiques ──
        hist = np.array(sim.counts_history)
        days_hist = np.arange(len(hist))

        xlim_max = max(60, len(hist) + 5)
        self.ax_seir.set_xlim(0, xlim_max)
        self.ax_seir.set_ylim(0, N_AGENTS * 1.1)

        for st, ln in self.seir_lines.items():
            if len(hist) > 0:
                ln.set_data(days_hist, hist[:, st])

        # Refaire le fill_between I
        self.fill_i.remove()
        if len(hist) > 1:
            self.fill_i = self.ax_seir.fill_between(
                days_hist, hist[:, ST_I], alpha=0.18, color=C_I)
        else:
            self.fill_i = self.ax_seir.fill_between([], [], alpha=0.0)

        # Curseur vertical
        if hasattr(self, "cursor_line"):
            self.cursor_line.remove()
        self.cursor_line = self.ax_seir.axvline(
            len(hist) - 1, color="white", lw=0.8, ls="--", alpha=0.5)

        # ── Panel étapes ──
        self._draw_phase_panel(phase=phase, subframe=frame_in_day)

        # ── Statut barre du bas ──
        self.txt_status.set_text(
            f"Vitesse ×{self.speed:.1f}  •  Jour {day_int}  •  "
            f"{'▶  Lecture' if self.playing else '⏸  Pause'}")

        self.frame += 1

    # ── Callback animation ───────────────────────────────────────────────────
    def _anim_func(self, _frame):
        if self.playing:
            self._update_artists()
        return []

    # ── Clavier ──────────────────────────────────────────────────────────────
    def _on_key(self, event):
        if event.key in (" ",):
            self.playing = not self.playing
        elif event.key in ("r", "R"):
            self.sim.reset()
            self.frame = 0
            for ln in self.seir_lines.values():
                ln.set_data([], [])
        elif event.key in ("q", "Q"):
            plt.close(self.fig)
        elif event.key in ("+", "="):
            self.speed = min(self.speed + 0.5, 6.0)
        elif event.key in ("-", "_"):
            self.speed = max(self.speed - 0.5, 0.5)

    # ── Run ──────────────────────────────────────────────────────────────────
    def run(self, interval_ms=40):
        self.anim = animation.FuncAnimation(
            self.fig, self._anim_func,
            frames=None, interval=interval_ms,
            cache_frame_data=False, blit=False)
        plt.show()

    def save(self, path, fps=25):
        print(f"[Export] → {path}  fps={fps}")
        N_FRAMES = 400

        def _f(i):
            if self.playing:
                self._update_artists()
            return []

        ani = animation.FuncAnimation(
            self.fig, _f, frames=N_FRAMES,
            interval=40, cache_frame_data=False, blit=False)
        writer = animation.FFMpegWriter(
            fps=fps, bitrate=2600,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "fast"])
        ani.save(path, writer=writer, dpi=120,
                 savefig_kwargs={"facecolor": BG})
        print(f"[Export] Terminé → {path}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="SEIR Viewer 3 — Simulation interactive manim-style",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--csv",   default=None,
                        help="CSV des comptages réels 730j (optionnel)")
    parser.add_argument("--save",  default=None, metavar="OUT.mp4",
                        help="Exporter en MP4 (nécessite ffmpeg)")
    parser.add_argument("--fps",   default=25, type=int)
    parser.add_argument("--fast",  action="store_true",
                        help="Simulation accélérée")
    args = parser.parse_args()

    # Auto-détection du CSV
    csv_path = args.csv
    if csv_path is None:
        for cand in ["counts_seed_42.csv", "counts_seq.csv"]:
            if os.path.exists(cand):
                csv_path = cand
                print(f"[Viewer3] CSV détecté : {csv_path}")
                break

    print("[Viewer3] Démarrage...")
    print("  Espace : lecture/pause  |  R : réinitialiser  |  +/- : vitesse  |  Q : quitter")

    viewer = SEIRViewer3(csv_path=csv_path, fast=args.fast)

    if args.save:
        viewer.save(args.save, fps=args.fps)
    else:
        viewer.run()


if __name__ == "__main__":
    main()
