#!/usr/bin/env python3
"""
pi_viewer.py — Visualisation complète Partie 1 : Calcul de Pi
M1 CHPS — Algorithmique et Programmation Parallèle

Onglets :
  1. Méthode des trapèzes      — animation convergence interactive
  2. Convergence               — erreur vs n (log-log)
  3. Scalabilité MPI           — speedup vs P + loi d'Amdahl
  4. Déséquilibre de charge    — statique vs dynamique
  5. Dashboard résultats       — tableaux CSV + métriques

Usage :
  python3 pi_viewer.py
  python3 pi_viewer.py --tab speedup
"""

import argparse, os, sys, math, time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D

# ── Thème ────────────────────────────────────────────────────────────
BG      = "#0b0c10"
BG2     = "#111827"
BG3     = "#1f2937"
BORDER  = "#374151"
TEXT    = "#e2e8f0"
MUTED   = "#6b7280"
BLUE    = "#3b82f6"
GREEN   = "#10b981"
AMBER   = "#f59e0b"
RED     = "#ef4444"
PURPLE  = "#8b5cf6"
CYAN    = "#22d3ee"

PI_REF  = 3.14159265358979323846

matplotlib.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG2,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   MUTED,
    "axes.titlecolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "legend.facecolor":  BG3,
    "legend.edgecolor":  BORDER,
    "grid.color":        BORDER,
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         9,
})

def f(x): return 1.0 / (1.0 + x * x)

def pi_trapezes(n):
    dx = 1.0 / n
    s  = 0.5 * (f(0.0) + f(1.0))
    for i in range(1, n):
        s += f(i * dx)
    return 4.0 * s * dx

# ── Onglet 1 : Animation trapèzes ────────────────────────────────────
def build_tab_trapeze(fig, gs):
    ax  = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(BG2)
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("f(x) = 1/(1+x²)", fontsize=10)
    ax.set_title("Méthode des trapèzes — π = 4 × ∫₀¹ 1/(1+x²) dx",
                 fontsize=11, pad=8)
    ax.grid(True, alpha=0.25)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

    # Courbe exacte
    xs = np.linspace(0, 1, 500)
    ax.plot(xs, f(xs), color=BLUE, lw=2.5, zorder=5, label="f(x) = 1/(1+x²)")
    ax.fill_between(xs, f(xs), alpha=0.08, color=BLUE)

    # Axes repères
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(1, color=MUTED, lw=0.5, ls=":")

    # Éléments dynamiques
    trapeze_patches = []
    pi_text = ax.text(0.02, 1.06, "", fontsize=12, fontweight="bold",
                       color=GREEN, transform=ax.transAxes)
    err_text = ax.text(0.02, 1.01, "", fontsize=10, color=AMBER,
                        transform=ax.transAxes)
    n_text   = ax.text(0.98, 1.06, "", fontsize=10, color=MUTED,
                        transform=ax.transAxes, ha="right")

    def update_trapeze(n):
        for p in trapeze_patches: p.remove()
        trapeze_patches.clear()
        dx = 1.0 / n
        colors = plt.cm.cool(np.linspace(0.1, 0.9, n))
        for i in range(n):
            xi, xi1 = i * dx, (i + 1) * dx
            yi, yi1 = f(xi), f(xi1)
            verts = [(xi, 0), (xi, yi), (xi1, yi1), (xi1, 0)]
            p = plt.Polygon(verts, alpha=0.4,
                            facecolor=colors[i], edgecolor="white",
                            linewidth=0.3, zorder=3)
            ax.add_patch(p)
            trapeze_patches.append(p)

        pi_approx = pi_trapezes(n)
        err       = abs(pi_approx - PI_REF)
        pi_text.set_text(f"π ≈ {pi_approx:.10f}")
        err_text.set_text(f"erreur = {err:.2e}  ({n} trapèzes)")
        n_text.set_text(f"n = {n}")
        fig.canvas.draw_idle()

    # Slider
    ax_sl = fig.add_subplot(gs[-1, 0])
    ax_sl.set_facecolor(BG)
    ax_sl.axis("off")
    sl_ax = fig.add_axes([0.07, 0.04, 0.38, 0.025], facecolor=BG3)
    sl = Slider(sl_ax, "n trapèzes", 2, 200, valinit=10, valstep=1,
                color=BLUE)
    sl.label.set_color(MUTED)
    sl.valtext.set_color(TEXT)
    sl.on_changed(lambda v: update_trapeze(int(v)))

    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    update_trapeze(10)


# ── Onglet 2 : Convergence ───────────────────────────────────────────
def build_tab_convergence(ax1, ax2):
    # Données calculées analytiquement (convergence théorique O(1/n²))
    ns = np.logspace(1, 9, 100).astype(int)
    errs = [abs(pi_trapezes(min(n, 50000)) - PI_REF) for n in
            np.logspace(1, 4, 30).astype(int)]
    ns_small = np.logspace(1, 4, 30).astype(int)

    # Lire depuis CSV si disponible
    csv_path = "results_seq_convergence.csv"
    ns_csv, errs_csv, times_csv = [], [], []
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        ns_csv   = df["n"].values
        errs_csv = df["erreur_abs"].values
        times_csv = df["temps_s"].values

    # Graphique erreur vs n
    ax1.set_facecolor(BG2)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("n (nombre de trapèzes)")
    ax1.set_ylabel("|π_approx − π_ref|")
    ax1.set_title("Convergence — erreur absolue vs n", pad=6)
    ax1.grid(True, alpha=0.25)
    for sp in ax1.spines.values(): sp.set_edgecolor(BORDER)

    # Courbe théorique O(1/n²)
    ns_th = np.logspace(1, 9, 200)
    ax1.plot(ns_th, 1.0/ns_th**2, "--", color=MUTED, lw=1.2,
             label="Théorique O(1/n²)")

    if len(ns_csv) > 0:
        ax1.plot(ns_csv, errs_csv, "o-", color=BLUE, lw=2,
                 markersize=5, label="Mesuré (Kahan)")
        # Régression en loi de puissance
        if len(ns_csv) > 3:
            log_n = np.log10(ns_csv)
            log_e = np.log10(errs_csv)
            slope, intercept = np.polyfit(log_n, log_e, 1)
            ax1.annotate(f"pente ≈ {slope:.2f}",
                         xy=(ns_csv[len(ns_csv)//2],
                             errs_csv[len(errs_csv)//2]),
                         fontsize=9, color=AMBER,
                         xytext=(20, -20), textcoords="offset points",
                         arrowprops=dict(arrowstyle="->", color=AMBER))
    else:
        ax1.plot(ns_small, errs, "o-", color=BLUE, lw=2,
                 markersize=5, label="Calculé")

    ax1.legend(fontsize=9)
    ax1.axvline(5000,    color=AMBER, lw=0.8, ls=":", alpha=0.7)
    ax1.axvline(1e9,     color=GREEN, lw=0.8, ls=":", alpha=0.7)
    ax1.text(5000*1.2,   ax1.get_ylim()[0]*3, "n=5k",  color=AMBER, fontsize=8)
    ax1.text(1e9*1.2,    ax1.get_ylim()[0]*3, "n=1B", color=GREEN, fontsize=8)

    # Graphique temps vs n
    ax2.set_facecolor(BG2)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("n (nombre de trapèzes)")
    ax2.set_ylabel("Temps de calcul (s)")
    ax2.set_title("Temps de calcul vs n (séquentiel)", pad=6)
    ax2.grid(True, alpha=0.25)
    for sp in ax2.spines.values(): sp.set_edgecolor(BORDER)

    if len(times_csv) > 0:
        ax2.plot(ns_csv, times_csv, "s-", color=GREEN, lw=2,
                 markersize=5, label="Mesuré")
        # Courbe O(n)
        idx = np.argmin(np.abs(ns_csv - 1000))
        scale = times_csv[idx] / ns_csv[idx]
        ax2.plot(ns_th, scale * ns_th, "--", color=MUTED, lw=1.2,
                 label="Théorique O(n)")
    ax2.legend(fontsize=9)


# ── Onglet 3 : Scalabilité ───────────────────────────────────────────
def build_tab_scalability(ax1, ax2):
    # Données CSV
    Ps, times, speedups = [], [], []
    t_seq = None
    csv_s = "results_mpi_static.csv"

    if os.path.exists(csv_s):
        df = pd.read_csv(csv_s)
        df_big = df[df["n"] == 1_000_000_000]
        if len(df_big) > 0:
            Ps      = df_big["P"].values
            times   = df_big["temps_s"].values
            speedups= df_big["speedup"].values
            t_seq   = df_big["temps_s"].iloc[0] if df_big["P"].iloc[0]==1 else None

    if len(Ps) == 0:
        # Données fictives si CSV absent
        Ps      = np.array([1, 2, 4])
        times   = np.array([0.72, 0.41, 0.24])
        speedups= 0.72 / times
        t_seq   = 0.72

    # Amdahl's law (s = fraction séquentielle ≈ 15% pour ce programme)
    P_range  = np.linspace(1, max(Ps)*1.5 + 1, 200)
    alpha    = 0.12   # fraction séquentielle estimée
    amdahl   = 1.0 / (alpha + (1 - alpha) / P_range)
    ideal    = P_range

    # Graphique speedup
    ax1.set_facecolor(BG2)
    ax1.grid(True, alpha=0.25)
    for sp in ax1.spines.values(): sp.set_edgecolor(BORDER)
    ax1.set_xlabel("P (nombre de processus)")
    ax1.set_ylabel("Speedup S(P) = T_seq / T_mpi(P)")
    ax1.set_title("Scalabilité MPI statique — Speedup vs P", pad=6)

    ax1.plot(P_range, ideal,   "--", color=MUTED, lw=1.2, label="Idéal S=P")
    ax1.plot(P_range, amdahl,  "-.", color=AMBER, lw=1.5,
             label=f"Amdahl α={alpha:.0%}")
    ax1.axhline(1, color=MUTED, lw=0.6, alpha=0.5)

    if len(Ps) > 0:
        ax1.plot(Ps, speedups, "o-", color=BLUE, lw=2.5, markersize=8,
                 label="MPI statique (mesuré)")
        for p, s in zip(Ps, speedups):
            ax1.annotate(f"×{s:.2f}", xy=(p, s), fontsize=9, color=BLUE,
                         xytext=(5, 8), textcoords="offset points")

    ax1.legend(fontsize=9)
    ax1.set_xlim(0.5, max(Ps)*1.2 + 0.5)

    # Graphique temps
    ax2.set_facecolor(BG2)
    ax2.grid(True, alpha=0.25, axis="y")
    for sp in ax2.spines.values(): sp.set_edgecolor(BORDER)
    ax2.set_xlabel("P (nombre de processus)")
    ax2.set_ylabel("Temps de calcul (s)")
    ax2.set_title("Temps de calcul — n = 1 milliard", pad=6)

    cols = [GREEN if s >= 1.3 else AMBER if s >= 1.0 else RED
            for s in speedups]
    bars = ax2.bar(Ps, times, color=cols, alpha=0.85, width=0.5,
                   edgecolor=BG, linewidth=0.8)
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{t:.3f}s", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax2.set_xticks(Ps)
    ax2.set_xticklabels([f"P={p}" for p in Ps])

    handles = [mpatches.Patch(color=c, label=l) for c, l in
               [(GREEN,"S≥×1.3"), (AMBER,"S≥×1.0"), (RED,"S<×1.0")]]
    ax2.legend(handles=handles, fontsize=8)


# ── Onglet 4 : Déséquilibre ──────────────────────────────────────────
def build_tab_imbalance(ax1, ax2, ax3):
    # Visualisation de la courbe et des zones
    P = 4
    xs = np.linspace(0, 1, 500)
    ax1.set_facecolor(BG2)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.1)
    ax1.set_xlabel("x"); ax1.set_ylabel("f(x)")
    ax1.set_title("Déséquilibre — f décroissante → charges inégales", pad=6)
    ax1.grid(True, alpha=0.2)
    for sp in ax1.spines.values(): sp.set_edgecolor(BORDER)

    PCOLS = [BLUE, GREEN, AMBER, RED]
    integrals = []
    for r in range(P):
        a, b   = r/P, (r+1)/P
        x_zone = np.linspace(a, b, 200)
        ax1.fill_between(x_zone, f(x_zone), alpha=0.45, color=PCOLS[r],
                         label=f"Rang {r}")
        integ  = np.trapezoid(f(x_zone), x_zone)
        integrals.append(integ)
        ax1.text((a+b)/2, 0.05, f"Rang {r}\n∫={integ:.3f}",
                 ha="center", fontsize=8, color=PCOLS[r])

    ax1.plot(xs, f(xs), color=CYAN, lw=2, zorder=5)
    ax1.legend(fontsize=8, loc="upper right")

    # Barres de timing statique (avec sleep)
    sf       = 0.5   # sleep factor
    max_int  = max(integrals)
    times_static  = [1.0 + i*sf for i in integrals]   # calcul + sleep
    max_static     = max(times_static)

    ax2.set_facecolor(BG2)
    ax2.set_title("MPI statique — timeline (avec sleep)", pad=6)
    ax2.set_xlabel("Temps (s)")
    ax2.set_xlim(0, max_static * 1.15)
    ax2.set_yticks(range(P))
    ax2.set_yticklabels([f"Rang {r}" for r in range(P)])
    ax2.grid(True, alpha=0.2, axis="x")
    for sp in ax2.spines.values(): sp.set_edgecolor(BORDER)

    for r in range(P):
        calc_t  = 1.0
        sleep_t = integrals[r] * sf
        ax2.barh(r, calc_t,              color=PCOLS[r], alpha=0.8, height=0.5)
        ax2.barh(r, sleep_t, left=calc_t, color=PCOLS[r], alpha=0.35,
                 height=0.5, hatch="///", edgecolor="white", linewidth=0.3)
        ax2.text(calc_t + sleep_t + 0.01, r,
                 f"  {calc_t+sleep_t:.2f}s", va="center", fontsize=8, color=PCOLS[r])

    ax2.axvline(max_static, color=RED, lw=1.5, ls="--",
                label=f"Durée totale = {max_static:.2f}s")
    calc_patch  = mpatches.Patch(color="gray", alpha=0.8, label="Calcul")
    sleep_patch = mpatches.Patch(color="gray", alpha=0.35, hatch="///",
                                  edgecolor="white", label="Sleep")
    ax2.legend(handles=[calc_patch, sleep_patch,
               Line2D([0],[0], color=RED, ls="--", label="Durée totale")],
               fontsize=8)

    # Barres de timing dynamique (maître-esclave équilibré)
    workers    = P - 1
    total_work = sum(integrals)
    work_each  = total_work / workers
    times_dyn  = [1.0 + work_each * sf] * workers
    max_dyn    = max(times_dyn)

    ax3.set_facecolor(BG2)
    ax3.set_title("MPI dynamique — timeline (équilibré)", pad=6)
    ax3.set_xlabel("Temps (s)")
    ax3.set_xlim(0, max_static * 1.15)
    ax3.set_yticks(range(workers + 1))
    ax3.set_yticklabels(["Maître"] + [f"Esclave {r+1}" for r in range(workers)])
    ax3.grid(True, alpha=0.2, axis="x")
    for sp in ax3.spines.values(): sp.set_edgecolor(BORDER)

    # Maître
    ax3.barh(0, max_dyn, color=PURPLE, alpha=0.5, height=0.5,
             label="Maître (distribue)")
    ax3.text(max_dyn + 0.01, 0, "  distribution", va="center",
             fontsize=8, color=PURPLE)

    # Esclaves équilibrés
    for w in range(workers):
        ax3.barh(w+1, 1.0,      color=PCOLS[w], alpha=0.8, height=0.5)
        ax3.barh(w+1, work_each*sf, left=1.0, color=PCOLS[w], alpha=0.35,
                 height=0.5, hatch="///", edgecolor="white", linewidth=0.3)
        ax3.text(1.0 + work_each*sf + 0.01, w+1,
                 f"  {1.0+work_each*sf:.2f}s", va="center",
                 fontsize=8, color=PCOLS[w])

    ax3.axvline(max_dyn, color=GREEN, lw=1.5, ls="--",
                label=f"Durée totale = {max_dyn:.2f}s")
    ax3.legend(fontsize=8)

    # Annotation gain
    gain = max_static / max_dyn
    ax3.text(0.98, 0.05,
             f"Gain : ×{gain:.2f}",
             transform=ax3.transAxes, fontsize=11, fontweight="bold",
             color=GREEN, ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BG3,
                       edgecolor=GREEN, alpha=0.9))


# ── Onglet 5 : Dashboard ─────────────────────────────────────────────
def build_tab_dashboard(ax):
    ax.set_facecolor(BG2)
    ax.axis("off")

    lines = ["RÉSULTATS — Calcul de π, méthode des trapèzes", ""]

    def load(path):
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    df_s   = load("results_seq.csv")
    df_st  = load("results_mpi_static.csv")
    df_dyn = load("results_mpi_dynamic.csv")

    if df_s is not None:
        lines += ["── Programme 1 : Séquentiel ──────────────────────────────"]
        lines += [f"  {'n':>12}  {'π approx':>18}  {'erreur':>10}  {'temps':>10}"]
        for _, row in df_s.iterrows():
            lines += [f"  {int(row['n']):>12,}  {row['pi_approx']:>18.12f}"
                      f"  {row['erreur_abs']:>10.2e}  {row['temps_s']:>9.4f}s"]
        lines += [""]

    if df_st is not None:
        lines += ["── Programme 2 : MPI statique (n = 1 milliard) ──────────"]
        lines += [f"  {'P':>4}  {'temps':>10}  {'speedup':>9}  {'π approx':>18}"]
        df_big = df_st[df_st["n"] == 1_000_000_000]
        for _, row in df_big.iterrows():
            spd = f"×{row['speedup']:.3f}" if "speedup" in row else "—"
            lines += [f"  {int(row['P']):>4}  {row['temps_s']:>9.4f}s"
                      f"  {spd:>9}  {row['pi_approx']:>18.12f}"]
        lines += [""]

    if df_dyn is not None:
        lines += ["── Programme 3 : MPI dynamique ───────────────────────────"]
        lines += [f"  {'P':>4}  {'ntasks':>7}  {'temps':>10}  {'gain':>8}"]
        for _, row in df_dyn.iterrows():
            spd = f"×{row['speedup_vs_static']:.3f}" if "speedup_vs_static" in row else "—"
            lines += [f"  {int(row['P']):>4}  {int(row['ntasks']):>7}"
                      f"  {row['temps_s']:>9.4f}s  {spd:>8}"]
        lines += [""]

    lines += [f"  Référence : π = {PI_REF:.15f}"]

    ax.text(0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, fontsize=9,
            va="top", ha="left", color=TEXT,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=BG3,
                      edgecolor=BORDER, alpha=0.95))


# ── Construction de la figure principale ─────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Viewer Pi déterministe — M1 CHPS")
    parser.add_argument("--tab", default="all",
                        choices=["all","trapeze","convergence",
                                 "speedup","dynamic","dashboard"])
    args = parser.parse_args()

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.canvas.manager.set_window_title(
        "Partie 1 — Calcul de π · Méthode des trapèzes · M1 CHPS UPVD")

    fig.text(0.5, 0.98,
             "Calcul de π — Méthode des trapèzes  |  M1 CHPS UPVD  |  "
             "π = 4 × ∫₀¹ 1/(1+x²) dx",
             ha="center", va="top", fontsize=11, fontweight="bold",
             color=TEXT)

    # GridSpec principal
    outer = gridspec.GridSpec(1, 2, figure=fig,
                              left=0.05, right=0.97,
                              top=0.94, bottom=0.08,
                              wspace=0.30)

    # Colonne gauche : trapèzes interactifs
    gs_left  = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0], hspace=0.08, height_ratios=[1, 0.05])
    build_tab_trapeze(fig, gs_left)

    # Colonne droite : 4 sous-graphiques
    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, 2, subplot_spec=outer[1], hspace=0.55, wspace=0.35)

    # Convergence
    ax_c1 = fig.add_subplot(gs_right[0, 0])
    ax_c2 = fig.add_subplot(gs_right[0, 1])
    build_tab_convergence(ax_c1, ax_c2)

    # Scalabilité
    ax_s1 = fig.add_subplot(gs_right[1, 0])
    ax_s2 = fig.add_subplot(gs_right[1, 1])
    build_tab_scalability(ax_s1, ax_s2)

    # Déséquilibre (3 sous-graphiques en ligne du bas)
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_right[2, :], wspace=0.40)
    ax_i1 = fig.add_subplot(gs_bot[0])
    ax_i2 = fig.add_subplot(gs_bot[1])
    ax_i3 = fig.add_subplot(gs_bot[2])
    build_tab_imbalance(ax_i1, ax_i2, ax_i3)

    # Barre du bas
    fig.text(0.05, 0.025,
             "Curseur gauche : faire varier n trapèzes en temps réel  |  "
             "CSV : results_seq.csv · results_mpi_static.csv · results_mpi_dynamic.csv",
             fontsize=8, color=MUTED)
    fig.text(0.97, 0.025,
             f"π_ref = {PI_REF:.10f}",
             fontsize=8, color=MUTED, ha="right",
             family="monospace")

    fig.canvas.mpl_connect("key_press_event",
                           lambda e: plt.close() if e.key == "q" else None)
    plt.show()


if __name__ == "__main__":
    main()
