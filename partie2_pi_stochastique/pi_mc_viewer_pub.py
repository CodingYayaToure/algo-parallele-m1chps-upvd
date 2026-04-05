#!/usr/bin/env python3
"""
pi_mc_viewer_pub.py — 2 figures publication separees — Monte Carlo MPI
M1 CHPS UPVD — YAYA TOURE · Mamadou G Diallo

  Figure 1 — pi_mc_fig1_principe_convergence.png / .pdf
    (a) Illustration Monte Carlo (points dans/hors du cercle)
    (b) Convergence erreur vs N

  Figure 2 — pi_mc_fig2_scalabilite.png / .pdf
    (d) N_total vs nombre de clients
    (e) Erreur vs nombre de clients
    (f) Scalabilite — N_total en 10 secondes selon P
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     "#1a1a1a",
    "axes.linewidth":     1.0,
    "axes.labelcolor":    "#111111",
    "axes.labelsize":     13,
    "axes.titlesize":     13,
    "axes.titleweight":   "semibold",
    "axes.titlelocation": "left",
    "xtick.color":        "#1a1a1a",
    "ytick.color":        "#1a1a1a",
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   5,
    "ytick.major.size":   5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "text.color":         "#111111",
    "legend.frameon":     True,
    "legend.framealpha":  0.95,
    "legend.edgecolor":   "#cccccc",
    "legend.fontsize":    11,
    "grid.color":         "#dedede",
    "grid.linewidth":     0.5,
    "grid.linestyle":     "--",
    "font.family":        "sans-serif",
    "font.size":          12,
    "lines.linewidth":    2.0,
    "lines.markersize":   7,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "text.usetex":        False,
})

C1 = "#0072B2"
C2 = "#D55E00"
C3 = "#009E73"
C4 = "#CC79A7"
C5 = "#56B4E9"
PI_REF = 3.141592653589793

# ── Donnees mesurees (HP 15s, xorshift64 seed=42) ────────────────────
Ns_conv = np.array([100, 1000, 10000, 100000, 1000000,
                    10000000, 100000000, 1000000000])
Errs    = np.array([5.84e-2, 7.04e-2, 2.28e-2, 9.39e-3, 2.59e-4,
                    2.01e-4, 1.31e-4, 7.66e-5])

DEBIT_CLIENT = 262.0    # M pts/s par client (mesure sequentielle HP 15s)
DURATION     = 10.0     # secondes

FOOTER = (
    "YAYA TOURE  ·  Mamadou G Diallo"
    "  |  M1 CHPS — Universite de Perpignan Via Domitia (UPVD)"
    "  |  Encadrant : B. Antunes"
)


def style_ax(ax):
    ax.grid(True, which="major", alpha=0.40, zorder=0)
    ax.grid(True, which="minor", alpha=0.15, zorder=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.9)

def panel_label(ax, letter):
    ax.text(-0.12, 1.06, "(" + letter + ")",
            transform=ax.transAxes,
            fontsize=18, fontweight="bold",
            va="top", ha="left", color="#111111")


# ════════════════════════════════════════════════════════════════
# FIGURE 1 — Principe + Convergence  (2 panneaux)
# ════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 2, figsize=(14, 7),
                           gridspec_kw={"wspace": 0.38})
fig1.patch.set_facecolor("white")
fig1.text(0.5, 0.98,
          "Calcul de pi — Methode de Monte Carlo",
          ha="center", va="top",
          fontsize=17, fontweight="bold", color="#111111")
fig1.text(0.5, 0.93,
          "pi/4 = N_in / N_total     Convergence en 1/sqrt(N)"
          "     Machine : HP 15s-eq0xxx",
          ha="center", va="top",
          fontsize=11, color="#555555", style="italic")

# ── (a) Illustration Monte Carlo ─────────────────────────────────────
ax = axes[0]
panel_label(ax, "a")
ax.set_title("Principe Monte Carlo", pad=10)
ax.set_xlabel("x", labelpad=6)
ax.set_ylabel("y", labelpad=6)

np.random.seed(42)
N_show = 2000
x = np.random.uniform(0, 1, N_show)
y = np.random.uniform(0, 1, N_show)
inside = x ** 2 + y ** 2 < 1

ax.scatter(x[inside],  y[inside],  s=5, color=C1, alpha=0.65, zorder=3,
           label="N_in = " + str(inside.sum()) + "  (dans le quart de disque)")
ax.scatter(x[~inside], y[~inside], s=5, color=C2, alpha=0.65, zorder=3,
           label="N_out = " + str((~inside).sum()) + "  (hors du disque)")

theta = np.linspace(0, np.pi / 2, 300)
ax.plot(np.cos(theta), np.sin(theta), color="#111111", lw=2.2, zorder=4)
ax.add_patch(mpatches.Rectangle((0, 0), 1, 1,
             fill=False, edgecolor="#555555", lw=1.2))

ax.set_xlim(-0.02, 1.06)
ax.set_ylim(-0.02, 1.06)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_aspect("equal")

pi_est = 4 * inside.sum() / N_show
# Formule en haut a droite
ax.text(0.97, 0.97,
        "pi/4 = N_in / N_total\npi ~ " + f"{pi_est:.4f}" + "  (N = " + str(N_show) + ")",
        transform=ax.transAxes,
        fontsize=10.5, ha="right", va="top", color="#111111",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor=C1, linewidth=1.2, alpha=0.96))

# Legende en bas a gauche
ax.legend(fontsize=10, loc="lower left", markerscale=2.5,
          handletextpad=0.5,
          bbox_to_anchor=(0.01, 0.01), borderaxespad=0.8)

style_ax(ax)

# ── (b) Convergence ──────────────────────────────────────────────────
ax = axes[1]
panel_label(ax, "b")
ax.set_title("Convergence — erreur absolue vs N", pad=10)
ax.set_xlabel("N  (nombre de points)", labelpad=6)
ax.set_ylabel("|pi_N  -  pi_ref|", labelpad=6)
ax.set_xscale("log")
ax.set_yscale("log")

N_th = np.logspace(2, 10, 300)
ax.plot(N_th, 1.0 / np.sqrt(N_th), "--",
        color="#aaaaaa", lw=1.5, label="1/sqrt(N)  theorique")
ax.plot(Ns_conv, Errs, "o-",
        color=C1, lw=2.2, markersize=8,
        markerfacecolor="white", markeredgewidth=2.0,
        label="Mesure  (xorshift64, seed = 42)")

ax.axvline(1e7, color=C2, lw=1.0, ls=":", alpha=0.8)
ax.axvline(1e9, color=C3, lw=1.0, ls=":", alpha=0.8)

ax.annotate("N = 10^7\nErr = 2.01e-4",
            xy=(1e7, 2.01e-4), xytext=(2e5, 3e-3),
            fontsize=10, color=C2,
            arrowprops=dict(arrowstyle="->", color=C2, lw=1.0))
ax.annotate("N = 10^9\nErr = 7.66e-5",
            xy=(1e9, 7.66e-5), xytext=(3e7, 5e-4),
            fontsize=10, color=C3,
            arrowprops=dict(arrowstyle="->", color=C3, lw=1.0))

ax.legend(fontsize=11, loc="lower left")
style_ax(ax)

fig1.text(0.5, 0.01, FOOTER,
          ha="center", va="bottom", fontsize=9.5, color="#777777")
fig1.subplots_adjust(top=0.88, bottom=0.12)
fig1.savefig("pi_mc_fig1_principe_convergence.png", dpi=200,
             bbox_inches="tight", facecolor="white")
fig1.savefig("pi_mc_fig1_principe_convergence.pdf", dpi=300,
             bbox_inches="tight", facecolor="white")
print("Figure 1 sauvegardee :")
print("  pi_mc_fig1_principe_convergence.png")
print("  pi_mc_fig1_principe_convergence.pdf")


# ════════════════════════════════════════════════════════════════
# FIGURE 2 — Scalabilite  (3 panneaux)
# ════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 6),
                            gridspec_kw={"wspace": 0.42})
fig2.patch.set_facecolor("white")
fig2.text(0.5, 0.98,
          "Calcul de pi — Scalabilite Monte Carlo MPI client/serveur",
          ha="center", va="top",
          fontsize=17, fontweight="bold", color="#111111")
fig2.text(0.5, 0.93,
          "Duree fixe = 10 s     N_total = (P-1) x debit_client x duree"
          "     Embarassingly parallel",
          ha="center", va="top",
          fontsize=11, color="#555555", style="italic")

Ps_cl = np.arange(1, 9)
N_M   = Ps_cl * DEBIT_CLIENT * DURATION / 1e6   # en millions

# ── (c) N_total vs clients ───────────────────────────────────────────
ax = axes2[0]
panel_label(ax, "c")
ax.set_title("N_total vs nombre de clients", pad=10)
ax.set_xlabel("Nombre de clients  (P - 1)", labelpad=6)
ax.set_ylabel("N_total  (millions de points)", labelpad=6)
ax.plot(Ps_cl, N_M, "o-", color=C1, lw=2.2, markersize=8,
        markerfacecolor="white", markeredgewidth=2.0,
        label="N_total  (estime)")
ax.plot(Ps_cl, Ps_cl * N_M[0], "--",
        color="#aaaaaa", lw=1.2, label="Lineaire ideal")
for p, n in zip(Ps_cl[:6], N_M[:6]):
    ax.annotate(f"{n:.0f}M",
                xy=(p, n), xytext=(p + 0.06, n + N_M[0] * 0.25),
                fontsize=9.5, color=C1)
ax.set_xlim(0.5, 8.5)
ax.set_xticks(Ps_cl)
ax.legend(fontsize=10.5)
style_ax(ax)

# ── (d) Erreur vs clients ────────────────────────────────────────────
ax = axes2[1]
panel_label(ax, "d")
ax.set_title("Erreur estimee vs nombre de clients", pad=10)
ax.set_xlabel("Nombre de clients  (P - 1)", labelpad=6)
ax.set_ylabel("Erreur  1/sqrt(N_total)", labelpad=6)
ax.set_yscale("log")
Errs_cl = 1.0 / np.sqrt(Ps_cl * DEBIT_CLIENT * 1e6 * DURATION)
ax.plot(Ps_cl, Errs_cl, "s-", color=C2, lw=2.2, markersize=8,
        markerfacecolor="white", markeredgewidth=2.0,
        label="Erreur estimee")
ax.plot(Ps_cl, Errs_cl[0] / np.sqrt(Ps_cl), "--",
        color="#aaaaaa", lw=1.2,
        label="1/sqrt(P-1)  ideal")
for p, e in zip(Ps_cl[:5], Errs_cl[:5]):
    ax.annotate(f"{e:.2e}",
                xy=(p, e), xytext=(p + 0.10, e * 1.5),
                fontsize=9, color=C2)
ax.set_xlim(0.5, 8.5)
ax.set_xticks(Ps_cl)
ax.legend(fontsize=10.5)
style_ax(ax)

# ── (e) N_total en 10s selon P total ────────────────────────────────
ax = axes2[2]
panel_label(ax, "e")
ax.set_title("N_total en 10 s selon P  (embarassingly parallel)", pad=10)
ax.set_xlabel("P  (processus MPI)", labelpad=6)
ax.set_ylabel("N_total  (millions de points)", labelpad=6)
Ps_tot  = np.array([2, 3, 4, 5, 6, 7, 8, 9])
Clients = Ps_tot - 1
N_tot_M = Clients * DEBIT_CLIENT * DURATION / 1e6
cols = [C3 if n >= 1000 else C2 if n >= 500 else C1 for n in N_tot_M]
bars = ax.bar(Ps_tot, N_tot_M, color=cols, alpha=0.85,
              width=0.6, edgecolor="white", linewidth=0.8)
for bar, n in zip(bars, N_tot_M):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + N_tot_M[0] * 0.04,
            f"{n:.0f}M",
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color="#111111")
ax.set_xticks(Ps_tot)
ax.set_xticklabels(["P=" + str(p) for p in Ps_tot], fontsize=10)
handles = [
    mpatches.Patch(color=C1, alpha=0.85, label="< 500M pts"),
    mpatches.Patch(color=C2, alpha=0.85, label="500M - 1B pts"),
    mpatches.Patch(color=C3, alpha=0.85, label="> 1B pts"),
]
ax.legend(handles=handles, fontsize=10)
style_ax(ax)

fig2.text(0.5, 0.01, FOOTER,
          ha="center", va="bottom", fontsize=9.5, color="#777777")
fig2.subplots_adjust(top=0.88, bottom=0.12)
fig2.savefig("pi_mc_fig2_scalabilite.png", dpi=200,
             bbox_inches="tight", facecolor="white")
fig2.savefig("pi_mc_fig2_scalabilite.pdf", dpi=300,
             bbox_inches="tight", facecolor="white")
print("Figure 2 sauvegardee :")
print("  pi_mc_fig2_scalabilite.png")
print("  pi_mc_fig2_scalabilite.pdf")

plt.show()