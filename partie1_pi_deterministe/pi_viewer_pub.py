#!/usr/bin/env python3
"""
pi_viewer_pub.py — 2 figures publication separees

  Figure 1 — pi_fig1_methode_convergence.pdf/png
    (a) Methode des trapezes
    (b) Convergence erreur vs n
    (c) Temps sequentiel vs n

  Figure 2 — pi_fig2_scalabilite_gantt.pdf/png
    (d) Speedup S(P) + table
    (e) Efficacite parallele E(P)
    (f) Chronogramme Gantt statique vs dynamique

M1 CHPS UPVD — YAYA TOURE · Mamadou G Diallo
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     "#1a1a1a",
    "axes.linewidth":     1.0,
    "axes.labelcolor":    "#111111",
    "axes.labelsize":     12,
    "axes.titlesize":     12,
    "axes.titleweight":   "semibold",
    "axes.titlelocation": "left",
    "xtick.color":        "#1a1a1a",
    "ytick.color":        "#1a1a1a",
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
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
    "legend.fontsize":    10.5,
    "grid.color":         "#dedede",
    "grid.linewidth":     0.5,
    "grid.linestyle":     "--",
    "font.family":        "sans-serif",
    "font.size":          11,
    "lines.linewidth":    2.0,
    "lines.markersize":   7,
    "figure.dpi":         130,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "text.usetex":        False,
})

# Palette colorblind-safe (Wong 2011)
C1 = "#0072B2"
C2 = "#D55E00"
C3 = "#009E73"
C4 = "#CC79A7"
C5 = "#56B4E9"

PI_REF = 3.141592653589793

def f(x):
    return 1.0 / (1.0 + x * x)

# ── Donnees mesurees sur HP 15s-eq0xxx ───────────────────────────────
T_SEQ  = 5.637621

Ps_s   = np.array([1,        2,        4       ])
Ts_s   = np.array([5.579020, 2.808411, 1.756244])
Ss_s   = T_SEQ / Ts_s
Eff_s  = Ss_s / Ps_s

Ns_conv = np.array([10, 100, 1000, 10000, 100000,
                    1000000, 10000000, 100000000, 1000000000])
Errs    = np.array([1.67e-3, 1.67e-5, 1.67e-7, 6.67e-9, 6.67e-11,
                    6.67e-13, 6.67e-15, 4.44e-16, 4.44e-16])

Ns_t    = np.array([5000, 100000, 1000000, 10000000, 100000000, 1000000000])
Ts_ref  = np.array([0.000029, 0.000580, 0.005800,
                    0.058000, 0.580000, 5.637621])


def style_ax(ax):
    ax.grid(True, which="major", alpha=0.40, zorder=0)
    ax.grid(True, which="minor", alpha=0.15, zorder=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.9)

def panel_label(ax, letter):
    ax.text(-0.14, 1.06, "(" + letter + ")",
            transform=ax.transAxes,
            fontsize=16, fontweight="bold",
            va="top", ha="left", color="#111111")

def integ_num(a, b, N=100000):
    xs = np.linspace(a, b, N)
    return np.trapezoid(f(xs), xs)

FOOTER = (
    "YAYA TOURE  ·  Mamadou G Diallo"
    "  |  M1 CHPS — Universite de Perpignan Via Domitia (UPVD)"
    "  |  Encadrant : B. Antunes"
)


# ════════════════════════════════════════════════════════════════════
#
#  FIGURE 1 — Methode, Convergence, Temps
#
# ════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(
    1, 3, figsize=(16, 6),
    gridspec_kw={"wspace": 0.40}
)
fig1.patch.set_facecolor("white")

fig1.text(
    0.5, 0.98,
    "Calcul de pi — Methode des trapezes composites",
    ha="center", va="top",
    fontsize=16, fontweight="bold", color="#111111"
)
fig1.text(
    0.5, 0.93,
    "pi = 4 x integral(0,1)  1/(1+x^2) dx     "
    "Sommation de Kahan     "
    "Machine : HP 15s-eq0xxx,  4 coeurs logiques",
    ha="center", va="top",
    fontsize=10.5, color="#555555", style="italic"
)

# ── (a) Illustration trapèzes ────────────────────────────────────────
ax_a = axes1[0]
panel_label(ax_a, "a")
ax_a.set_title("Methode des trapezes composites", pad=8)
ax_a.set_xlabel("x", labelpad=5)
ax_a.set_ylabel("f(x) = 1 / (1 + x\u00b2)", labelpad=5)

xs = np.linspace(0, 1, 500)
ax_a.plot(xs, f(xs), color=C1, lw=2.4, label="f(x)", zorder=5)
ax_a.fill_between(xs, f(xs), alpha=0.06, color=C1)

n_trap  = 8
dx_trap = 1.0 / n_trap
for i in range(n_trap):
    xi, xi1 = i * dx_trap, (i + 1) * dx_trap
    col = C2 if i < n_trap // 2 else C3
    ax_a.fill_between([xi, xi1], [f(xi), f(xi1)],
                       alpha=0.55, color=col,
                       edgecolor="#333333", linewidth=0.6, zorder=3)

ax_a.text(0.50, 0.70,
          "pi/4 = integral(0,1) f(x) dx",
          transform=ax_a.transAxes,
          fontsize=10.5, color=C1, ha="center",
          bbox=dict(boxstyle="round,pad=0.4",
                    facecolor="white", edgecolor=C1, alpha=0.95))
ax_a.text(0.50, 0.09, "n = 8 trapezes",
          transform=ax_a.transAxes,
          fontsize=10, color="#444444", ha="center")

ax_a.set_xlim(-0.02, 1.05)
ax_a.set_ylim(-0.05, 1.18)
ax_a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_a.legend(loc="upper right", fontsize=10.5)
style_ax(ax_a)

# Legende couleurs
p_gauche = mpatches.Patch(color=C2, alpha=0.7, label="Rangs gauche (grande valeur)")
p_droite = mpatches.Patch(color=C3, alpha=0.7, label="Rangs droite (petite valeur)")
ax_a.legend(handles=[Line2D([0],[0], color=C1, lw=2.2, label="f(x)"),
                      p_gauche, p_droite],
            fontsize=9, loc="upper right")

# ── (b) Convergence ──────────────────────────────────────────────────
ax_b = axes1[1]
panel_label(ax_b, "b")
ax_b.set_title("Convergence — erreur absolue vs n", pad=8)
ax_b.set_xlabel("n  (nombre de trapezes)", labelpad=5)
ax_b.set_ylabel("|pi_n  -  pi_ref|", labelpad=5)
ax_b.set_xscale("log")
ax_b.set_yscale("log")

n_th  = np.logspace(1, 10, 300)
scale = Errs[3] * (Ns_conv[3] ** 2)
ax_b.plot(n_th, scale / n_th ** 2, "--",
          color="#aaaaaa", lw=1.4, label="O(n^-2)  theorique")
ax_b.plot(Ns_conv, Errs, "o-",
          color=C1, lw=2.0, markersize=7,
          markerfacecolor="white", markeredgewidth=1.8,
          label="Sommation Kahan")

for n_m, col, lab in [(5000, C2, "n = 5 000"),
                       (1e9,  C3, "n = 10^9 ")]:
    ax_b.axvline(n_m, color=col, lw=1.0, ls=":", alpha=0.85)
    idx = np.argmin(np.abs(Ns_conv - n_m))
    ax_b.annotate(lab,
                  xy=(n_m, Errs[idx]),
                  xytext=(n_m * 0.18, Errs[idx] * 18),
                  fontsize=10, color=col,
                  arrowprops=dict(arrowstyle="-", color=col, lw=0.8))

# Annotations valeurs cles
ax_b.annotate("n=5k\nErr=6.7e-9",
              xy=(5000, 6.67e-9),
              xytext=(800, 5e-7),
              fontsize=9, color=C2,
              arrowprops=dict(arrowstyle="->", color=C2, lw=0.8))
ax_b.annotate("n=1B\nErr=4.4e-16",
              xy=(1e9, 4.44e-16),
              xytext=(5e6, 1e-14),
              fontsize=9, color=C3,
              arrowprops=dict(arrowstyle="->", color=C3, lw=0.8))

ax_b.legend(loc="lower left", fontsize=10.5)
style_ax(ax_b)

# ── (c) Temps vs n ───────────────────────────────────────────────────
ax_c = axes1[2]
panel_label(ax_c, "c")
ax_c.set_title("Temps de calcul sequentiel vs n", pad=8)
ax_c.set_xlabel("n  (nombre de trapezes)", labelpad=5)
ax_c.set_ylabel("T  (secondes)", labelpad=5)
ax_c.set_xscale("log")
ax_c.set_yscale("log")

n_th2 = np.logspace(3, 9.5, 200)
st    = Ts_ref[2] / Ns_t[2]
ax_c.plot(n_th2, st * n_th2, "--",
          color="#aaaaaa", lw=1.4, label="O(n)  theorique")
ax_c.plot(Ns_t, Ts_ref, "s-",
          color=C2, lw=2.0, markersize=7,
          markerfacecolor="white", markeredgewidth=1.8,
          label="Mesure  (HP 15s)")

ax_c.axvline(5000, color=C2, lw=1.0, ls=":", alpha=0.75)
ax_c.axvline(1e9,  color=C3, lw=1.0, ls=":", alpha=0.75)

ax_c.annotate("n=5k\nT=29 us",
              xy=(5000, 0.000029),
              xytext=(700, 0.00045),
              fontsize=9, color=C2,
              arrowprops=dict(arrowstyle="->", color=C2, lw=0.8))
ax_c.annotate("n=1B\nT=5.64 s",
              xy=(1e9, 5.637621),
              xytext=(1e7, 0.9),
              fontsize=9, color=C3,
              arrowprops=dict(arrowstyle="->", color=C3, lw=0.8))

ax_c.legend(loc="upper left", fontsize=10.5)
style_ax(ax_c)

fig1.text(0.5, 0.01, FOOTER,
          ha="center", va="bottom",
          fontsize=9, color="#777777")

fig1.subplots_adjust(top=0.88, bottom=0.12)
fig1.savefig("pi_fig1_methode_convergence.png", dpi=200,
             bbox_inches="tight", facecolor="white")
fig1.savefig("pi_fig1_methode_convergence.pdf", dpi=300,
             bbox_inches="tight", facecolor="white")
print("Figure 1 sauvegardee :")
print("  pi_fig1_methode_convergence.png")
print("  pi_fig1_methode_convergence.pdf")


# ════════════════════════════════════════════════════════════════════
#
#  FIGURE 2 — Scalabilite, Efficacite, Gantt
#
# ════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(16, 10))
fig2.patch.set_facecolor("white")

fig2.text(
    0.5, 0.98,
    "Calcul de pi — Parallelisation MPI et equilibrage de charge",
    ha="center", va="top",
    fontsize=16, fontweight="bold", color="#111111"
)
fig2.text(
    0.5, 0.94,
    "n = 1 milliard de trapezes     "
    "Machine : HP 15s-eq0xxx,  4 coeurs logiques     "
    "T_seq = 5.638 s",
    ha="center", va="top",
    fontsize=10.5, color="#555555", style="italic"
)

gs2 = gridspec.GridSpec(
    2, 3, figure=fig2,
    left=0.08, right=0.97,
    top=0.90, bottom=0.09,
    hspace=0.55, wspace=0.40
)

# ── (d) Speedup ──────────────────────────────────────────────────────
ax_d = fig2.add_subplot(gs2[0, :2])
panel_label(ax_d, "d")
ax_d.set_title(
    "Scalabilite MPI statique   S(P) = T_seq / T_MPI(P)",
    pad=8)
ax_d.set_xlabel("P  (nombre de processus MPI)", labelpad=5)
ax_d.set_ylabel("Speedup  S(P)", labelpad=5)

P_range = np.linspace(1, 5.5, 200)
alpha   = 0.03
ax_d.plot(P_range, P_range, ":",
          color="#aaaaaa", lw=1.4, label="Ideal   S = P")
ax_d.plot(P_range, 1.0 / (alpha + (1.0 - alpha) / P_range),
          "-.", color=C4, lw=1.6,
          label="Loi d Amdahl   (alpha = 3 %)")
ax_d.plot(Ps_s, Ss_s, "o-",
          color=C1, lw=2.4, markersize=9,
          markerfacecolor="white", markeredgewidth=2.2,
          label="MPI statique   (mesure)")

for p, s in zip(Ps_s, Ss_s):
    ax_d.annotate("x" + f"{s:.2f}",
                  xy=(p, s), xytext=(p + 0.12, s + 0.12),
                  fontsize=11.5, color=C1, fontweight="bold")

ax_d.axhline(1.0, color="#dddddd", lw=0.8)
ax_d.set_xlim(0.6, 5.2)
ax_d.set_ylim(0.0, 5.2)
ax_d.set_xticks([1, 2, 4])
ax_d.set_xticklabels(["P = 1", "P = 2", "P = 4"], fontsize=11)
ax_d.legend(loc="upper left", fontsize=11)

# Table
rows = [["P = 1", f"{Ts_s[0]:.2f} s", f"{Ss_s[0]:.2f}", f"{Eff_s[0]:.3f}"],
        ["P = 2", f"{Ts_s[1]:.2f} s", f"{Ss_s[1]:.2f}", f"{Eff_s[1]:.3f}"],
        ["P = 4", f"{Ts_s[2]:.2f} s", f"{Ss_s[2]:.2f}", f"{Eff_s[2]:.3f}"]]
tbl = ax_d.table(cellText=rows,
                 colLabels=["P", "T (s)", "S(P)", "E(P)"],
                 cellLoc="center", loc="lower right",
                 bbox=[0.68, 0.06, 0.30, 0.62])
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    cell.set_linewidth(0.5)
    cell.set_height(0.18)
    if r == 0:
        cell.set_facecolor("#dde8f5")
        cell.set_text_props(fontweight="bold", fontsize=11)
    else:
        cell.set_facecolor("white" if r % 2 else "#f5f8fc")
        cell.set_text_props(fontsize=11)
style_ax(ax_d)

# ── (e) Efficacite ───────────────────────────────────────────────────
ax_e = fig2.add_subplot(gs2[0, 2])
panel_label(ax_e, "e")
ax_e.set_title("Efficacite parallele   E(P) = S(P) / P", pad=8)
ax_e.set_xlabel("P  (processus)", labelpad=5)
ax_e.set_ylabel("E(P)", labelpad=5)

ax_e.axhline(1.0, color="#cccccc", lw=1.2, ls="--", label="Ideal   E = 1")
ax_e.axhline(0.5, color="#eeeeee", lw=0.8, ls=":")
ax_e.plot(Ps_s, Eff_s, "o-",
          color=C1, lw=2.4, markersize=9,
          markerfacecolor="white", markeredgewidth=2.2,
          label="MPI statique")

for p, e in zip(Ps_s, Eff_s):
    off = 0.016
    ax_e.annotate(f"{e:.3f}",
                  xy=(p, e), xytext=(p + 0.12, e + off),
                  fontsize=11.5, color=C1, fontweight="bold")

ax_e.set_ylim(0.45, 1.20)
ax_e.set_xlim(0.6, 4.8)
ax_e.set_xticks([1, 2, 4])
ax_e.set_xticklabels(["P = 1", "P = 2", "P = 4"], fontsize=11)
ax_e.legend(loc="lower left", fontsize=11)
style_ax(ax_e)

# ── (f) Chronogramme Gantt ───────────────────────────────────────────
ax_f = fig2.add_subplot(gs2[1, :])
panel_label(ax_f, "f")
ax_f.set_title(
    "Desequilibre de charge — MPI statique  vs  MPI dynamique     "
    "P = 4     n = 1 milliard     sleep proportionnel a l integrale partielle",
    pad=10)
ax_f.set_xlabel("Temps  (secondes)", labelpad=6)

P      = 4
sf     = 0.5
T_calc = T_SEQ / P

integrals = [integ_num(r / P, (r + 1) / P) for r in range(P)]
T_sleep_s = [v * sf for v in integrals]
T_total_s = [T_calc + s for s in T_sleep_s]
T_max_s   = max(T_total_s)
workers   = P - 1
each_int  = sum(integrals) / workers
T_each_d  = T_calc + each_int * sf
T_max_d   = T_each_d

CRANK = [C1, C2, C3, C5]
BAR_H = 0.60

y_stat = [8.2, 6.8, 5.4, 4.0]
y_sep  = 3.3
y_mast = 2.5
y_dyn  = [1.7, 0.9, 0.1]

# Statique
for r in range(P):
    y = y_stat[r]
    ax_f.barh(y, T_calc, height=BAR_H,
              color=CRANK[r], alpha=0.88, edgecolor="none", zorder=3)
    ax_f.barh(y, T_sleep_s[r], height=BAR_H, left=T_calc,
              color=CRANK[r], alpha=0.28,
              hatch="///", edgecolor="white", linewidth=0.5, zorder=3)
    ax_f.text(-0.09, y, "Rang " + str(r),
              va="center", ha="right", fontsize=11.5, color="#222222")
    ax_f.text(T_total_s[r] + T_max_s * 0.007, y,
              f"{T_total_s[r]:.2f} s",
              va="center", fontsize=11, color=CRANK[r], fontweight="bold")

# Separateur + etiquettes
ax_f.axhline(y_sep, color="#bbbbbb", lw=1.0, ls="--")
ax_f.text(T_max_s * 0.004, y_sep + 0.28,
          "Statique  (desequilibre)",
          fontsize=11.5, color="#444444", style="italic", fontweight="semibold")
ax_f.text(T_max_s * 0.004, y_sep - 0.60,
          "Dynamique  (maitre-esclave)",
          fontsize=11.5, color="#444444", style="italic", fontweight="semibold")

# Maitre
ax_f.barh(y_mast, T_max_d, height=BAR_H,
          color="#aaaaaa", alpha=0.40, edgecolor="none", zorder=3)
ax_f.text(-0.09, y_mast, "Maitre",
          va="center", ha="right", fontsize=11.5, color="#555555")
ax_f.text(T_max_d * 0.45, y_mast,
          "distribution des taches",
          va="center", ha="center", fontsize=10.5, color="#666666")

# Esclaves
for w in range(workers):
    y = y_dyn[w]
    ax_f.barh(y, T_calc, height=BAR_H,
              color=CRANK[w + 1], alpha=0.88, edgecolor="none", zorder=3)
    ax_f.barh(y, each_int * sf, height=BAR_H, left=T_calc,
              color=CRANK[w + 1], alpha=0.28,
              hatch="///", edgecolor="white", linewidth=0.5, zorder=3)
    ax_f.text(-0.09, y, "Esclave " + str(w + 1),
              va="center", ha="right", fontsize=11.5, color="#222222")
    ax_f.text(T_each_d + T_max_s * 0.007, y,
              f"{T_each_d:.2f} s",
              va="center", fontsize=11, color=CRANK[w + 1], fontweight="bold")

# Lignes de fin
ax_f.axvline(T_max_s, color=C2, lw=2.0, ls="--",
             label="Fin statique    " + f"{T_max_s:.2f} s")
ax_f.axvline(T_max_d, color=C3, lw=2.0, ls="--",
             label="Fin dynamique  " + f"{T_max_d:.2f} s")

# Double fleche gain
y_arrow = -0.55
gain = T_max_s / T_max_d
ax_f.annotate("", xy=(T_max_d, y_arrow), xytext=(T_max_s, y_arrow),
              arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.4))
ax_f.text((T_max_s + T_max_d) / 2, y_arrow - 0.70,
          "Gain = x" + f"{gain:.2f}",
          ha="center", fontsize=13, color="#111111", fontweight="bold")

# Legende
calc_p  = mpatches.Patch(facecolor="#777777", alpha=0.88,
                          label="Calcul")
sleep_p = mpatches.Patch(facecolor="#777777", alpha=0.28,
                          hatch="///", edgecolor="white",
                          label="Sleep  (proportionnel a l integrale)")
ax_f.legend(
    handles=[
        calc_p, sleep_p,
        Line2D([0], [0], color=C2, ls="--", lw=2.0,
               label="Fin statique    " + f"{T_max_s:.2f} s"),
        Line2D([0], [0], color=C3, ls="--", lw=2.0,
               label="Fin dynamique  " + f"{T_max_d:.2f} s"),
    ],
    fontsize=10.5, loc="lower right", ncol=2, columnspacing=1.5)

ax_f.set_xlim(-0.01, T_max_s * 1.25)
ax_f.set_ylim(-1.6, 9.2)
ax_f.set_yticks([])
ax_f.grid(True, which="major", axis="x", alpha=0.35, zorder=0)
for sp in ["top", "right", "left"]:
    ax_f.spines[sp].set_visible(False)
ax_f.spines["bottom"].set_linewidth(0.8)
ax_f.tick_params(axis="x", labelsize=11)

fig2.text(0.5, 0.01, FOOTER,
          ha="center", va="bottom",
          fontsize=9, color="#777777")

fig2.savefig("pi_fig2_scalabilite_gantt.png", dpi=200,
             bbox_inches="tight", facecolor="white")
fig2.savefig("pi_fig2_scalabilite_gantt.pdf", dpi=300,
             bbox_inches="tight", facecolor="white")
print("Figure 2 sauvegardee :")
print("  pi_fig2_scalabilite_gantt.png")
print("  pi_fig2_scalabilite_gantt.pdf")

plt.show()