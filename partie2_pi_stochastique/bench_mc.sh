#!/bin/bash
# bench_mc.sh — Benchmark Partie 2 — compatible toutes machines
export LC_ALL=C

DURATION=10
NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
P_LIST=$(python3 -c "n=$NPROC; print(' '.join(str(p) for p in [2,4,8,16] if p<=max(n,2)))")
SEP="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   BENCHMARK — Monte Carlo MPI (pi = 4 * N_in / N_total)    ║"
printf "║   Machine : %-28s — %2s coeurs  ║\n" "$(hostname)" "$NPROC"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  pi_ref = 3.14159265358979323846"
echo "  Convergence : erreur ~ 1/sqrt(N_total)"
echo "  P_SCALE : $P_LIST"
echo ""

# ── Sequentiel ───────────────────────────────────────────────────────
echo "$SEP"
echo " [SEQ] Monte Carlo sequentiel — N = 10 millions"
echo "$SEP"
./pi_mc_seq 10000000 2>&1
echo ""

# ── Convergence ──────────────────────────────────────────────────────
echo "$SEP"
echo " [CONVERGENCE] Erreur vs N"
echo "$SEP"
./pi_mc_seq convergence 2>&1
echo ""

# ── Client/serveur ────────────────────────────────────────────────────
echo "$SEP"
echo " [MPI] Client/serveur — duree = ${DURATION}s"
echo "$SEP"
echo ""
printf "  %-6s  %-10s  %-18s  %-14s  %-12s\n" \
    "P" "Clients" "N_total (pts)" "pi approx" "erreur abs"
printf "  %-6s  %-10s  %-18s  %-14s  %-12s\n" \
    "------" "----------" "------------------" "--------------" "------------"

for P in $P_LIST; do
    OUT=$(mpirun --oversubscribe -np $P ./pi_mc_cs $DURATION 2>&1)
    PI_V=$(echo "$OUT"  | grep -i "pi approx"      | grep -oE '[0-9]+\.[0-9]{4,}' | head -1)
    ERR=$(echo  "$OUT"  | grep -i "erreur absolue"  | grep -oE '[0-9]+\.[0-9]+e[+-][0-9]+' | head -1)
    N_TOT=$(echo "$OUT" | grep -i "N_total"         | grep -oE '[0-9]{7,}' | head -1)
    CLIENTS=$((P-1))
    if [ -z "$PI_V"  ]; then PI_V="---";  fi
    if [ -z "$ERR"   ]; then ERR="---";   fi
    if [ -z "$N_TOT" ]; then N_TOT="---"; fi
    printf "  P=%-4s  %-10s  %-18s  %-14s  %s\n" \
        "$P" "$CLIENTS" "$N_TOT" "$PI_V" "$ERR"
done

echo ""
echo "$SEP"
echo " Rappel : erreur ~ 1/sqrt(N_total) — N_total lineaire avec (P-1)"
echo "$SEP"
echo ""
echo "  CSV generes :"
ls -lh results_mc_*.csv 2>/dev/null | awk '{print "    " $5 "  " $9}'
echo ""
