#!/bin/bash
# bench_pi.sh — Benchmark Partie 1 — compatible toutes machines
export LC_ALL=C

NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
P_LIST=$(python3 -c "n=$NPROC; print(' '.join(str(p) for p in [1,2,4,8,16] if p<=max(n,2)))")
SEP="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

get_time() {
    grep -oE "[0-9]+\.[0-9]+ s" | grep -oE "[0-9]+\.[0-9]+" | head -1
}

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       BENCHMARK — Calcul de pi, methode des trapezes        ║"
printf "║   Machine : %-28s — %2s coeurs  ║\n" "$(hostname)" "$NPROC"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  pi_ref = 3.14159265358979323846"
echo "  P_SCALE : $P_LIST"
echo ""

# ── Programme 1 : sequentiel ─────────────────────────────────────────
echo "$SEP"
echo " [PROG 1] Sequentiel — n = 5 000 vs n = 1 milliard"
echo "$SEP"
echo "--- n = 5 000 ---"
./pi_seq 5000 2>&1
echo ""
echo "--- n = 1 milliard ---"
OUT=$(./pi_seq 1000000000 2>&1)
echo "$OUT"
T_SEQ=$(echo "$OUT" | get_time)
echo ""
echo "  Reference : T_seq = ${T_SEQ} s"

# ── Programme 2 : MPI statique ───────────────────────────────────────
echo ""
echo "$SEP"
echo " [PROG 2] MPI statique — scalabilite n = 1 milliard"
echo "$SEP"
echo ""
printf "  %-5s  %-12s  %-14s  %-24s  %s\n" \
    "P" "Temps (s)" "Speedup S(P)" "Formule" "Verdict"
printf "  %-5s  %-12s  %-14s  %-24s  %s\n" \
    "-----" "----------" "------------" "------------------------" "-------"

for P in $P_LIST; do
    OUT=$(mpirun --oversubscribe -np $P ./pi_mpi_static 1000000000 2>&1)
    T=$(echo "$OUT" | get_time)
    if [ -n "$T" ] && [ -n "$T_SEQ" ]; then
        SPD=$(awk "BEGIN{printf \"%.3f\", $T_SEQ/$T}")
        VERDICT=$(awk "BEGIN{s=$SPD; if(s>=1.5) print \"[EXCELLENT]\"
            else if(s>=1.0) print \"[OK]\"
            else print \"[sur-souscription]\"}")
        printf "  P=%-3s  %-12s  x%-13s  %-24s  %s\n" \
            "$P" "${T} s" "$SPD" "$T_SEQ / $T" "$VERDICT"
    else
        printf "  P=%-3s  %-12s  %-14s\n" "$P" "ERREUR" "---"
    fi
done

# ── Programme 3 : MPI dynamique ──────────────────────────────────────
echo ""
echo "$SEP"
echo " [PROG 3] MPI dynamique vs statique — equilibrage de charge"
echo "$SEP"
echo ""
printf "  %-22s  %-5s  %-12s  %-14s\n" "Version" "P" "Temps (s)" "Gain vs statique"
printf "  %-22s  %-5s  %-12s  %-14s\n" "----------------------" "-----" "----------" "----------------"

for P in $P_LIST; do
    if [ $P -ge 2 ]; then
        OUT_S=$(mpirun --oversubscribe -np $P ./pi_mpi_static  1000000000 2>&1)
        OUT_D=$(mpirun --oversubscribe -np $P ./pi_mpi_dynamic 1000000000 2>&1)
        T_S=$(echo "$OUT_S" | get_time)
        T_D=$(echo "$OUT_D" | get_time)
        printf "  %-22s  P=%-3s  %-12s  %s\n" "MPI statique"  "$P" "${T_S} s" "reference"
        if [ -n "$T_S" ] && [ -n "$T_D" ]; then
            GAIN=$(awk "BEGIN{printf \"%.3f\", $T_S/$T_D}")
            printf "  %-22s  P=%-3s  %-12s  x%s\n" "MPI dynamique" "$P" "${T_D} s" "$GAIN"
        else
            printf "  %-22s  P=%-3s  %-12s\n" "MPI dynamique" "$P" "ERREUR"
        fi
        echo ""
    fi
done

echo "$SEP"
echo ""
echo "  CSV generes :"
ls -lh results_*.csv 2>/dev/null | awk '{print "    " $5 "  " $9}'
echo ""
