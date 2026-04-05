#!/bin/bash
# bench.sh — Benchmark SEIR avec calcul du speedup
# Fix locale : LC_ALL=C force le point comme séparateur décimal

export LC_ALL=C   # ← empêche awk d'utiliser la virgule française

# Extraction du temps : premier nombre flottant X.XXXX sur toute ligne "Temps"
get_time() {
    grep -E "^Temps" | grep -oE '[0-9]+\.[0-9]+' | head -1
}

# Verdict selon speedup
verdict() {
    local spd="$1"
    awk -v s="$spd" 'BEGIN{
        if (s+0 >= 1.5) print "[EXCELLENT]"
        else if (s+0 >= 1.0) print "[OK]"
        else print "[sur-souscription]"
    }'
}

SEP="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       BENCHMARK SEIR — S(P) = T_seq / T(P)                 ║"
printf "║  Machine : %-28s — %2s cœurs  ║\n" "$(hostname)" "$(nproc)"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─── Référence séquentielle ───────────────────────────────────────────
echo "$SEP"
echo " [SEQ] Version séquentielle — srand(42)"
echo "$SEP"
OUTPUT=$(./seir_seq 2>&1)
echo "$OUTPUT" | grep -E "Jour[[:space:]]+(0|30|730)|Temps"
T_SEQ=$(echo "$OUTPUT" | get_time)

if [ -z "$T_SEQ" ]; then
    echo "  ERREUR : impossible d'extraire le temps séquentiel"
    exit 1
fi
echo ""
echo "  T_seq = ${T_SEQ} s  →  référence : S(P) = ${T_SEQ} / T(P)"

# ─── Asynchrone ───────────────────────────────────────────────────────
echo ""
echo "$SEP"
echo " [ASYNC] Version asynchrone — Fisher-Yates"
echo "$SEP"
OUTPUT=$(./seir_async 2>&1)
echo "$OUTPUT" | grep -E "Jour[[:space:]]+(0|30|730)|Temps"
T=$(echo "$OUTPUT" | get_time)
SPD=$(awk "BEGIN{printf \"%.2f\", $T_SEQ/$T}")
echo ""
printf "  Temps   = %s s\n" "$T"
printf "  Speedup = x%s   (%s / %s)\n" "$SPD" "$T_SEQ" "$T"

# ─── MPI non-reproductible ────────────────────────────────────────────
echo ""
echo "$SEP"
echo " [MPI] Non-reproductible — rand() par rang"
echo "$SEP"
printf "  %-5s  %-12s  %-10s  %-22s  %s\n" \
    "P" "Temps (s)" "Speedup" "Formule" "Verdict"
printf "  %-5s  %-12s  %-10s  %-22s  %s\n" \
    "-----" "----------" "--------" "----------------------" "-------"

for P in 1 2 4 8 16; do
    OUTPUT=$(mpirun --oversubscribe -np $P ./seir_mpi 2>&1)
    T=$(echo "$OUTPUT" | get_time)
    if [ -z "$T" ]; then
        printf "  P=%-3s  %-12s  %-10s\n" "$P" "ERREUR" "---"
        continue
    fi
    SPD=$(awk "BEGIN{printf \"%.2f\", $T_SEQ/$T}")
    VER=$(verdict "$SPD")
    printf "  P=%-3s  %-12s  x%-9s  %-22s  %s\n" \
        "$P" "${T} s" "$SPD" "${T_SEQ} / ${T}" "$VER"
done

# ─── MPI reproductible ────────────────────────────────────────────────
echo ""
echo "$SEP"
echo " [REPRO] Reproductible — xorshift128+ par agent"
echo "$SEP"
printf "  %-5s  %-12s  %-10s  %-22s  %s\n" \
    "P" "Temps (s)" "Speedup" "Formule" "Verdict"
printf "  %-5s  %-12s  %-10s  %-22s  %s\n" \
    "-----" "----------" "--------" "----------------------" "-------"

for P in 1 2 4 8 16; do
    OUTPUT=$(mpirun --oversubscribe -np $P ./seir_mpi_repro 2>&1)
    T=$(echo "$OUTPUT" | get_time)
    if [ -z "$T" ]; then
        printf "  P=%-3s  %-12s  %-10s\n" "$P" "ERREUR" "---"
        continue
    fi
    SPD=$(awk "BEGIN{printf \"%.2f\", $T_SEQ/$T}")
    VER=$(verdict "$SPD")
    printf "  P=%-3s  %-12s  x%-9s  %-22s  %s\n" \
        "$P" "${T} s" "$SPD" "${T_SEQ} / ${T}" "$VER"
done

# ─── Récapitulatif ────────────────────────────────────────────────────
echo ""
echo "$SEP"
echo " RÉCAPITULATIF — S(P) = T_seq / T_mpi(P)"
echo "$SEP"
printf "  %-22s  %-5s  %-12s  %-12s\n" \
    "Version" "P" "Temps (s)" "Speedup S(P)"
printf "  %-22s  %-5s  %-12s  %-12s\n" \
    "----------------------" "-----" "------------" "------------"
printf "  %-22s  %-5s  %-12s  %-12s\n" \
    "Sequentiel" "--" "${T_SEQ} s" "x1.00 (ref)"

for P in 1 2 4 8 16; do
    OUTPUT=$(mpirun --oversubscribe -np $P ./seir_mpi 2>&1)
    T=$(echo "$OUTPUT" | get_time)
    [ -z "$T" ] && continue
    SPD=$(awk "BEGIN{printf \"%.2f\", $T_SEQ/$T}")
    printf "  %-22s  P=%-3s  %-12s  x%s\n" "MPI non-repro" "$P" "${T} s" "$SPD"
done

for P in 1 2 4 8 16; do
    OUTPUT=$(mpirun --oversubscribe -np $P ./seir_mpi_repro 2>&1)
    T=$(echo "$OUTPUT" | get_time)
    [ -z "$T" ] && continue
    SPD=$(awk "BEGIN{printf \"%.2f\", $T_SEQ/$T}")
    printf "  %-22s  P=%-3s  %-12s  x%s\n" "MPI repro" "$P" "${T} s" "$SPD"
done
echo "$SEP"
echo ""
