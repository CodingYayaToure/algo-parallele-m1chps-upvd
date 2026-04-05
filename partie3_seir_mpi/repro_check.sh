#!/bin/bash
# repro_check.sh — Vérifie la reproductibilité (I@J30 identique pour tout P)

echo "========================================"
echo " Vérification de la reproductibilité"
echo " I@J30 doit être identique pour tout P"
echo "========================================"

for P in 1 2 4 8 16; do
    mpirun --oversubscribe -np $P ./seir_mpi_repro > /dev/null 2>&1
    echo "  P=$P : counts_repro_${P}.csv généré"
done

echo ""
echo "Comparaison diff (vide = IDENTIQUES) :"
ALL_OK=true
for P in 2 4 8 16; do
    if diff -q counts_repro_1.csv counts_repro_${P}.csv > /dev/null 2>&1; then
        echo "  P=1 vs P=$P : IDENTIQUES  [OK]"
    else
        echo "  P=1 vs P=$P : DIFFERENTS  [ECHEC]"
        ALL_OK=false
    fi
done

echo ""
echo "Valeur I au Jour 30 (doit être constante) :"
for P in 1 2 4 8 16; do
    VAL=$(grep "^30," counts_repro_${P}.csv | cut -d',' -f4)
    echo "  counts_repro_$P.csv  →  I@J30 = $VAL"
done

echo ""
if $ALL_OK; then
    echo "  REPRODUCTIBILITÉ : OK — xorshift128+ par agent fonctionne"
else
    echo "  REPRODUCTIBILITÉ : ECHEC"
fi
