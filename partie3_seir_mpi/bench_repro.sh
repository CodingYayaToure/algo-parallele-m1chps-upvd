#!/bin/bash
# bench_repro.sh — Benchmark MPI reproductible uniquement
export LC_ALL=C
get_time() { grep -E "^Temps" | grep -oE '[0-9]+\.[0-9]+' | head -1; }

echo "=== Benchmark MPI reproductible ==="
OUTPUT=$(./seir_seq 2>&1)
T_SEQ=$(echo "$OUTPUT" | get_time)
echo "Seq : ${T_SEQ} s  (référence)"

for P in 1 2 4 8 16; do
    OUTPUT=$(mpirun --oversubscribe -np $P ./seir_mpi_repro 2>&1)
    T=$(echo "$OUTPUT" | get_time)
    SPD=$(awk "BEGIN{printf \"%.2f\", $T_SEQ/$T}")
    echo "REPRO P=$P : ${T} s   S($P) = $T_SEQ / $T = x$SPD"
done
