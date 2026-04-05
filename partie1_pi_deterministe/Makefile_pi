# =============================================================================
# Makefile — Partie 1 : Calcul de Pi, methode deterministe
#
# Compatible toutes machines (--oversubscribe toujours actif)
#   HP 15s-eq0xxx : 2 coeurs physiques
#   i9-11950H     : 8 coeurs physiques
#
# DEMARRAGE RAPIDE :
#   make all && make bench
# =============================================================================

CC      = gcc
MPICC   = mpicc
OPT     = -O2 -Wall -std=c99 -D_POSIX_C_SOURCE=200809L -march=native
LIBS    = -lm

NPROC  := $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
P_SCALE := $(shell python3 -c "n=$(NPROC); print(' '.join(str(p) for p in [1,2,4,8,16] if p<=max(n,2)))")

.PHONY: all bench bench_seq bench_static bench_dynamic bench_p viz viz_pub clean help

all: pi_seq pi_mpi_static pi_mpi_dynamic
	@echo ""
	@echo "  Compilation OK — $(shell hostname) — $(NPROC) coeurs"
	@echo "    pi_seq  pi_mpi_static  pi_mpi_dynamic"
	@echo "  Prochaine etape : make bench"
	@echo ""

pi_seq: pi_seq.c
	$(CC) $(OPT) -o $@ $< $(LIBS)
	@echo "  [OK] pi_seq"

pi_mpi_static: pi_mpi_static.c
	$(MPICC) $(OPT) -o $@ $< $(LIBS)
	@echo "  [OK] pi_mpi_static"

pi_mpi_dynamic: pi_mpi_dynamic.c
	$(MPICC) $(OPT) -o $@ $< $(LIBS)
	@echo "  [OK] pi_mpi_dynamic"

bench: all
	@chmod +x bench_pi.sh
	./bench_pi.sh

bench_seq: pi_seq
	@echo "=== Sequentiel ==="
	./pi_seq 5000
	./pi_seq 1000000000

bench_static: pi_seq pi_mpi_static
	@echo "=== MPI statique — P = $(P_SCALE) ==="
	./pi_seq 1000000000
	@for P in $(P_SCALE); do \
	    echo "--- P = $$P ---"; \
	    mpirun --oversubscribe -np $$P ./pi_mpi_static 1000000000; \
	done

bench_dynamic: pi_mpi_static pi_mpi_dynamic
	@echo "=== MPI dynamique vs statique ==="
	@for P in $(P_SCALE); do \
	    if [ $$P -ge 2 ]; then \
	        echo "--- Statique  P=$$P ---"; \
	        mpirun --oversubscribe -np $$P ./pi_mpi_static  1000000000; \
	        echo "--- Dynamique P=$$P ---"; \
	        mpirun --oversubscribe -np $$P ./pi_mpi_dynamic 1000000000; \
	    fi; \
	done

bench_p: pi_seq pi_mpi_static pi_mpi_dynamic
	@if [ -z "$(P)" ]; then echo "Usage : make bench_p P=8"; exit 1; fi
	./pi_seq 1000000000
	mpirun --oversubscribe -np $(P) ./pi_mpi_static  1000000000
	mpirun --oversubscribe -np $(P) ./pi_mpi_dynamic 1000000000

viz:
	python3 pi_viewer.py

viz_pub:
	python3 pi_viewer_pub.py

clean:
	rm -f pi_seq pi_mpi_static pi_mpi_dynamic
	rm -f results_*.csv

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║   Partie 1 — Calcul de Pi, methode deterministe             ║"
	@echo "║   Machine : $(shell hostname) — $(NPROC) coeurs                       ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  make all              Compiler les 3 binaires"
	@echo "  make bench            Benchmark complet"
	@echo "  make bench_seq        Sequentiel uniquement"
	@echo "  make bench_static     MPI statique, P auto"
	@echo "  make bench_dynamic    Statique vs dynamique"
	@echo "  make bench_p P=8      P specifique"
	@echo "  make viz_pub          Figures publication"
	@echo "  make clean"
	@echo ""
	@echo "  P_SCALE : $(P_SCALE)"
	@echo ""
