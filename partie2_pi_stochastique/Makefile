# =============================================================================
# Makefile — Partie 2 : Calcul de Pi, methode stochastique (Monte Carlo)
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
OPT     = -O2 -Wall -std=c99 -march=native
LIBS    = -lm

NPROC   := $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
P_SCALE := $(shell python3 -c "n=$(NPROC); print(' '.join(str(p) for p in [2,4,8,16] if p<=max(n,2)))")
DURATION = 10

.PHONY: all bench bench_seq bench_cs bench_convergence bench_scalability bench_p \
        viz viz_pub clean help

all: pi_mc_seq pi_mc_cs
	@echo ""
	@echo "  Compilation OK — $(shell hostname) — $(NPROC) coeurs"
	@echo "    pi_mc_seq   pi_mc_cs"
	@echo "  P_SCALE : $(P_SCALE)"
	@echo "  Prochaine etape : make bench"
	@echo ""

pi_mc_seq: pi_mc_seq.c
	$(CC) $(OPT) -o $@ $< $(LIBS)
	@echo "  [OK] pi_mc_seq"

pi_mc_cs: pi_mc_client_server.c
	$(MPICC) $(OPT) -o $@ $< $(LIBS)
	@echo "  [OK] pi_mc_cs"

bench: all
	@chmod +x bench_mc.sh
	./bench_mc.sh

bench_seq: pi_mc_seq
	./pi_mc_seq 10000000
	./pi_mc_seq convergence

bench_cs: pi_mc_cs
	@echo "=== Monte Carlo client/serveur — P = $(P_SCALE) ==="
	@for P in $(P_SCALE); do \
	    echo "--- P = $$P  ($$(( $$P - 1 )) clients) ---"; \
	    mpirun --oversubscribe -np $$P ./pi_mc_cs $(DURATION); \
	done

bench_scalability: pi_mc_cs
	@echo "=== Scalabilite Monte Carlo — duree = $(DURATION)s ==="
	@for P in $(P_SCALE); do \
	    echo "--- P = $$P ---"; \
	    mpirun --oversubscribe -np $$P ./pi_mc_cs $(DURATION); \
	done

bench_convergence: pi_mc_seq
	./pi_mc_seq convergence

bench_p: pi_mc_cs
	@if [ -z "$(P)" ]; then echo "Usage : make bench_p P=8 D=30"; exit 1; fi
	mpirun --oversubscribe -np $(P) ./pi_mc_cs $(if $(D),$(D),$(DURATION))

viz:
	python3 pi_mc_viewer_pub.py

viz_pub:
	python3 pi_mc_viewer_pub.py

clean:
	rm -f pi_mc_seq pi_mc_cs
	rm -f results_mc_*.csv

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║   Partie 2 — Calcul de Pi, Monte Carlo MPI                  ║"
	@echo "║   Machine : $(shell hostname) — $(NPROC) coeurs                       ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  make all                  Compiler les 2 binaires"
	@echo "  make bench                Benchmark complet"
	@echo "  make bench_seq            Sequentiel + convergence"
	@echo "  make bench_cs             Client/serveur, P auto"
	@echo "  make bench_scalability    Scalabilite, duree = $(DURATION)s"
	@echo "  make bench_convergence    Convergence sequentiel"
	@echo "  make bench_p P=8 D=30     P et duree personnalises"
	@echo "  make viz_pub              Figures publication"
	@echo "  make clean"
	@echo ""
	@echo "  P_SCALE : $(P_SCALE)"
	@echo ""
