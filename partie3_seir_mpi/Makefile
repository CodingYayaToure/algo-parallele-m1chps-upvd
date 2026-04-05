# =============================================================================
# Makefile — Simulation SEIR multi-agents (730 jours)
#
# Machine cible : Intel i9-11950H · 8 cœurs physiques · 128 Go RAM
# Note : --oversubscribe ajouté automatiquement uniquement si P > 8
#
# DÉMARRAGE RAPIDE :
#   make all && make data_all && make bench
#
# AIDE :
#   make help
# =============================================================================

CC      = gcc
MPICC   = mpicc
OPT     = -O2 -Wall -std=c99 -march=native
LIBS    = -lm
# Sur i9-11950H (8 cœurs) : pas d'oversubscribe jusqu'à P=8
# Sur laptop 2 cœurs     : toujours --oversubscribe
MPI_RUN = mpirun

.PHONY: all \
        bench bench_nr bench_repro repro_check \
        run_seq run_async run_mpi run_repro run_all_mpi run_all_repro seeds data_all \
        viz viz_mpi2 viz_mpi4 viz_mpi8 viz_repro2 viz_repro4 viz_async \
        viz2 viz2_auto viz3 viz3_fast viz3_seq \
        compare_3 compare_5 compare_repro compare_all_mpi \
        compare_sync_async compare_all_repro \
        export_seq export_mpi2 export_async export_spaghetti export_viewer3 export_all \
        clean cleanviz help

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPILATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

all: seir_seq seir_async seir_mpi seir_mpi_repro seir_multiseed
	@chmod +x bench.sh bench_nr.sh bench_repro.sh repro_check.sh 2>/dev/null || true
	@echo ""
	@echo "  Compilation OK :"
	@echo "    seir_seq  seir_async  seir_mpi  seir_mpi_repro  seir_multiseed"
	@echo ""
	@echo "  Prochaine etape : make data_all && make bench"
	@echo ""

seir_seq: seir_seq.c
	$(CC) $(OPT) -o $@ $< $(LIBS)
	@echo "  [OK] seir_seq"

seir_async: seir_async.c
	$(CC) $(OPT) -o $@ $< $(LIBS)
	@echo "  [OK] seir_async"

seir_mpi: seir_mpi.c
	$(MPICC) $(OPT) -o $@ $< $(LIBS)
	@echo "  [OK] seir_mpi"

seir_mpi_repro: seir_mpi_repro.c
	$(MPICC) $(OPT) -o $@ $< $(LIBS)
	@echo "  [OK] seir_mpi_repro"

seir_multiseed: seir_multiseed.c
	$(CC) $(OPT) -o $@ $< $(LIBS)
	@echo "  [OK] seir_multiseed"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GÉNÉRATION DES DONNÉES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

run_seq: seir_seq
	@echo "--- Sequentiel srand(42) ---"
	./seir_seq

run_async: seir_async
	@echo "--- Asynchrone Fisher-Yates ---"
	./seir_async

run_all_mpi: seir_mpi
	@for P in 1 2 4 8 16; do \
	    echo "--- MPI non-repro P=$$P ---"; \
	    $(MPI_RUN) --oversubscribe -np $$P ./seir_mpi; \
	done

run_all_repro: seir_mpi_repro
	@for P in 1 2 4 8 16; do \
	    echo "--- MPI repro P=$$P ---"; \
	    $(MPI_RUN) --oversubscribe -np $$P ./seir_mpi_repro; \
	done

run_mpi: seir_mpi
	$(MPI_RUN) --oversubscribe -np 2 ./seir_mpi
	$(MPI_RUN) --oversubscribe -np 4 ./seir_mpi

run_repro: seir_mpi_repro
	$(MPI_RUN) --oversubscribe -np 2 ./seir_mpi_repro
	$(MPI_RUN) --oversubscribe -np 4 ./seir_mpi_repro

seeds: seir_multiseed
	@echo "--- Multi-seeds 42 a 51 ---"
	@for seed in 42 43 44 45 46 47 48 49 50 51; do \
	    ./seir_multiseed $$seed; \
	done
	@echo "  Done : counts_seed_42..51.csv + frames_seed_42..51.bin"

# Tout générer en une seule commande
data_all: seir_seq seir_async seir_mpi seir_mpi_repro seir_multiseed
	@echo "========================================"
	@echo " Generation de toutes les donnees"
	@echo "========================================"
	@echo "--- [1/5] Sequentiel ---"
	./seir_seq
	@echo "--- [2/5] Asynchrone ---"
	./seir_async
	@echo "--- [3/5] MPI non-repro P=1,2,4,8,16 ---"
	@for P in 1 2 4 8 16; do \
	    $(MPI_RUN) --oversubscribe -np $$P ./seir_mpi; \
	done
	@echo "--- [4/5] MPI repro P=1,2,4,8,16 ---"
	@for P in 1 2 4 8 16; do \
	    $(MPI_RUN) --oversubscribe -np $$P ./seir_mpi_repro; \
	done
	@echo "--- [5/5] Multi-seeds 42-51 ---"
	@for seed in 42 43 44 45 46 47 48 49 50 51; do \
	    ./seir_multiseed $$seed; \
	done
	@echo ""
	@echo "  Toutes les donnees generees."
	@echo "  Prochaine etape : make bench"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BENCHMARK — délègue aux scripts shell (évite problèmes locale + $$ Make)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bench: seir_seq seir_async seir_mpi seir_mpi_repro
	@chmod +x bench.sh
	./bench.sh

bench_nr: seir_seq seir_mpi
	@chmod +x bench_nr.sh
	./bench_nr.sh

bench_repro: seir_seq seir_mpi_repro
	@chmod +x bench_repro.sh
	./bench_repro.sh

repro_check: seir_mpi_repro
	@chmod +x repro_check.sh
	./repro_check.sh

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEWER 1 — Animation grille + courbes + métriques
# Badge coloré auto : BLEU=seq · JAUNE=mpi · VERT=repro · VIOLET=async
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

viz: counts_seq.csv frames_seq.bin
	@echo ">>> Viewer 1 - Sequentiel [badge BLEU]"
	python3 seir_viewer.py --bin frames_seq.bin --csv counts_seq.csv

viz_mpi2: counts_mpi_2.csv frames_mpi_2.bin
	@echo ">>> Viewer 1 - MPI P=2 non-repro [badge JAUNE]"
	python3 seir_viewer.py --bin frames_mpi_2.bin --csv counts_mpi_2.csv

viz_mpi4: counts_mpi_4.csv frames_mpi_4.bin
	@echo ">>> Viewer 1 - MPI P=4 non-repro [badge JAUNE]"
	python3 seir_viewer.py --bin frames_mpi_4.bin --csv counts_mpi_4.csv

viz_mpi8: counts_mpi_8.csv frames_mpi_8.bin
	@echo ">>> Viewer 1 - MPI P=8 non-repro [badge JAUNE]"
	python3 seir_viewer.py --bin frames_mpi_8.bin --csv counts_mpi_8.csv

viz_repro2: counts_repro_2.csv frames_repro_2.bin
	@echo ">>> Viewer 1 - MPI repro P=2 [badge VERT]"
	python3 seir_viewer.py --bin frames_repro_2.bin --csv counts_repro_2.csv

viz_repro4: counts_repro_4.csv frames_repro_4.bin
	@echo ">>> Viewer 1 - MPI repro P=4 [badge VERT]"
	python3 seir_viewer.py --bin frames_repro_4.bin --csv counts_repro_4.csv

viz_async: counts_async.csv frames_async.bin
	@echo ">>> Viewer 1 - Asynchrone Fisher-Yates [badge VIOLET]"
	python3 seir_viewer.py --bin frames_async.bin --csv counts_async.csv

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEWER 2 — Spaghetti 730j (10 seeds superposés)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

viz2: counts_seed_42.csv frames_seed_42.bin
	@echo ">>> Viewer 2 - Spaghetti 730j (10 seeds)"
	python3 seir_viewer_2.py \
	    --main counts_seed_42.csv \
	    --frames frames_seed_42.bin \
	    --seeds counts_seed_43.csv counts_seed_44.csv counts_seed_45.csv \
	            counts_seed_46.csv counts_seed_47.csv counts_seed_48.csv \
	            counts_seed_49.csv counts_seed_50.csv counts_seed_51.csv

viz2_auto: seeds
	@echo ">>> Viewer 2 - Spaghetti 730j (seeds generees)"
	python3 seir_viewer_2.py

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEWER 3 — Simulation interactive manim-style
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

viz3: counts_seed_42.csv
	@echo ">>> Viewer 3 - Simulation interactive (Espace=pause R=reset +/-=vitesse Q=quitter)"
	python3 seir_viewer_3.py --csv counts_seed_42.csv

viz3_fast: counts_seed_42.csv
	@echo ">>> Viewer 3 - Mode accelere"
	python3 seir_viewer_3.py --csv counts_seed_42.csv --fast

viz3_seq: counts_seq.csv
	@echo ">>> Viewer 3 - Donnees sequentielles"
	python3 seir_viewer_3.py --csv counts_seq.csv

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPARAISONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

compare_3: counts_seq.csv counts_mpi_2.csv counts_async.csv
	@echo ">>> Comparaison : Seq / MPI P=2 / Async"
	python3 seir_viewer.py \
	    --compare counts_seq.csv counts_mpi_2.csv counts_async.csv \
	    --labels "Sequentiel srand(42)" "MPI P=2 non-repro" "Async Fisher-Yates"

compare_5: counts_seq.csv counts_mpi_2.csv counts_mpi_4.csv \
           counts_mpi_8.csv counts_async.csv
	@echo ">>> Comparaison : Seq / MPI P=2,4,8 / Async"
	python3 seir_viewer.py \
	    --compare counts_seq.csv \
	              counts_mpi_2.csv counts_mpi_4.csv counts_mpi_8.csv \
	              counts_async.csv \
	    --labels "Sequentiel" "MPI P=2" "MPI P=4" "MPI P=8" "Async"

compare_repro: counts_seq.csv counts_mpi_2.csv counts_repro_2.csv
	@echo ">>> Comparaison : non-repro vs repro (P=2)"
	python3 seir_viewer.py \
	    --compare counts_seq.csv counts_mpi_2.csv counts_repro_2.csv \
	    --labels "Sequentiel srand(42)" "MPI P=2 non-repro" "MPI P=2 repro xorshift128+"

compare_all_mpi: counts_seq.csv counts_mpi_1.csv counts_mpi_2.csv \
                 counts_mpi_4.csv counts_mpi_8.csv counts_mpi_16.csv
	@echo ">>> Comparaison : Seq + MPI P=1/2/4/8/16"
	python3 seir_viewer.py \
	    --compare counts_seq.csv \
	              counts_mpi_1.csv counts_mpi_2.csv \
	              counts_mpi_4.csv counts_mpi_8.csv counts_mpi_16.csv \
	    --labels "Sequentiel" "MPI P=1" "MPI P=2" "MPI P=4" "MPI P=8" "MPI P=16"

compare_sync_async: counts_seq.csv counts_async.csv
	@echo ">>> Comparaison : Synchrone vs Asynchrone (pic I +20.8%)"
	python3 seir_viewer.py \
	    --compare counts_seq.csv counts_async.csv \
	    --labels "Synchrone srand(42)" "Asynchrone Fisher-Yates"

compare_all_repro: counts_repro_1.csv counts_repro_2.csv \
                   counts_repro_4.csv counts_repro_8.csv counts_repro_16.csv
	@echo ">>> Comparaison : toutes versions repro (doivent etre identiques)"
	python3 seir_viewer.py \
	    --compare counts_repro_1.csv counts_repro_2.csv \
	              counts_repro_4.csv counts_repro_8.csv counts_repro_16.csv \
	    --labels "Repro P=1" "Repro P=2" "Repro P=4" "Repro P=8" "Repro P=16"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPORTS MP4 (nécessite ffmpeg)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export_seq: counts_seq.csv frames_seq.bin
	python3 seir_viewer.py \
	    --bin frames_seq.bin --csv counts_seq.csv \
	    --save anim_sequentiel.mp4 --fps 25
	@echo "  Export : anim_sequentiel.mp4"

export_mpi2: counts_mpi_2.csv frames_mpi_2.bin
	python3 seir_viewer.py \
	    --bin frames_mpi_2.bin --csv counts_mpi_2.csv \
	    --save anim_mpi_p2.mp4 --fps 25
	@echo "  Export : anim_mpi_p2.mp4"

export_async: counts_async.csv frames_async.bin
	python3 seir_viewer.py \
	    --bin frames_async.bin --csv counts_async.csv \
	    --save anim_async.mp4 --fps 25
	@echo "  Export : anim_async.mp4"

export_spaghetti: counts_seed_42.csv frames_seed_42.bin
	python3 seir_viewer_2.py --save anim_spaghetti_730j.mp4 --fps 20
	@echo "  Export : anim_spaghetti_730j.mp4"

export_viewer3: counts_seed_42.csv
	python3 seir_viewer_3.py \
	    --csv counts_seed_42.csv --save anim_viewer3.mp4 --fps 25
	@echo "  Export : anim_viewer3.mp4"

export_all: export_seq export_mpi2 export_async export_spaghetti
	@echo ""
	@echo "  Exports termines :"
	@ls -lh anim_*.mp4 2>/dev/null || echo "  Aucun fichier MP4 trouve"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AIDE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║   SEIR MPI — M1 CHPS — Commandes disponibles                ║"
	@echo "║   Machine : i9-11950H · 8 coeurs · 128 Go RAM               ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  DEMARRAGE RAPIDE"
	@echo "    make all && make data_all && make bench"
	@echo ""
	@echo "  COMPILATION"
	@echo "    make all               Compiler les 5 binaires"
	@echo ""
	@echo "  DONNEES"
	@echo "    make data_all          Tout generer (recommande)"
	@echo "    make run_seq           Sequentiel srand(42)"
	@echo "    make run_async         Asynchrone Fisher-Yates"
	@echo "    make run_all_mpi       MPI non-repro P=1,2,4,8,16"
	@echo "    make run_all_repro     MPI repro P=1,2,4,8,16"
	@echo "    make seeds             10 seeds pour spaghetti"
	@echo ""
	@echo "  BENCHMARK + SPEEDUP  S(P) = T_seq / T(P)"
	@echo "    make bench             Complet avec formule affichee"
	@echo "    make bench_nr          MPI non-repro uniquement"
	@echo "    make bench_repro       MPI repro uniquement"
	@echo "    make repro_check       Verifie I@J30 identique pour tout P"
	@echo ""
	@echo "  VIEWER 1 — Animation grille + courbes + metriques"
	@echo "    make viz               Sequentiel          [badge BLEU]"
	@echo "    make viz_mpi2          MPI P=2 non-repro   [badge JAUNE]"
	@echo "    make viz_mpi4          MPI P=4 non-repro   [badge JAUNE]"
	@echo "    make viz_mpi8          MPI P=8 non-repro   [badge JAUNE]"
	@echo "    make viz_repro2        MPI repro P=2       [badge VERT]"
	@echo "    make viz_repro4        MPI repro P=4       [badge VERT]"
	@echo "    make viz_async         Asynchrone          [badge VIOLET]"
	@echo ""
	@echo "  VIEWER 2 — Spaghetti 730j"
	@echo "    make viz2              10 seeds superposes"
	@echo "    make viz2_auto         Genere seeds si besoin + lance"
	@echo ""
	@echo "  VIEWER 3 — Simulation interactive"
	@echo "    make viz3              Normal"
	@echo "    make viz3_fast         Mode accelere"
	@echo ""
	@echo "  COMPARAISONS"
	@echo "    make compare_3         Seq / MPI P=2 / Async"
	@echo "    make compare_5         Seq / MPI P=2,4,8 / Async"
	@echo "    make compare_repro     Non-repro vs Repro (P=2)"
	@echo "    make compare_all_mpi   Seq + tous les P MPI"
	@echo "    make compare_sync_async Sync vs Async"
	@echo "    make compare_all_repro Toutes versions repro"
	@echo ""
	@echo "  EXPORTS MP4 (necessite ffmpeg)"
	@echo "    make export_seq        -> anim_sequentiel.mp4"
	@echo "    make export_mpi2       -> anim_mpi_p2.mp4"
	@echo "    make export_async      -> anim_async.mp4"
	@echo "    make export_spaghetti  -> anim_spaghetti_730j.mp4"
	@echo "    make export_all        -> toutes les videos"
	@echo ""
	@echo "  NETTOYAGE"
	@echo "    make clean             Binaires + CSV + BIN"
	@echo "    make cleanviz          Videos MP4 uniquement"
	@echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NETTOYAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

clean:
	rm -f seir_seq seir_async seir_mpi seir_mpi_repro seir_multiseed
	rm -f counts_*.csv frames_*.bin

cleanviz:
	rm -f anim_*.mp4