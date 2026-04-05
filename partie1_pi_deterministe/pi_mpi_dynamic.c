#define _POSIX_C_SOURCE 200809L
/*
 * pi_mpi_dynamic.c — Calcul de π, MPI, équilibrage dynamique de charge
 *
 * Problème motivant :
 *   La fonction f(x) = 1/(1+x²) est décroissante. Les premiers rangs
 *   MPI traitent des zones où f est grande → somme partielle grande →
 *   sleep long → mauvaise efficacité parallèle en décomposition statique.
 *
 * Solution — patron maître-esclave :
 *   - Rang 0 (maître) : distribue de petites tâches (chunks), ne calcule pas
 *   - Rangs 1..P-1 (esclaves) : demandent une tâche, calculent, dorment,
 *     renvoient le résultat, recommencent immédiatement
 *
 * Protocole MPI :
 *   Esclave → MPI_Send(demande) → Maître
 *   Maître  → MPI_Send(chunk_id ou -1) → Esclave
 *   Esclave calcule + sleep → MPI_Send(résultat) → Maître
 *   Maître accumule. Quand toutes les tâches sont distribuées,
 *   envoie chunk_id = -1 (signal de fin) à chaque esclave.
 *
 * Paramètres ajustables :
 *   NTASKS       : nombre de tâches (chunks) — indépendant de P
 *   SLEEP_FACTOR : facteur du sleep (en secondes par unité d'intégrale)
 *
 * Compilation :
 *   mpicc -O2 -Wall -std=c99 -march=native -o pi_mpi_dynamic pi_mpi_dynamic.c -lm
 *
 * Usage :
 *   mpirun -np 4 ./pi_mpi_dynamic
 *   mpirun -np 4 ./pi_mpi_dynamic 1000000000 200 0.05
 *                                 n           chunks sleep_factor
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include <unistd.h>

#define PI_REF       3.14159265358979323846264338327950288419716939937510
#define TAG_REQUEST  1   /* esclave → maître : demande de tâche          */
#define TAG_TASK     2   /* maître → esclave : index de chunk (ou -1)    */
#define TAG_RESULT   3   /* esclave → maître : résultat partiel           */

static inline double f(double x) { return 1.0 / (1.0 + x * x); }

/* ------------------------------------------------------------------ */
/* Intégrale des trapèzes sur [a, b] avec n_loc points — Kahan        */
/* ------------------------------------------------------------------ */
static double trapezes_chunk(double a, double b, long n_loc)
{
    const double dx = (b - a) / (double)n_loc;
    double sum = 0.0, comp = 0.0;
    double y = 0.5 * (f(a) + f(b)) - comp;
    double t = sum + y; comp = (t - sum) - y; sum = t;
    for (long i = 1; i < n_loc; i++) {
        double xi = a + (double)i * dx;
        y = f(xi) - comp;
        t = sum + y; comp = (t - sum) - y; sum = t;
    }
    return sum * dx;
}

/* ------------------------------------------------------------------ */
/* Maître (rang 0)                                                      */
/* ------------------------------------------------------------------ */
static double master(int nprocs, int ntasks)
{
    int next_task = 0;     /* prochaine tâche à distribuer    */
    int tasks_done = 0;    /* nombre de résultats reçus        */
    int workers = nprocs - 1;
    double pi_total = 0.0;

    /* Pré-charger : envoyer une tâche à chaque esclave */
    for (int w = 1; w <= workers && next_task < ntasks; w++) {
        MPI_Send(&next_task, 1, MPI_INT, w, TAG_TASK, MPI_COMM_WORLD);
        next_task++;
    }
    /* Si moins de tâches que d'esclaves : envoyer le signal de fin */
    for (int w = ntasks + 1; w <= workers; w++) {
        int stop = -1;
        MPI_Send(&stop, 1, MPI_INT, w, TAG_TASK, MPI_COMM_WORLD);
    }

    /* Boucle de collecte */
    while (tasks_done < ntasks) {
        double result;
        MPI_Status status;
        /* Attendre un résultat de n'importe quel esclave */
        MPI_Recv(&result, 1, MPI_DOUBLE, MPI_ANY_SOURCE, TAG_RESULT,
                 MPI_COMM_WORLD, &status);
        pi_total += result;
        tasks_done++;

        /* Envoyer la prochaine tâche ou le signal de fin */
        int src = status.MPI_SOURCE;
        if (next_task < ntasks) {
            MPI_Send(&next_task, 1, MPI_INT, src, TAG_TASK, MPI_COMM_WORLD);
            next_task++;
        } else {
            int stop = -1;
            MPI_Send(&stop, 1, MPI_INT, src, TAG_TASK, MPI_COMM_WORLD);
        }
    }

    return 4.0 * pi_total;
}

/* ------------------------------------------------------------------ */
/* Esclave (rang 1 à P-1)                                              */
/* ------------------------------------------------------------------ */
static void worker(long n, int ntasks, double sleep_factor)
{
    double chunk_size = 1.0 / (double)ntasks;
    long   n_per_chunk = n / (long)ntasks;
    if (n_per_chunk < 1) n_per_chunk = 1;

    while (1) {
        /* Demander une tâche au maître */
        int dummy = 0;
        MPI_Send(&dummy, 1, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD);

        int task_id;
        MPI_Recv(&task_id, 1, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        if (task_id < 0) break;   /* signal de fin */

        /* Bornes du chunk */
        double a = task_id * chunk_size;
        double b = (task_id + 1 < ntasks)
                   ? (task_id + 1) * chunk_size
                   : 1.0;

        /* Calcul de l'intégrale partielle */
        double partial = trapezes_chunk(a, b, n_per_chunk);

        /* Sleep proportionnel à la valeur de l'intégrale partielle.
         * Simule un coût de calcul non uniforme : la courbe étant
         * décroissante, les premiers chunks "coûtent" plus cher.    */
        long sleep_us = (long)(partial * sleep_factor * 1e6);
        if (sleep_us > 0) usleep((useconds_t)sleep_us);

        /* Renvoyer le résultat */
        MPI_Send(&partial, 1, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
    }
}

/* ------------------------------------------------------------------ */
/* Écriture CSV                                                         */
/* ------------------------------------------------------------------ */
static void write_csv(const char *path, long n, int P, int ntasks,
                      double sf, double pi, double t, double t_static,
                      int append)
{
    FILE *fp = fopen(path, append ? "a" : "w");
    if (!fp) return;
    if (!append)
        fprintf(fp, "n,P,ntasks,sleep_factor,pi_approx,"
                    "erreur_abs,erreur_rel_pct,temps_s,"
                    "speedup_vs_static\n");

    double err_abs = fabs(pi - PI_REF);
    double err_rel = err_abs / PI_REF * 100.0;
    double spd_static = (t_static > 0.0) ? t_static / t : 0.0;
    fprintf(fp, "%ld,%d,%d,%.3f,%.15f,%.2e,%.2e,%.6f,%.3f\n",
            n, P, ntasks, sf, pi, err_abs, err_rel, t, spd_static);
    fclose(fp);
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs < 2) {
        if (rank == 0)
            fprintf(stderr, "Erreur : pi_mpi_dynamic nécessite P >= 2\n");
        MPI_Finalize(); return 1;
    }

    long   n            = (argc > 1) ? atol(argv[1]) : 1000000000L;
    int    ntasks       = (argc > 2) ? atoi(argv[2]) : 8 * nprocs;
    double sleep_factor = (argc > 3) ? atof(argv[3]) : 0.05;

    if (rank == 0) {
        printf("=======================================================\n");
        printf("  Calcul de π — MPI dynamique (maître-esclave)\n");
        printf("  n = %ld  |  P = %d  |  ntasks = %d  |  sleep = %.3f\n",
               n, nprocs, ntasks, sleep_factor);
        printf("=======================================================\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    double pi = 0.0;
    if (rank == 0)
        pi = master(nprocs, ntasks);
    else
        worker(n, ntasks, sleep_factor);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t_start;

    if (rank == 0) {
        double err = fabs(pi - PI_REF);
        printf("\n  π ≈ %.15f\n", pi);
        printf("  erreur absolue  = %.2e\n", err);
        printf("  temps de calcul = %.6f s\n", elapsed);

        /* Lire le temps statique de référence depuis le CSV */
        double t_static = 0.0;
        FILE *fref = fopen("results_mpi_static.csv", "r");
        if (fref) {
            char line[256]; long nn; int pp, nt;
            double pi_r,ea,er,ts,sp;
            fgets(line, sizeof(line), fref);
            while (fscanf(fref, "%ld,%d,%lf,%lf,%lf,%lf,%lf\n",
                          &nn, &pp, &pi_r, &ea, &er, &ts, &sp) == 7)
                if (nn == n && pp == nprocs) { t_static = ts; break; }
            fclose(fref);
        }
        if (t_static > 0.0)
            printf("  speedup vs static = %.3f ×\n", t_static / elapsed);

        int first = (access("results_mpi_dynamic.csv", F_OK) != 0);
        write_csv("results_mpi_dynamic.csv", n, nprocs, ntasks,
                  sleep_factor, pi, elapsed, t_static, !first);
        printf("\n  Résultats → results_mpi_dynamic.csv\n\n");
    }

    MPI_Finalize();
    return 0;
}
