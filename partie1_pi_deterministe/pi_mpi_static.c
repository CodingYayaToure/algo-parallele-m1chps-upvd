#define _POSIX_C_SOURCE 200809L
/*
 * pi_mpi_static.c — Calcul de π par MPI, décomposition statique
 *
 * Stratégie : chaque rang traite une tranche continue égale de [0, 1].
 *   Rang r → [r/P, (r+1)/P]  avec n/P trapèzes
 *
 * Boucle par rang :
 *   1. Chaque rang calcule sa somme partielle (Kahan)
 *   2. MPI_Reduce somme toutes les contributions → rang 0
 *   3. Rang 0 affiche et sauvegarde les résultats
 *
 * Limitation : si le coût de calcul par point n'est pas uniforme,
 *   la décomposition statique sera déséquilibrée (voir pi_mpi_dynamic.c)
 *
 * Compilation :
 *   mpicc -O2 -Wall -std=c99 -march=native -o pi_mpi_static pi_mpi_static.c -lm
 *
 * Usage :
 *   mpirun -np 4 ./pi_mpi_static           # n = 1 milliard
 *   mpirun -np 4 ./pi_mpi_static 5000      # n personnalisé
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>

#define PI_REF 3.14159265358979323846264338327950288419716939937510

static inline double f(double x) { return 1.0 / (1.0 + x * x); }

/* ------------------------------------------------------------------ */
/* Somme de Kahan sur la tranche [x_lo, x_hi] avec n_local trapèzes   */
/* ------------------------------------------------------------------ */
static double trapezes_local_kahan(double x_lo, double x_hi, long n_local)
{
    const double dx  = (x_hi - x_lo) / (double)n_local;
    double sum  = 0.0;
    double comp = 0.0;

    /* Extrémités de la tranche locale : contribuent à moitié */
    double y = 0.5 * (f(x_lo) + f(x_hi)) - comp;
    double t = sum + y; comp = (t - sum) - y; sum = t;

    /* Points intérieurs */
    for (long i = 1; i < n_local; i++) {
        double xi = x_lo + (double)i * dx;
        y = f(xi) - comp;
        t = sum + y; comp = (t - sum) - y; sum = t;
    }
    return sum * dx;
}

/* ------------------------------------------------------------------ */
/* Écriture CSV (rang 0 uniquement)                                     */
/* Format : n,P,pi_approx,erreur_abs,erreur_rel_pct,temps_s,speedup   */
/* ------------------------------------------------------------------ */
static void write_csv(const char *path, long n, int P, double pi,
                      double t, double t_seq_ref, int append)
{
    FILE *fp = fopen(path, append ? "a" : "w");
    if (!fp) return;
    if (!append)
        fprintf(fp, "n,P,pi_approx,erreur_abs,erreur_rel_pct,temps_s,speedup\n");

    double err_abs = fabs(pi - PI_REF);
    double err_rel = err_abs / PI_REF * 100.0;
    double speedup  = (t_seq_ref > 0.0) ? t_seq_ref / t : 0.0;
    fprintf(fp, "%ld,%d,%.15f,%.2e,%.2e,%.6f,%.3f\n",
            n, P, pi, err_abs, err_rel, t, speedup);
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

    long n = (argc > 1) ? atol(argv[1]) : 1000000000L;
    if (n < (long)nprocs) {
        if (rank == 0)
            fprintf(stderr, "Erreur : n (%ld) doit être >= P (%d)\n", n, nprocs);
        MPI_Finalize(); return 1;
    }

    /* Partitionnement équitable (le dernier rang prend le reste) */
    long base   = n / nprocs;
    long lo     = (long)rank * base;
    long hi     = (rank == nprocs - 1) ? n : lo + base;
    long n_loc  = hi - lo;

    double dx_glob = 1.0 / (double)n;
    double x_lo    = (double)lo * dx_glob;
    double x_hi    = (double)hi * dx_glob;

    if (rank == 0) {
        printf("=======================================================\n");
        printf("  Calcul de π — MPI statique\n");
        printf("  n = %ld  |  P = %d  |  n/P ≈ %ld trapèzes/rang\n",
               n, nprocs, base);
        printf("=======================================================\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* Chaque rang calcule sa somme partielle */
    double partial = trapezes_local_kahan(x_lo, x_hi, n_loc);

    /* MPI_Reduce : somme de toutes les contributions */
    double total = 0.0;
    MPI_Reduce(&partial, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    if (rank == 0) {
        double pi = 4.0 * total;
        double err = fabs(pi - PI_REF);

        printf("\n  π ≈ %.15f\n", pi);
        printf("  erreur absolue   = %.2e\n", err);
        printf("  erreur relative  = %.2e %%\n", err / PI_REF * 100.0);
        printf("  temps de calcul  = %.6f s\n", elapsed);

        /* Lire le temps séquentiel de référence depuis le CSV s'il existe */
        double t_seq_ref = 0.0;
        FILE *fref = fopen("results_seq.csv", "r");
        if (fref) {
            char line[256]; long nn; double pi_r,ea,er,ts;
            fgets(line, sizeof(line), fref); /* header */
            while (fscanf(fref, "%ld,%lf,%lf,%lf,%lf\n",
                          &nn, &pi_r, &ea, &er, &ts) == 5)
                if (nn == n) { t_seq_ref = ts; break; }
            fclose(fref);
        }
        if (t_seq_ref > 0.0)
            printf("  speedup          = %.3f ×  (ref seq: %.6f s)\n",
                   t_seq_ref / elapsed, t_seq_ref);

        /* CSV */
        int first = (access("results_mpi_static.csv", F_OK) != 0);
        write_csv("results_mpi_static.csv", n, nprocs, pi,
                  elapsed, t_seq_ref, !first);
        printf("\n  Résultats → results_mpi_static.csv\n\n");
    }

    MPI_Finalize();
    return 0;
}
