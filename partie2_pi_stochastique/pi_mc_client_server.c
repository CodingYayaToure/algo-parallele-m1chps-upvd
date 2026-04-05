/*
 * pi_mc_client_server.c — Monte Carlo MPI, modele client/serveur
 *
 * Architecture :
 *   Rang 0 = Serveur  : collecte les resultats partiels, gere le temps,
 *                        renvoie les ordres aux clients (continuer/arreter)
 *   Rangs 1..P-1 = Clients : calculent des paquets de BATCH_SIZE points,
 *                             envoient le compte N_in au serveur,
 *                             recoivent l ordre de continuer ou de s arreter
 *
 * Protocole MPI :
 *   Client -> Serveur : MPI_Send(n_in, 1, LONG, 0, TAG_RESULT)
 *   Serveur -> Client : MPI_Send(go,   1, INT,  r, TAG_ORDER)
 *                       go = 1 (continuer)  ou  go = 0 (arreter)
 *
 * Duree : le serveur tourne pendant DURATION secondes (defaut 10s).
 * Chaque client recoit l ordre d arreter quand le serveur a ecoule sa duree.
 *
 * Propriete : embarrassingly parallel — aucune dependance entre les clients.
 * Scalabilite : N_total augmente lineairement avec P-1 clients.
 * Convergence : erreur ~ 1/sqrt(N_total) ~ 1/sqrt((P-1) * N_par_client)
 *
 * Compilation :
 *   mpicc -O2 -Wall -std=c99 -march=native -o pi_mc_cs pi_mc_client_server.c -lm
 *
 * Usage :
 *   mpirun -np 4 ./pi_mc_cs            # 10 secondes, 3 clients
 *   mpirun -np 8 ./pi_mc_cs 10         # 10s, 7 clients
 *   mpirun -np 4 ./pi_mc_cs 30         # 30s, 3 clients
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>

#define PI_REF      3.14159265358979323846
#define BATCH_SIZE  10000000UL     /* 10 millions de points par paquet */
#define TAG_RESULT  10             /* client -> serveur : n_in du paquet */
#define TAG_ORDER   20             /* serveur -> client : 1=continuer 0=stop */

/* ── Generateur xorshift64 ── */
static inline uint64_t xr_next64(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return (*s = x);
}
static inline double xr_double64(uint64_t *s)
{
    return (double)(xr_next64(s) >> 11) * (1.0 / (double)(1ULL << 53));
}

/* ── Simulation d un paquet ── */
static unsigned long simulate_batch(uint64_t *rng, unsigned long n)
{
    unsigned long n_in = 0;
    for (unsigned long i = 0; i < n; i++) {
        double x = xr_double64(rng);
        double y = xr_double64(rng);
        if (x * x + y * y < 1.0)
            n_in++;
    }
    return n_in;
}

/* ── Ecriture CSV ── */
static void write_csv(const char *path, int nclients,
                      double duration, double pi,
                      unsigned long N_total, double t_real,
                      int append)
{
    FILE *fp = fopen(path, append ? "a" : "w");
    if (!fp) return;
    if (!append)
        fprintf(fp, "clients,duree_demandee,pi_approx,erreur_abs,"
                    "erreur_rel_pct,N_total,temps_reel_s,debit_Mpts_s\n");
    double err = fabs(pi - PI_REF);
    fprintf(fp, "%d,%.1f,%.10f,%.6e,%.6e,%lu,%.4f,%.2f\n",
            nclients, duration, pi, err,
            err / PI_REF * 100.0, N_total,
            t_real, (double)N_total / t_real / 1e6);
    fclose(fp);
}

/* ═══════════════════════════════════════════════════════════════
 * SERVEUR — rang 0
 * ═══════════════════════════════════════════════════════════════ */
static void server(int nprocs, double duration)
{
    int        nclients  = nprocs - 1;
    double     t_start   = MPI_Wtime();
    unsigned long N_total  = 0;
    unsigned long N_in     = 0;
    int        active    = nclients;   /* clients encore en vie */

    printf("  [Serveur] demarre — %d clients — duree = %.0f s\n",
           nclients, duration);
    fflush(stdout);

    while (active > 0) {
        MPI_Status status;
        long n_in_batch;

        /* Recevoir le resultat de n importe quel client */
        MPI_Recv(&n_in_batch, 1, MPI_LONG, MPI_ANY_SOURCE,
                 TAG_RESULT, MPI_COMM_WORLD, &status);

        N_in    += (unsigned long)n_in_batch;
        N_total += BATCH_SIZE;

        /* Decider si le client doit continuer */
        double elapsed = MPI_Wtime() - t_start;
        int go = (elapsed < duration) ? 1 : 0;
        MPI_Send(&go, 1, MPI_INT, status.MPI_SOURCE,
                 TAG_ORDER, MPI_COMM_WORLD);

        if (!go) active--;
    }

    double t_total = MPI_Wtime() - t_start;
    double pi = 4.0 * (double)N_in / (double)N_total;
    double err = fabs(pi - PI_REF);

    printf("\n");
    printf("  pi approx        = %.10f\n", pi);
    printf("  pi reference     = %.10f\n", PI_REF);
    printf("  N_total          = %lu  (%lu paquets de %lu pts)\n",
           N_total, N_total / BATCH_SIZE, BATCH_SIZE);
    printf("  N_in             = %lu\n", N_in);
    printf("  erreur absolue   = %.4e\n", err);
    printf("  erreur relative  = %.4e %%\n", err / PI_REF * 100.0);
    printf("  erreur theorique = %.4e  (1/sqrt(N))\n",
           1.0 / sqrt((double)N_total));
    printf("  temps reel       = %.4f s\n", t_total);
    printf("  debit total      = %.2f M pts/s  (%d clients)\n",
           (double)N_total / t_total / 1e6, nclients);
    printf("\n");

    int append = (access("results_mc_cs.csv", F_OK) == 0);
    write_csv("results_mc_cs.csv", nclients, duration,
              pi, N_total, t_total, append);
    printf("  -> results_mc_cs.csv\n\n");
}

/* ═══════════════════════════════════════════════════════════════
 * CLIENT — rangs 1..P-1
 * ═══════════════════════════════════════════════════════════════ */
static void client(int rank)
{
    /* Chaque client a un seed different pour l independance statistique */
    uint64_t rng = 42ULL * (uint64_t)(rank * 6364136223846793005ULL + 1442695040888963407ULL);
    if (rng == 0) rng = 1;

    unsigned long paquets = 0;

    while (1) {
        /* Simuler un paquet de BATCH_SIZE points */
        unsigned long n_in = simulate_batch(&rng, BATCH_SIZE);
        paquets++;

        /* Envoyer le resultat au serveur */
        long n_in_l = (long)n_in;
        MPI_Send(&n_in_l, 1, MPI_LONG, 0, TAG_RESULT, MPI_COMM_WORLD);

        /* Attendre l ordre du serveur */
        int go;
        MPI_Recv(&go, 1, MPI_INT, 0, TAG_ORDER, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        if (!go) break;
    }
}

/* ── Main ─────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs < 2) {
        if (rank == 0)
            fprintf(stderr,
                "Erreur : pi_mc_cs necessite au moins 2 processus "
                "(1 serveur + 1 client)\n");
        MPI_Finalize(); return 1;
    }

    double duration = (argc > 1) ? atof(argv[1]) : 10.0;

    if (rank == 0) {
        printf("=======================================================\n");
        printf("  Calcul de pi — Monte Carlo MPI client/serveur\n");
        printf("  P = %d  (%d clients)  duree = %.0f s\n",
               nprocs, nprocs - 1, duration);
        printf("  Paquet par client = %lu pts\n", BATCH_SIZE);
        printf("=======================================================\n");
        server(nprocs, duration);
    } else {
        client(rank);
    }

    MPI_Finalize();
    return 0;
}
