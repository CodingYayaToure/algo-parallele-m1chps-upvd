/*
 * seir_mpi.c — Simulation SEIR multi-agents (version MPI, non reproductible)
 *
 * Stratégie de parallélisation : grille répliquée, agents partitionnés par ID.
 *
 * Chaque rang MPI possède une tranche d'agents [lo, hi[.
 * La grille complète est répliquée sur chaque rang pour le calcul du
 * voisinage de Moore sans communication supplémentaire.
 *
 * Boucle par pas de temps :
 *   1. Chaque rang déplace ses agents locaux (rand() propre à chaque rang)
 *   2. MPI_Allgatherv : synchronisation des nouvelles positions
 *   3. Reconstruction de la grille complète sur chaque rang
 *   4. Chaque rang calcule les nouveaux états de ses agents locaux
 *   5. MPI_Allgatherv : synchronisation des nouveaux états
 *   6. Rang 0 : comptage et écriture des sorties
 *
 * Non-reproductible : voir rapport.md pour l'explication.
 *
 * Compilation :
 *   mpicc -O2 -Wall -std=c99 -o seir_mpi seir_mpi.c -lm
 * Exécution :
 *   mpirun --oversubscribe -np 4 ./seir_mpi
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

/* ------------------------------------------------------------------ */
#define GRID_W      300
#define GRID_H      300
#define N_AGENTS    20000
#define N_INIT_I    20
#define N_STEPS     730
#define BETA        0.5
#define MEAN_E      3.0
#define MEAN_I      7.0
#define MEAN_R      365.0
#define FRAME_EVERY 5

#define ST_S 0
#define ST_E 1
#define ST_I 2
#define ST_R 3

typedef struct {
    int status;
    int time_in_state;
    int dE, dI, dR;
    int x, y;
} Agent;

/* ------------------------------------------------------------------ */
/* Données globales répliquées sur tous les rangs                      */
/* ------------------------------------------------------------------ */
static Agent agents[N_AGENTS];
static int   cell_head[GRID_H * GRID_W];
static int   cell_next[N_AGENTS];
static int   new_status[N_AGENTS];
static int   new_time[N_AGENTS];

static int rank, nprocs;
static int local_lo, local_hi, local_n;  /* tranche locale d'agents  */

/* Paramètres pour MPI_Allgatherv */
static int allgv_cnt[128];   /* nombre d'octets envoyés par rang     */
static int allgv_dsp[128];   /* déplacement en octets dans le buffer */

/* ------------------------------------------------------------------ */
/* PRNG (rand() de la librairie standard)                              */
/* ------------------------------------------------------------------ */
static inline double rand01(void)    { return rand() / (double)RAND_MAX; }
static inline int rand_int(int a, int b) { return a + rand() % (b-a+1); }

static int negExp(double mean)
{
    double u;
    do { u = rand01(); } while (u >= 1.0);
    int v = (int)ceil(-mean * log(1.0 - u));
    return v < 1 ? 1 : v;
}

/* ------------------------------------------------------------------ */
/* Initialisation — exécutée sur le rang 0 puis diffusée              */
/* ------------------------------------------------------------------ */
static void init_agents(void)
{
    for (int i = 0; i < N_AGENTS; i++) {
        agents[i].status        = (i < N_INIT_I) ? ST_I : ST_S;
        agents[i].time_in_state = 0;
        agents[i].dE = negExp(MEAN_E);
        agents[i].dI = negExp(MEAN_I);
        agents[i].dR = negExp(MEAN_R);
        agents[i].x  = rand_int(0, GRID_W - 1);
        agents[i].y  = rand_int(0, GRID_H - 1);
    }
}

/* ------------------------------------------------------------------ */
/* Grille                                                              */
/* ------------------------------------------------------------------ */
static void rebuild_grid(void)
{
    memset(cell_head, -1, sizeof(cell_head));
    for (int i = 0; i < N_AGENTS; i++) {
        int idx        = agents[i].y * GRID_W + agents[i].x;
        cell_next[i]   = cell_head[idx];
        cell_head[idx] = i;
    }
}

/* ------------------------------------------------------------------ */
/* Voisinage de Moore (9 cellules, torique)                            */
/* ------------------------------------------------------------------ */
static int count_I_moore(int cx, int cy)
{
    int count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        int ny = ((cy + dy) % GRID_H + GRID_H) % GRID_H;
        for (int dx = -1; dx <= 1; dx++) {
            int nx = ((cx + dx) % GRID_W + GRID_W) % GRID_W;
            for (int ag = cell_head[ny * GRID_W + nx]; ag != -1; ag = cell_next[ag])
                if (agents[ag].status == ST_I)
                    count++;
        }
    }
    return count;
}

/* ------------------------------------------------------------------ */
/* Synchronisation de la tranche locale via Allgatherv                 */
/* ------------------------------------------------------------------ */
static void sync_agents(void)
{
    MPI_Allgatherv(
        &agents[local_lo], allgv_cnt[rank], MPI_BYTE,
        agents, allgv_cnt, allgv_dsp, MPI_BYTE,
        MPI_COMM_WORLD
    );
}

/* ------------------------------------------------------------------ */
/* Déplacement des agents locaux                                       */
/* ------------------------------------------------------------------ */
static void step_move_local(void)
{
    for (int i = local_lo; i < local_hi; i++) {
        agents[i].x = rand_int(0, GRID_W - 1);
        agents[i].y = rand_int(0, GRID_H - 1);
    }
}

/* ------------------------------------------------------------------ */
/* Mise à jour synchrone des agents locaux                             */
/* La grille est lue en lecture seule (états non modifiés pendant      */
/* le calcul). Application simultanée après la boucle.                */
/* ------------------------------------------------------------------ */
static void step_update_local(void)
{
    for (int i = local_lo; i < local_hi; i++) {
        new_status[i] = agents[i].status;
        new_time[i]   = agents[i].time_in_state + 1;

        switch (agents[i].status) {
            case ST_S: {
                int ni = count_I_moore(agents[i].x, agents[i].y);
                if (ni > 0) {
                    double p = 1.0 - exp(-BETA * (double)ni);
                    if (rand01() < p) {
                        new_status[i] = ST_E;
                        new_time[i]   = 0;
                    }
                }
                break;
            }
            case ST_E:
                if (new_time[i] >= agents[i].dE) {
                    new_status[i] = ST_I;
                    new_time[i]   = 0;
                }
                break;
            case ST_I:
                if (new_time[i] >= agents[i].dI) {
                    new_status[i] = ST_R;
                    new_time[i]   = 0;
                }
                break;
            case ST_R:
                if (new_time[i] >= agents[i].dR) {
                    new_status[i] = ST_S;
                    new_time[i]   = 0;
                }
                break;
        }
    }
    for (int i = local_lo; i < local_hi; i++) {
        agents[i].status        = new_status[i];
        agents[i].time_in_state = new_time[i];
    }
}

/* ------------------------------------------------------------------ */
/* Comptage et sortie (rang 0 uniquement)                              */
/* ------------------------------------------------------------------ */
static void get_counts(int *s, int *e, int *ii, int *r)
{
    *s = *e = *ii = *r = 0;
    for (int i = 0; i < N_AGENTS; i++) {
        switch (agents[i].status) {
            case ST_S: (*s)++;  break;
            case ST_E: (*e)++;  break;
            case ST_I: (*ii)++; break;
            case ST_R: (*r)++;  break;
        }
    }
}

static void write_frame(FILE *fp)
{
    static int cnt[GRID_H * GRID_W][4];
    memset(cnt, 0, sizeof(cnt));
    for (int i = 0; i < N_AGENTS; i++)
        cnt[agents[i].y * GRID_W + agents[i].x][agents[i].status]++;
    unsigned char frame[GRID_H * GRID_W];
    for (int idx = 0; idx < GRID_H * GRID_W; idx++) {
        if      (cnt[idx][ST_I]) frame[idx] = 3;
        else if (cnt[idx][ST_E]) frame[idx] = 2;
        else if (cnt[idx][ST_R]) frame[idx] = 4;
        else if (cnt[idx][ST_S]) frame[idx] = 1;
        else                      frame[idx] = 0;
    }
    fwrite(frame, 1, GRID_H * GRID_W, fp);
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* ---- Partitionnement des agents par ID ----------------------- */
    int base  = N_AGENTS / nprocs;
    local_lo  = rank * base;
    local_hi  = (rank == nprocs - 1) ? N_AGENTS : local_lo + base;
    local_n   = local_hi - local_lo;

    /* Calcul des paramètres Allgatherv                               */
    for (int r = 0; r < nprocs; r++) {
        int lo = r * base;
        int hi = (r == nprocs - 1) ? N_AGENTS : lo + base;
        allgv_cnt[r] = (hi - lo) * (int)sizeof(Agent);
        allgv_dsp[r] = lo        * (int)sizeof(Agent);
    }

    /* ---- Initialisation (rang 0) puis broadcast ------------------- */
    if (rank == 0) {
        srand(42);
        init_agents();
    }
    MPI_Bcast(agents, N_AGENTS * (int)sizeof(Agent), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* Chaque rang utilise une graine différente pour la simulation   */
    /* → non reproductible si P change (expliqué dans le rapport)     */
    srand(42 + rank + 1);

    /* ---- Fichiers de sortie (rang 0) ------------------------------ */
    FILE *f_counts = NULL, *f_frames = NULL;
    if (rank == 0) {
        char name[64];
        snprintf(name, sizeof(name), "counts_mpi_%d.csv", nprocs);
        f_counts = fopen(name, "w");
        snprintf(name, sizeof(name), "frames_mpi_%d.bin", nprocs);
        f_frames = fopen(name, "wb");
        int hdr[2] = { GRID_W, GRID_H };
        fwrite(hdr, sizeof(int), 2, f_frames);
        fprintf(f_counts, "step,S,E,I,R\n");
    }

    /* ---- Step 0 : état initial ------------------------------------ */
    rebuild_grid();
    if (rank == 0) {
        int s, e, ii, r2;
        get_counts(&s, &e, &ii, &r2);
        fprintf(f_counts, "0,%d,%d,%d,%d\n", s, e, ii, r2);
        write_frame(f_frames);
        printf("=== SEIR MPI | P=%d | agents=%d | grille=%dx%d ===\n",
               nprocs, N_AGENTS, GRID_W, GRID_H);
        printf("Jour %3d : S=%5d  E=%4d  I=%4d  R=%5d\n", 0, s, e, ii, r2);
    }

    /* ---- Boucle principale ---------------------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);
    struct timeval t0, t1;
    if (rank == 0) gettimeofday(&t0, NULL);

    for (int step = 1; step <= N_STEPS; step++) {

        /* 1. Déplacement des agents locaux */
        step_move_local();

        /* 2. Sync des nouvelles positions sur tous les rangs */
        sync_agents();

        /* 3. Reconstruction de la grille complète (sur chaque rang) */
        rebuild_grid();

        /* 4. Calcul synchrone des nouveaux états (agents locaux) */
        step_update_local();

        /* 5. Sync des nouveaux états sur tous les rangs */
        sync_agents();

        /* 6. Sortie (rang 0) */
        if (rank == 0) {
            int s, e, ii, r2;
            get_counts(&s, &e, &ii, &r2);
            fprintf(f_counts, "%d,%d,%d,%d,%d\n", step, s, e, ii, r2);
            if (step % FRAME_EVERY == 0) write_frame(f_frames);
            if (step % 30 == 0 || step == 1)
                printf("Jour %3d : S=%5d  E=%4d  I=%4d  R=%5d\n",
                       step, s, e, ii, r2);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        gettimeofday(&t1, NULL);
        double elapsed = (t1.tv_sec  - t0.tv_sec)
                       + (t1.tv_usec - t0.tv_usec) * 1e-6;
        printf("\nTemps de calcul : %.4f s  (P=%d)\n", elapsed, nprocs);
        fclose(f_counts);
        fclose(f_frames);
    }

    MPI_Finalize();
    return 0;
}
