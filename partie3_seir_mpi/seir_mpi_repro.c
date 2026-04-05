/*
 * seir_mpi_repro.c — Simulation SEIR MPI reproductible
 *
 * Problème de reproductibilité avec rand() global :
 *   rand() maintient un état global par rang. La séquence de tirages
 *   dépend du nombre d'agents locaux et du nombre de transitions.
 *   Ces deux quantités varient avec P → résultats différents pour la
 *   même seed mais un nombre de rangs différent.
 *
 * Solution : PRNG xorshift128+ par agent.
 *   Chaque agent porte son propre état PRNG initialisé une fois à
 *   partir de son identifiant unique (ID global). Ainsi, toutes les
 *   décisions aléatoires d'un agent (position, infection) ne dépendent
 *   que de son ID et de son historique — pas du rang MPI qui le traite.
 *   Les résultats sont donc identiques pour tout P.
 *
 * Compilation :
 *   mpicc -O2 -Wall -std=c99 -o seir_mpi_repro seir_mpi_repro.c -lm
 * Exécution :
 *   mpirun --oversubscribe -np 4 ./seir_mpi_repro
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
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

/* ------------------------------------------------------------------ */
/* PRNG xorshift128+ : rapide, période 2^128-1, qualité statistique   */
/* suffisante pour une simulation épidémiologique.                     */
/* Chaque agent porte son propre état → déterminisme indépendant de P. */
/* ------------------------------------------------------------------ */
typedef struct { uint64_t s0, s1; } RNG;

static inline uint64_t xsh128_next(RNG *rng)
{
    uint64_t s1 = rng->s0;
    uint64_t s0 = rng->s1;
    rng->s0 = s0;
    s1 ^= s1 << 23;
    s1 ^= s1 >> 17;
    s1 ^= s0 ^ (s0 >> 26);
    rng->s1 = s1;
    return s0 + s1;
}

/* double dans [0, 1) — 53 bits de mantisse */
static inline double rng_double(RNG *rng)
{
    return (double)(xsh128_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

static inline int rng_int(RNG *rng, int a, int b)
{
    return a + (int)(xsh128_next(rng) % (uint64_t)(b - a + 1));
}

/* Initialisation de l'état RNG depuis l'ID de l'agent (splitmix64)  */
static void rng_seed_from_id(RNG *rng, uint64_t id)
{
    uint64_t z = id + UINT64_C(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    rng->s0 = z ^ (z >> 31);
    z += UINT64_C(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    rng->s1 = z ^ (z >> 31);
}

/* ------------------------------------------------------------------ */
/* Structure agent (plus grande qu'en version non-repro : +16 octets)  */
/* ------------------------------------------------------------------ */
typedef struct {
    int    status;
    int    time_in_state;
    int    dE, dI, dR;
    int    x, y;
    int    id;        /* identifiant global unique [0, N_AGENTS)      */
    int    _pad;      /* alignement sur 8 octets pour le RNG          */
    RNG    rng;       /* état PRNG propre à l'agent (16 octets)        */
} Agent;

/* ------------------------------------------------------------------ */
static Agent agents[N_AGENTS];
static int   cell_head[GRID_H * GRID_W];
static int   cell_next[N_AGENTS];
static int   new_status[N_AGENTS];
static int   new_time[N_AGENTS];

static int rank, nprocs;
static int local_lo, local_hi, local_n;

static int allgv_cnt[128];
static int allgv_dsp[128];

/* ------------------------------------------------------------------ */
/* Fonctions PRNG wrappers pour l'initialisation (rand() standard)     */
/* Utilisé uniquement dans init_agents() sur le rang 0.               */
/* ------------------------------------------------------------------ */
static inline double rand01_std(void) { return rand() / (double)RAND_MAX; }
static inline int    rand_int_std(int a, int b) { return a + rand() % (b-a+1); }



/* Exponentielle négative via le PRNG de l'agent                      */
static int negExp_rng(RNG *rng, double mean)
{
    double u;
    do { u = rng_double(rng); } while (u >= 1.0);
    int v = (int)ceil(-mean * log(1.0 - u));
    return v < 1 ? 1 : v;
}

/* ------------------------------------------------------------------ */
/* Initialisation                                                       */
/* Exécutée sur le rang 0 avec srand(42) pour la compatibilité avec   */
/* la version séquentielle. Les durées dE/dI/dR sont tirées via le    */
/* PRNG de chaque agent pour garantir la reproductibilité.             */
/* ------------------------------------------------------------------ */
static void init_agents(void)
{
    /* srand(42) est déjà appelé par main() sur le rang 0 */
    for (int i = 0; i < N_AGENTS; i++) {
        agents[i].id            = i;
        agents[i].status        = (i < N_INIT_I) ? ST_I : ST_S;
        agents[i].time_in_state = 0;

        /* Seed du PRNG de l'agent : déterministe, basé sur son ID   */
        rng_seed_from_id(&agents[i].rng, (uint64_t)i);

        /* Durées propres tirées via le PRNG de l'agent               */
        agents[i].dE = negExp_rng(&agents[i].rng, MEAN_E);
        agents[i].dI = negExp_rng(&agents[i].rng, MEAN_I);
        agents[i].dR = negExp_rng(&agents[i].rng, MEAN_R);

        /* Position initiale via rand() standard (rang 0 seulement)   */
        agents[i].x  = rand_int_std(0, GRID_W - 1);
        agents[i].y  = rand_int_std(0, GRID_H - 1);
    }
}

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

static void sync_agents(void)
{
    MPI_Allgatherv(
        &agents[local_lo], allgv_cnt[rank], MPI_BYTE,
        agents, allgv_cnt, allgv_dsp, MPI_BYTE,
        MPI_COMM_WORLD
    );
}

/* ------------------------------------------------------------------ */
/* Déplacement et mise à jour via le PRNG de chaque agent             */
/* → même séquence de tirages quel que soit le rang MPI               */
/* ------------------------------------------------------------------ */
static void step_move_local(void)
{
    for (int i = local_lo; i < local_hi; i++) {
        agents[i].x = rng_int(&agents[i].rng, 0, GRID_W - 1);
        agents[i].y = rng_int(&agents[i].rng, 0, GRID_H - 1);
    }
}

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
                    if (rng_double(&agents[i].rng) < p) {
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
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int base = N_AGENTS / nprocs;
    local_lo = rank * base;
    local_hi = (rank == nprocs - 1) ? N_AGENTS : local_lo + base;
    local_n  = local_hi - local_lo;

    for (int r = 0; r < nprocs; r++) {
        int lo = r * base;
        int hi = (r == nprocs - 1) ? N_AGENTS : lo + base;
        allgv_cnt[r] = (hi - lo) * (int)sizeof(Agent);
        allgv_dsp[r] = lo        * (int)sizeof(Agent);
    }

    /* Init sur rang 0, broadcast */
    if (rank == 0) {
        srand(42);
        init_agents();
    }
    MPI_Bcast(agents, N_AGENTS * (int)sizeof(Agent), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* Pas de srand() par rang : le PRNG est dans chaque agent       */

    FILE *f_counts = NULL, *f_frames = NULL;
    if (rank == 0) {
        char name[64];
        snprintf(name, sizeof(name), "counts_repro_%d.csv", nprocs);
        f_counts = fopen(name, "w");
        snprintf(name, sizeof(name), "frames_repro_%d.bin", nprocs);
        f_frames = fopen(name, "wb");
        int hdr[2] = { GRID_W, GRID_H };
        fwrite(hdr, sizeof(int), 2, f_frames);
        fprintf(f_counts, "step,S,E,I,R\n");
    }

    rebuild_grid();
    if (rank == 0) {
        int s, e, ii, r2;
        get_counts(&s, &e, &ii, &r2);
        fprintf(f_counts, "0,%d,%d,%d,%d\n", s, e, ii, r2);
        write_frame(f_frames);
        printf("=== SEIR MPI reproductible | P=%d ===\n", nprocs);
        printf("Jour %3d : S=%5d  E=%4d  I=%4d  R=%5d\n", 0, s, e, ii, r2);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    struct timeval t0, t1;
    if (rank == 0) gettimeofday(&t0, NULL);

    for (int step = 1; step <= N_STEPS; step++) {
        step_move_local();
        sync_agents();
        rebuild_grid();
        step_update_local();
        sync_agents();

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
        printf("\nTemps de calcul : %.4f s  (P=%d, reproductible)\n",
               elapsed, nprocs);
        fclose(f_counts);
        fclose(f_frames);
    }

    MPI_Finalize();
    return 0;
}
