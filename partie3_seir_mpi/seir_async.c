/*
 * seir_async.c — Simulation SEIR à ordonnancement asynchrone
 *
 * Planification asynchrone (question Q4 du sujet) :
 *   L'ordre d'exécution des agents est aléatoire à chaque pas.
 *   Lorsqu'un agent est choisi, il se déplace ET met à jour son état
 *   immédiatement, en voyant l'état courant du monde (y compris les
 *   changements effectués par les agents précédents dans ce même pas).
 *   Il n'y a pas de mémorisation / double buffer.
 *
 * Différence avec la version synchrone :
 *   - Synchrone : tous lisent l'état à t, tous écrivent l'état à t+1.
 *   - Asynchrone : chaque agent lit l'état "maintenant", qui inclut
 *     les transitions déjà effectuées dans ce pas de temps.
 *
 * Problème pour la parallélisation MPI :
 *   Un asynchronisme global implique qu'un agent activé sur le rang A
 *   voit immédiatement le changement d'un agent sur le rang B. Cela
 *   nécessiterait une communication à chaque activation individuelle
 *   (N communications par pas de temps), ce qui est prohibitif.
 *   On peut implémenter un asynchronisme LOCAL par rang avec
 *   synchronisation en fin de pas — ce n'est pas équivalent au modèle
 *   global mais c'est la seule approximation raisonnable avec MPI.
 *   Cette version est donc présentée en séquentiel.
 *
 * Compilation :
 *   gcc -O2 -Wall -std=c99 -o seir_async seir_async.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

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

static Agent agents[N_AGENTS];
static int   cell_head[GRID_H * GRID_W];
static int   cell_next[N_AGENTS];
static int   order[N_AGENTS];   /* ordre d'activation aléatoire      */

/* ------------------------------------------------------------------ */
static inline double rand01(void)        { return rand() / (double)RAND_MAX; }
static inline int rand_int(int a, int b) { return a + rand() % (b - a + 1); }

static int negExp(double mean)
{
    double u;
    do { u = rand01(); } while (u >= 1.0);
    int v = (int)ceil(-mean * log(1.0 - u));
    return v < 1 ? 1 : v;
}

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
        order[i]     = i;
    }
}

/* ------------------------------------------------------------------ */
/* Grille : on gère les insertions/suppressions individuellement       */
/* pour refléter la position courante après chaque déplacement.        */
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

/* Suppression d'un agent de sa cellule courante — O(agents/cellule)  */
static void remove_from_cell(int id)
{
    int idx = agents[id].y * GRID_W + agents[id].x;
    if (cell_head[idx] == id) {
        cell_head[idx] = cell_next[id];
        return;
    }
    int prev = cell_head[idx];
    while (prev != -1 && cell_next[prev] != id)
        prev = cell_next[prev];
    if (prev != -1)
        cell_next[prev] = cell_next[id];
}

/* Insertion d'un agent dans sa nouvelle cellule — O(1)               */
static void insert_in_cell(int id)
{
    int idx        = agents[id].y * GRID_W + agents[id].x;
    cell_next[id]  = cell_head[idx];
    cell_head[idx] = id;
}

/* ------------------------------------------------------------------ */
/* Comptage I dans le voisinage de Moore (état courant, pas copie)     */
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
/* Mélange de Fisher-Yates pour l'ordre aléatoire des agents           */
/* ------------------------------------------------------------------ */
static void shuffle_order(void)
{
    for (int i = N_AGENTS - 1; i > 0; i--) {
        int j = rand_int(0, i);
        int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }
}

/* ------------------------------------------------------------------ */
/* Un pas de temps asynchrone                                          */
/* Pour chaque agent, dans un ordre aléatoire :                        */
/*   1. On le retire de sa cellule courante                            */
/*   2. On le place dans une nouvelle cellule aléatoire                */
/*   3. On calcule et applique immédiatement son nouvel état            */
/* Les agents suivants voient donc l'état mis à jour de cet agent.    */
/* ------------------------------------------------------------------ */
static void step_async(void)
{
    shuffle_order();

    for (int k = 0; k < N_AGENTS; k++) {
        int     id = order[k];
        Agent  *a  = &agents[id];

        /* Déplacement : mise à jour immédiate de la grille           */
        remove_from_cell(id);
        a->x = rand_int(0, GRID_W - 1);
        a->y = rand_int(0, GRID_H - 1);
        insert_in_cell(id);

        /* Mise à jour de l'état (voit le monde courant)              */
        a->time_in_state++;

        switch (a->status) {
            case ST_S: {
                int ni = count_I_moore(a->x, a->y);
                if (ni > 0) {
                    double p = 1.0 - exp(-BETA * (double)ni);
                    if (rand01() < p) {
                        a->status        = ST_E;
                        a->time_in_state = 0;
                    }
                }
                break;
            }
            case ST_E:
                if (a->time_in_state >= a->dE) {
                    a->status        = ST_I;
                    a->time_in_state = 0;
                }
                break;
            case ST_I:
                if (a->time_in_state >= a->dI) {
                    a->status        = ST_R;
                    a->time_in_state = 0;
                }
                break;
            case ST_R:
                if (a->time_in_state >= a->dR) {
                    a->status        = ST_S;
                    a->time_in_state = 0;
                }
                break;
        }
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
int main(void)
{
    struct timeval t0, t1;

    srand(42);

    FILE *f_counts = fopen("counts_async.csv", "w");
    FILE *f_frames = fopen("frames_async.bin", "wb");
    if (!f_counts || !f_frames) { perror("fopen"); return 1; }

    int hdr[2] = { GRID_W, GRID_H };
    fwrite(hdr, sizeof(int), 2, f_frames);
    fprintf(f_counts, "step,S,E,I,R\n");

    init_agents();
    rebuild_grid();

    int s, e, ii, r;
    get_counts(&s, &e, &ii, &r);
    fprintf(f_counts, "0,%d,%d,%d,%d\n", s, e, ii, r);
    write_frame(f_frames);

    printf("=== SEIR asynchrone | agents=%d | grille=%dx%d | steps=%d ===\n",
           N_AGENTS, GRID_W, GRID_H, N_STEPS);
    printf("Jour %3d : S=%5d  E=%4d  I=%4d  R=%5d\n", 0, s, e, ii, r);

    gettimeofday(&t0, NULL);

    for (int step = 1; step <= N_STEPS; step++) {
        step_async();

        get_counts(&s, &e, &ii, &r);
        fprintf(f_counts, "%d,%d,%d,%d,%d\n", step, s, e, ii, r);
        if (step % FRAME_EVERY == 0) write_frame(f_frames);
        if (step % 30 == 0 || step == 1)
            printf("Jour %3d : S=%5d  E=%4d  I=%4d  R=%5d\n", step, s, e, ii, r);
    }

    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec  - t0.tv_sec)
                   + (t1.tv_usec - t0.tv_usec) * 1e-6;
    printf("\nTemps total : %.4f s\n", elapsed);

    fclose(f_counts);
    fclose(f_frames);
    return 0;
}
