/*
 * seir_seq.c — Simulation SEIR multi-agents stochastique (version séquentielle)
 *
 * Grille torique 300x300, 20000 agents, modèle SEIR synchrone.
 * Référence pour les mesures de speedup MPI.
 *
 * Compilation :
 *   gcc -O2 -Wall -std=c99 -o seir_seq seir_seq.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* ------------------------------------------------------------------ */
/* Paramètres du modèle                                                */
/* ------------------------------------------------------------------ */
#define GRID_W      300          /* largeur de la grille              */
#define GRID_H      300          /* hauteur de la grille              */
#define N_AGENTS    20000        /* nombre total d'individus          */
#define N_INIT_I    20           /* individus infectieux au départ    */
#define N_STEPS     730          /* durée de la simulation (jours)    */
#define BETA        0.5          /* force d'infection                 */
#define MEAN_E      3.0          /* durée moyenne en état E (jours)   */
#define MEAN_I      7.0          /* durée moyenne en état I (jours)   */
#define MEAN_R      365.0        /* durée moyenne en état R (jours)   */
#define FRAME_EVERY 5            /* fréquence d'écriture des frames   */

/* États discrets */
#define ST_S 0   /* Susceptible  */
#define ST_E 1   /* Exposé       */
#define ST_I 2   /* Infectieux   */
#define ST_R 3   /* Rétabli      */

/* ------------------------------------------------------------------ */
/* Structure agent                                                      */
/* ------------------------------------------------------------------ */
typedef struct {
    int status;           /* état courant : ST_S, ST_E, ST_I, ST_R  */
    int time_in_state;    /* jours passés dans l'état courant        */
    int dE, dI, dR;       /* durées propres à l'individu (fixes)     */
    int x, y;             /* position sur la grille                  */
} Agent;

/* ------------------------------------------------------------------ */
/* Données globales                                                     */
/* ------------------------------------------------------------------ */
static Agent agents[N_AGENTS];

/* Grille : liste chaînée par cellule pour un accès O(1) aux voisins  */
static int cell_head[GRID_H * GRID_W];  /* tête de liste par cellule */
static int cell_next[N_AGENTS];         /* maillon suivant            */

/* Tableaux temporaires pour la mise à jour synchrone                  */
static int new_status[N_AGENTS];
static int new_time[N_AGENTS];

/* ------------------------------------------------------------------ */
/* Générateur de nombres aléatoires (librairie standard C)             */
/* ------------------------------------------------------------------ */
static inline double rand01(void)
{
    return rand() / (double)RAND_MAX;
}

/* Loi exponentielle négative — paramètre = moyenne en jours.
 * genrand_real2() correspond à un double dans [0, 1[ ;
 * on rejette la valeur 1.0 pour éviter log(0).               */
static int negExp(double mean)
{
    double u;
    do { u = rand01(); } while (u >= 1.0);
    int v = (int)ceil(-mean * log(1.0 - u));
    return v < 1 ? 1 : v;
}

static inline int rand_int(int a, int b)
{
    return a + rand() % (b - a + 1);
}

/* ------------------------------------------------------------------ */
/* Initialisation                                                       */
/* ------------------------------------------------------------------ */
static void init_agents(void)
{
    for (int i = 0; i < N_AGENTS; i++) {
        /* Les 20 premiers agents sont infectieux, les autres sains   */
        agents[i].status        = (i < N_INIT_I) ? ST_I : ST_S;
        agents[i].time_in_state = 0;
        /* Durées propres tirées une seule fois à la création         */
        agents[i].dE = negExp(MEAN_E);
        agents[i].dI = negExp(MEAN_I);
        agents[i].dR = negExp(MEAN_R);
        /* Position aléatoire sur la grille                           */
        agents[i].x  = rand_int(0, GRID_W - 1);
        agents[i].y  = rand_int(0, GRID_H - 1);
    }
}

/* ------------------------------------------------------------------ */
/* Gestion de la grille (listes chaînées)                              */
/* ------------------------------------------------------------------ */
static void rebuild_grid(void)
{
    memset(cell_head, -1, sizeof(cell_head));
    for (int i = 0; i < N_AGENTS; i++) {
        int idx       = agents[i].y * GRID_W + agents[i].x;
        cell_next[i]  = cell_head[idx];
        cell_head[idx]= i;
    }
}

/* ------------------------------------------------------------------ */
/* Déplacement : saut global aléatoire (pas un déplacement local)      */
/* ------------------------------------------------------------------ */
static void step_move(void)
{
    for (int i = 0; i < N_AGENTS; i++) {
        agents[i].x = rand_int(0, GRID_W - 1);
        agents[i].y = rand_int(0, GRID_H - 1);
    }
}

/* ------------------------------------------------------------------ */
/* Comptage des infectieux dans le voisinage de Moore (9 cellules)     */
/* L'espace est torique : bords périodiques.                           */
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
/* Mise à jour des états — modèle SYNCHRONE                            */
/*                                                                      */
/* Planification : déplacement à t, calcul des états à t+1.           */
/* Pour garantir la synchronicité, on calcule d'abord tous les         */
/* nouveaux états (en lisant l'état courant), puis on les applique     */
/* simultanément. Cela évite qu'un agent nouvellement infectieux        */
/* influence ses voisins dans le même pas de temps.                    */
/* ------------------------------------------------------------------ */
static void step_update(void)
{
    /* Phase 1 : calcul des nouveaux états (lecture seule de agents[]) */
    for (int i = 0; i < N_AGENTS; i++) {
        new_status[i] = agents[i].status;
        new_time[i]   = agents[i].time_in_state + 1;  /* 1 jour de plus */

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

    /* Phase 2 : application simultanée des nouveaux états            */
    for (int i = 0; i < N_AGENTS; i++) {
        agents[i].status        = new_status[i];
        agents[i].time_in_state = new_time[i];
    }
}

/* ------------------------------------------------------------------ */
/* Comptage S/E/I/R                                                    */
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

/* ------------------------------------------------------------------ */
/* Écriture d'une frame binaire                                        */
/* Format : 1 octet/cellule, état dominant (I > E > R > S > vide)     */
/* Encodage : 0=vide 1=S 2=E 3=I 4=R                                  */
/* ------------------------------------------------------------------ */
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
/* Point d'entrée                                                       */
/* ------------------------------------------------------------------ */
int main(void)
{
    struct timeval t0, t1;

    srand(42);  /* graine imposée par le sujet */

    FILE *f_counts = fopen("counts_seq.csv", "w");
    FILE *f_frames = fopen("frames_seq.bin", "wb");
    if (!f_counts || !f_frames) { perror("fopen"); return 1; }

    /* En-tête binaire : dimensions W H */
    int hdr[2] = { GRID_W, GRID_H };
    fwrite(hdr, sizeof(int), 2, f_frames);
    fprintf(f_counts, "step,S,E,I,R\n");

    init_agents();
    rebuild_grid();

    int s, e, ii, r;
    get_counts(&s, &e, &ii, &r);
    fprintf(f_counts, "0,%d,%d,%d,%d\n", s, e, ii, r);
    write_frame(f_frames);

    printf("=== SEIR sequentiel | agents=%d | grille=%dx%d | steps=%d ===\n",
           N_AGENTS, GRID_W, GRID_H, N_STEPS);
    printf("Jour %3d : S=%5d  E=%4d  I=%4d  R=%5d\n", 0, s, e, ii, r);

    gettimeofday(&t0, NULL);

    for (int step = 1; step <= N_STEPS; step++) {
        step_move();
        rebuild_grid();
        step_update();

        get_counts(&s, &e, &ii, &r);
        fprintf(f_counts, "%d,%d,%d,%d,%d\n", step, s, e, ii, r);
        if (step % FRAME_EVERY == 0)
            write_frame(f_frames);
        if (step % 30 == 0 || step == 1)
            printf("Jour %3d : S=%5d  E=%4d  I=%4d  R=%5d\n", step, s, e, ii, r);
    }

    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec  - t0.tv_sec)
                   + (t1.tv_usec - t0.tv_usec) * 1e-6;
    printf("\nTemps de calcul : %.4f s\n", elapsed);

    fclose(f_counts);
    fclose(f_frames);
    return 0;
}
