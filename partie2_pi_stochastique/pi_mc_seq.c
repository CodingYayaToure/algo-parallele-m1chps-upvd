/*
 * pi_mc_seq.c — Calcul de pi par Monte Carlo (version sequentielle)
 *
 * Principe :
 *   On tire N points (x,y) uniformement dans [0,1]^2.
 *   Si x^2 + y^2 < 1  ->  le point est dans le quart de disque.
 *   pi/4 = N_in / N_total  =>  pi = 4 * N_in / N_total
 *
 * Convergence : erreur proportionnelle a 1/sqrt(N) (loi des grands nombres).
 * Generateur   : xorshift64 (rapide, periode 2^64-1, sans etat global).
 *
 * Compilation :
 *   gcc -O2 -Wall -std=c99 -march=native -o pi_mc_seq pi_mc_seq.c -lm
 *
 * Usage :
 *   ./pi_mc_seq                    # N = 10 millions (defaut)
 *   ./pi_mc_seq 1000000000         # N personnalise
 *   ./pi_mc_seq convergence        # etude de convergence
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define PI_REF   3.14159265358979323846
#define N_DEFAULT 10000000UL

/* ── Generateur xorshift64 (Marsaglia 2003) ──────────────────────────── */
static uint64_t xr_state = 0;

static void xr_seed(uint64_t s)
{
    xr_state = s ? s : 1ULL;
}

static inline uint64_t xr_next(void)
{
    uint64_t x = xr_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return (xr_state = x);
}

/* Tire un double dans [0,1) */
static inline double xr_double(void)
{
    return (double)(xr_next() >> 11) * (1.0 / (double)(1ULL << 53));
}

/* ── Mesure du temps ────────────────────────────────────────────────── */
static double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── Simulation Monte Carlo ─────────────────────────────────────────── */
static double monte_carlo_pi(unsigned long N, unsigned long *n_in_out)
{
    unsigned long n_in = 0;
    for (unsigned long i = 0; i < N; i++) {
        double x = xr_double();
        double y = xr_double();
        if (x * x + y * y < 1.0)
            n_in++;
    }
    if (n_in_out) *n_in_out = n_in;
    return 4.0 * (double)n_in / (double)N;
}

/* ── Ecriture CSV ───────────────────────────────────────────────────── */
static void write_csv(const char *path, unsigned long N, double pi,
                      double t, int append)
{
    FILE *fp = fopen(path, append ? "a" : "w");
    if (!fp) return;
    if (!append)
        fprintf(fp, "N,pi_approx,erreur_abs,erreur_rel_pct,"
                    "erreur_theorique,temps_s\n");
    double err  = fabs(pi - PI_REF);
    double err_th = 1.0 / sqrt((double)N);   /* ordre de grandeur theorique */
    fprintf(fp, "%lu,%.10f,%.6e,%.6e,%.6e,%.6f\n",
            N, pi, err, err / PI_REF * 100.0, err_th, t);
    fclose(fp);
}

/* ── Etude de convergence ───────────────────────────────────────────── */
static void convergence_study(void)
{
    unsigned long ns[] = {
        100, 1000, 10000, 100000, 1000000,
        10000000, 100000000, 1000000000UL
    };
    int m = (int)(sizeof(ns) / sizeof(ns[0]));

    printf("\n=== Etude de convergence — Monte Carlo ===\n");
    printf("%-12s  %-14s  %-12s  %-14s  %s\n",
           "N", "pi approx", "erreur abs", "1/sqrt(N)", "temps (s)");
    printf("%s\n",
           "-------------------------------------------------------------");

    FILE *fp = fopen("results_mc_convergence.csv", "w");
    if (fp)
        fprintf(fp, "N,pi_approx,erreur_abs,erreur_theorique,temps_s\n");

    for (int i = 0; i < m; i++) {
        xr_seed(42ULL);
        double t0 = get_time();
        double pi = monte_carlo_pi(ns[i], NULL);
        double dt = get_time() - t0;
        double err = fabs(pi - PI_REF);
        double eth = 1.0 / sqrt((double)ns[i]);

        printf("%-12lu  %-14.10f  %-12.4e  %-14.4e  %.4f\n",
               ns[i], pi, err, eth, dt);

        if (fp)
            fprintf(fp, "%lu,%.10f,%.4e,%.4e,%.6f\n",
                    ns[i], pi, err, eth, dt);
    }
    if (fp) fclose(fp);
    printf("\n  -> results_mc_convergence.csv\n");
}

/* ── Main ───────────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    printf("=======================================================\n");
    printf("  Calcul de pi — Monte Carlo (sequentiel)\n");
    printf("  pi/4 = N_in / N_total\n");
    printf("  Reference : pi = %.15f\n", PI_REF);
    printf("  Generateur : xorshift64, seed = 42\n");
    printf("=======================================================\n");

    if (argc > 1 && strcmp(argv[1], "convergence") == 0) {
        xr_seed(42ULL);
        convergence_study();
        return 0;
    }

    unsigned long N = (argc > 1) ? (unsigned long)atol(argv[1]) : N_DEFAULT;

    printf("\n  N = %lu points\n", N);
    xr_seed(42ULL);

    unsigned long n_in = 0;
    double t0 = get_time();
    double pi = monte_carlo_pi(N, &n_in);
    double dt = get_time() - t0;

    double err = fabs(pi - PI_REF);
    printf("\n  pi approx        = %.10f\n", pi);
    printf("  pi reference     = %.10f\n", PI_REF);
    printf("  N_in             = %lu / %lu\n", n_in, N);
    printf("  erreur absolue   = %.4e\n", err);
    printf("  erreur relative  = %.4e %%\n", err / PI_REF * 100.0);
    printf("  erreur theorique = %.4e  (1/sqrt(N))\n",
           1.0 / sqrt((double)N));
    printf("  temps de calcul  = %.6f s\n", dt);
    printf("  debit            = %.2f M pts/s\n",
           (double)N / dt / 1e6);

    write_csv("results_mc_seq.csv", N, pi, dt, 0);
    printf("\n  -> results_mc_seq.csv\n\n");

    return 0;
}
