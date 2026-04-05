#define _POSIX_C_SOURCE 200809L
/*
 * pi_seq.c — Calcul de π par la méthode des trapèzes (version séquentielle)
 *
 * Formule :
 *   π = 4 × ∫₀¹ 1/(1+x²) dx
 *
 * Méthode des trapèzes composites :
 *   ∫₀¹ f(x) dx ≈ Δx × [f(x₀)/2 + f(x₁) + ... + f(x_{n-1}) + f(xₙ)/2]
 *   avec Δx = 1/n,  xᵢ = i × Δx
 *
 * Précision numérique : sommation de Kahan pour réduire l'erreur d'arrondi
 * avec n = 1 milliard de points.
 *
 * Compilation :
 *   gcc -O2 -Wall -std=c99 -march=native -o pi_seq pi_seq.c -lm
 *
 * Usage :
 *   ./pi_seq                   # n = 5000 et n = 1 000 000 000
 *   ./pi_seq 1000000           # n personnalisé
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* Valeur de référence (50 décimales) */
#define PI_REF 3.14159265358979323846264338327950288419716939937510

/* ------------------------------------------------------------------ */
/* Intégrand : f(x) = 1 / (1 + x²)                                   */
/* ------------------------------------------------------------------ */
static inline double f(double x)
{
    return 1.0 / (1.0 + x * x);
}

/* ------------------------------------------------------------------ */
/* Mesure du temps (nanosecondes → secondes)                           */
/* ------------------------------------------------------------------ */
static double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/* Méthode des trapèzes avec sommation de Kahan                        */
/*                                                                      */
/* Algorithme de Kahan : corrige l'erreur d'arrondi accumulée lors     */
/* de l'addition de n grands nombres en virgule flottante.             */
/* Pour n = 1 milliard, l'amélioration de précision est significative. */
/* ------------------------------------------------------------------ */
static double trapezes_kahan(long n)
{
    const double dx = 1.0 / (double)n;
    double sum  = 0.0;   /* accumulateur principal        */
    double comp = 0.0;   /* compensation d'erreur (Kahan) */

    /* Contribution des extrémités : f(0)/2 + f(1)/2 */
    double y = 0.5 * (f(0.0) + f(1.0)) - comp;
    double t = sum + y;
    comp = (t - sum) - y;
    sum  = t;

    /* Points intérieurs : f(x₁) + f(x₂) + ... + f(x_{n-1}) */
    for (long i = 1; i < n; i++) {
        double xi = (double)i * dx;
        y = f(xi) - comp;
        t = sum + y;
        comp = (t - sum) - y;
        sum  = t;
    }

    return 4.0 * sum * dx;
}

/* ------------------------------------------------------------------ */
/* Écriture des résultats en CSV                                        */
/* Format : n,pi_approx,erreur_abs,erreur_rel_pct,temps_s              */
/* ------------------------------------------------------------------ */
static void write_csv(const char *path, long n, double pi, double t,
                      int append)
{
    FILE *fp = fopen(path, append ? "a" : "w");
    if (!fp) { perror(path); return; }

    if (!append)
        fprintf(fp, "n,pi_approx,erreur_abs,erreur_rel_pct,temps_s\n");

    double err_abs = fabs(pi - PI_REF);
    double err_rel = err_abs / PI_REF * 100.0;
    fprintf(fp, "%ld,%.15f,%.2e,%.2e,%.6f\n",
            n, pi, err_abs, err_rel, t);
    fclose(fp);
}

/* ------------------------------------------------------------------ */
/* Affichage formaté d'un résultat                                      */
/* ------------------------------------------------------------------ */
static void print_result(long n, double pi, double t)
{
    double err = fabs(pi - PI_REF);
    printf("  n = %12ld  |  π ≈ %.15f  |  erreur = %.2e  |  temps = %.6f s\n",
           n, pi, err, t);
}

/* ------------------------------------------------------------------ */
/* Étude de convergence : plusieurs valeurs de n                        */
/* ------------------------------------------------------------------ */
static void convergence_study(const char *csv_path)
{
    long ns[] = {10, 100, 1000, 10000, 100000, 1000000,
                 10000000, 100000000, 1000000000L};
    int m = (int)(sizeof(ns)/sizeof(ns[0]));

    printf("\n=== Étude de convergence (méthode des trapèzes) ===\n");
    printf("%-14s  %-20s  %-12s  %s\n",
           "n", "π approx", "erreur abs", "temps (s)");
    printf("%s\n", "----------------------------------------------------------------");

    FILE *fp = fopen(csv_path, "w");
    if (fp) fprintf(fp, "n,pi_approx,erreur_abs,erreur_rel_pct,temps_s\n");

    for (int i = 0; i < m; i++) {
        double t0 = get_time();
        double pi = trapezes_kahan(ns[i]);
        double t1 = get_time();
        double dt = t1 - t0;

        print_result(ns[i], pi, dt);

        if (fp) {
            double err_abs = fabs(pi - PI_REF);
            double err_rel = err_abs / PI_REF * 100.0;
            fprintf(fp, "%ld,%.15f,%.2e,%.2e,%.6f\n",
                    ns[i], pi, err_abs, err_rel, dt);
        }
    }
    if (fp) fclose(fp);
}

/* ------------------------------------------------------------------ */
/* Point d'entrée                                                       */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[])
{
    printf("=======================================================\n");
    printf("  Calcul de π — Méthode des trapèzes (séquentiel)\n");
    printf("  π = 4 × ∫₀¹ 1/(1+x²) dx\n");
    printf("  Référence : π = %.15f\n", PI_REF);
    printf("=======================================================\n");

    if (argc > 1) {
        /* Mode : valeur de n personnalisée */
        long n = atol(argv[1]);
        if (n < 2) { fprintf(stderr, "Erreur : n doit être >= 2\n"); return 1; }

        printf("\n  Calcul avec n = %ld trapèzes...\n", n);
        double t0 = get_time();
        double pi = trapezes_kahan(n);
        double dt = get_time() - t0;

        print_result(n, pi, dt);
        write_csv("results_seq.csv", n, pi, dt, 0);
        printf("\n  Résultats sauvegardés → results_seq.csv\n");

    } else {
        /* Mode par défaut : comparaison n = 5 000 vs n = 1 milliard */
        printf("\n--- Comparaison n = 5 000 vs n = 1 000 000 000 ---\n");

        long n1 = 5000L;
        double t0 = get_time();
        double pi1 = trapezes_kahan(n1);
        double dt1 = get_time() - t0;
        print_result(n1, pi1, dt1);
        write_csv("results_seq.csv", n1, pi1, dt1, 0);

        long n2 = 1000000000L;
        printf("  (calcul de %ld trapèzes en cours...)\n", n2);
        t0 = get_time();
        double pi2 = trapezes_kahan(n2);
        double dt2 = get_time() - t0;
        print_result(n2, pi2, dt2);
        write_csv("results_seq.csv", n2, pi2, dt2, 1);

        printf("\n  Gain de précision  : ×%.0f\n",
               fabs(pi1 - PI_REF) / fabs(pi2 - PI_REF));
        printf("  Coût en temps      : ×%.0f\n", dt2 / dt1);

        /* Étude de convergence complète */
        convergence_study("results_seq_convergence.csv");

        printf("\n  Résultats sauvegardés →\n");
        printf("    results_seq.csv\n");
        printf("    results_seq_convergence.csv\n");
    }

    printf("\n");
    return 0;
}
