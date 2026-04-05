# Algorithmique et Programmation Parallèle

![C](https://img.shields.io/badge/C-C99-lightgrey)
![MPI](https://img.shields.io/badge/MPI-OpenMPI_4.0+-orange)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

**M1 CHPS — Université de Perpignan Via Domitia (UPVD)**
Cours : Algorithmes et Programmation Parallèle

**Auteurs :** YAYA TOURE · Mamadou G Diallo
**Encadrant :** [Benjamin Antunes](https://scholar.google.com/citations?user=o5rgTqEAAAAJ&hl=en) — Assistant Professor, UPVD

---

## Demo — Simulation SEIR Multi-Agents MPI

<p align="center">
  <img src="seir_demo.gif" width="100%" alt="Simulation SEIR MPI — déplacements stochastiques et comparaison Séquentiel vs MPI"/>
</p>

<p align="center">
  <em>730 pas de temps · grille 300×300 torique · 20 000 agents</em><br>
  <strong>Susceptible</strong> (bleu) · <strong>Exposé</strong> (jaune) · <strong>Infecté</strong> (rouge) · <strong>Remis</strong> (vert)
</p>

> Grille 300×300 · 20 000 agents · comparaison Séquentiel vs MPI
> États : **Susceptible** (bleu) · **Exposé** (jaune) · **Infecté** (rouge) · **Remis** (vert)

---

| Partie | Sujet | Dossier |
|---|---|---|
| 1 | Calcul de Pi — méthode déterministe | [partie1_pi_deterministe](./partie1_pi_deterministe) |
| 2 | Calcul de Pi — méthode stochastique (Monte Carlo MPI) | [partie2_pi_stochastique](./partie2_pi_stochastique) |
| 3 | Simulation SEIR multi-agents MPI | [partie3_seir_mpi](./partie3_seir_mpi) |

Machine de référence : Intel i9-11950H · 8 coeurs physiques · 128 Go RAM