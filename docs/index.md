# CODE_EF_2D_UPPA

Bienvenue dans la bibliothèque **mes_packages** de l’UPPA. Ce projet fournit des fonctions Python pour assembler et résoudre des problèmes d’éléments finis 2D (méthodes CG et DG, hyperboliques, génération de maillages, etc.). Vous trouverez ci‑dessous un aperçu de la documentation.

## Table des matières

1. [Installation](installation.md) – Mise en place de l’environnement et installation de la bibliothèque.
2. Théorie des éléments finis
    - a. [Eléments finis continus](fem.md) Rappel des fondements mathématiques des éléments finis continus.
    - b. [SIPDG](sipdg.md) Rappel des fondements mathématiques des éléments finis continus.

3. API – Description détaillée de l'assemblage des matrices :

     - a. [assemble_volume](assemble_volume.md) – Assemblage des matrices de volume.

     - b. [assemble_surface](assemble_surface.md) – Assemblage des intégrales de bord.

     - c. Assemblage des termes du squelette par élément (DG).
     [assemble_skeleton_par_element](assemble_skeleton_element.md)  
     [assemble_skeleton_face](assemble_skeleton_face.md) 

4. Notebooks et scripts illustrant la résolution de problèmes CG.

    - a  [Un problème de Helmholtz](exemple_helmholtz_CG.md) Résolution d'un problème de Helmholtz avec source volumique

    - b [Un problème aéroacoustique](exemple_aeroac_CG.md) Résolution d'un problème aéroacoustique

    - c

5. Notebooks et scripts illustrant la résolution de problèmes SIPDG.

    - a - [Un problème de Helmholtz](exemple_helmholtz_sipdg.md) Résolution d'un problème de Helmholtz avec source volumique

## Démarrage rapide

Pour cloner et installer la bibliothèque :

```bash
git clone git@…:stordeux/ef_ed_upp.git
cd CODE_EF_2D_UPPA
python -m venv .venv
source .venv/bin/activate  # sous Windows : .venv\Scripts\activate
pip install -e .
mkdocs serve
```
