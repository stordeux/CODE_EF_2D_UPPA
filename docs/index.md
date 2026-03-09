# CODE_EF_2D_UPPA

Bienvenue dans la bibliothèque **mes_packages** de l’UPPA. Ce projet fournit des fonctions Python pour assembler et résoudre des problèmes d’éléments finis 2D (méthodes CG et DG, hyperboliques, génération de maillages, etc.). Vous trouverez ci‑dessous un aperçu de la documentation.

## Table des matières

1. [Installation](installation.md) – Mise en place de l’environnement et installation de la bibliothèque.
2. [Théorie des éléments finis continus](fem.md) – Rappel des fondements mathématiques des éléments finis continus.
3. API – Description détaillée de l'assemblage des matrices :
   - a. [assemble_volume](assemble_volume.md) – Assemblage des matrices de volume.
   - b. [assemble_surface](assemble_surface.md) – Assemblage des intégrales de bord.
   - c. [assemble_skeleton](assemble_skeleton.md) – Assemblage des termes de squelette (DG).
4. [Exemples](helmholtz_cg.md) – Notebooks et scripts illustrant la résolution de problèmes CG/DG.

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
