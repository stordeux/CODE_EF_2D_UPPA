# Exercices - Méthode des Éléments Finis 2D
# Exercises - 2D Finite Element Method

## Exercice 1: Découverte du code / Exercise 1: Code Discovery

**Objectif / Objective**: Se familiariser avec la structure du code

1. Explorez les fichiers dans le répertoire `src/`
2. Exécutez l'exemple 1 (équation de Laplace)
3. Visualisez les résultats avec ParaView ou en ouvrant l'image PNG

**Questions**:
- Combien de noeuds et d'éléments dans le maillage 10x10?
- Quelle est la valeur de la solution au centre du domaine?

## Exercice 2: Équation de Laplace / Exercise 2: Laplace Equation

**Problème**: Résoudre `-Δu = 0` dans `Ω = [0,1] x [0,1]`

**Conditions aux limites**:
- `u = 0` sur x = 0, x = 1, y = 0
- `u = sin(πx)` sur y = 1

**À faire / To do**:
1. Modifier `example1_laplace.py` pour ces nouvelles conditions
2. Créer un nouveau fichier `exercice2.py`
3. Comparer avec différentes résolutions de maillage (5x5, 10x10, 20x20)

## Exercice 3: Terme source constant / Exercise 3: Constant Source Term

**Problème**: Résoudre `-Δu = 1` dans `Ω = [0,1] x [0,1]`

**Conditions aux limites**: `u = 0` sur tout le bord

**À faire / To do**:
1. Utiliser `solver.apply_source_term()` avec f(x,y) = 1
2. Visualiser la solution
3. Où se trouve le maximum de u?

**Solution attendue / Expected solution**: Maximum au centre

## Exercice 4: Problème de conduction thermique / Exercise 4: Heat Conduction Problem

**Problème physique**: Plaque carrée avec:
- Bord gauche: T = 100°C (chaud)
- Bord droit: T = 0°C (froid)
- Bords haut et bas: isolés (∂T/∂n = 0)

**À faire / To do**:
1. Créer un nouveau fichier `exercice4_conduction.py`
2. Résoudre avec k = 1 (conductivité thermique)
3. Tracer les isothermes (lignes de température constante)
4. Calculer le flux de chaleur: q = -k ∇T

## Exercice 5: Convergence / Exercise 5: Convergence

**Objectif**: Étudier la convergence de la méthode

**À faire / To do**:
1. Utiliser le problème de l'exemple 2 (solution analytique connue)
2. Résoudre pour différentes résolutions: nx = ny = 5, 10, 20, 40, 80
3. Calculer l'erreur L2 pour chaque cas
4. Tracer log(erreur) vs log(h) où h = 1/nx
5. Quelle est l'ordre de convergence?

**Formule de l'erreur L2**:
```python
error_L2 = np.sqrt(np.sum((u_numerical - u_analytical)**2) / len(u_numerical))
```

## Exercice 6: Maillage non-uniforme / Exercise 6: Non-uniform Mesh

**Objectif**: Créer un maillage raffiné localement

**À faire / To do**:
1. Créer un maillage plus fin près d'un point singulier
2. Modifier la fonction `generate_rectangular_mesh` ou créer un nouveau maillage manuellement
3. Comparer les résultats avec un maillage uniforme

## Exercice 7: Problème en L / Exercise 7: L-shaped Domain

**Domaine**: Domaine en forme de L

```
  (0,1) ---- (1,1)
    |          |
    |    (0.5,0.5) ---- (1,0.5)
    |          |
  (0,0) ----(0.5,0)
```

**À faire / To do**:
1. Créer le maillage pour ce domaine
2. Résoudre `-Δu = 1` avec `u = 0` sur le bord
3. Observer la solution près du coin rentrant

## Projet Final: Problème au choix / Final Project: Custom Problem

Choisir un des problèmes suivants:

1. **Diffusion-réaction**: `-Δu + cu = f` avec c > 0
2. **Problème non-homogène**: Conditions de Neumann variables
3. **Optimisation de forme**: Trouver la forme optimale pour minimiser un critère

## Ressources / Resources

- Documentation dans `docs/guide_utilisation.md`
- Exemples dans `examples/`
- Code source dans `src/`

## Aide / Help

Pour toute question, consultez:
1. La documentation du code
2. Les exemples fournis
3. Les références bibliographiques

Bon travail! / Good work!
