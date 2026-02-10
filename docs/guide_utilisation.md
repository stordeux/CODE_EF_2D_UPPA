# Guide d'utilisation du code d'éléments finis 2D
# 2D Finite Element Code User Guide

## Introduction

Ce code implémente la méthode des éléments finis en 2D pour résoudre des équations aux dérivées partielles elliptiques, notamment l'équation de Laplace et de Poisson.

This code implements the finite element method in 2D to solve elliptic partial differential equations, particularly the Laplace and Poisson equations.

## Structure du code / Code Structure

### Modules principaux / Main Modules

#### `mesh.py` - Gestion du maillage / Mesh Management
- **Classe `Node`**: Représente un noeud du maillage avec ses coordonnées (x, y)
- **Classe `Element`**: Représente un élément fini générique
- **Classe `Mesh`**: Gère l'ensemble du maillage
  - `add_node(x, y)`: Ajoute un noeud
  - `add_element(node_indices)`: Ajoute un élément
  - `generate_rectangular_mesh(lx, ly, nx, ny)`: Génère un maillage rectangulaire

#### `element.py` - Types d'éléments finis / Finite Element Types
- **Classe `TriangleElement`**: Élément triangulaire à 3 noeuds (P1)
  - `compute_area()`: Calcule l'aire de l'élément
  - `shape_functions(xi, eta)`: Fonctions de forme
  - `compute_B_matrix()`: Matrice des dérivées des fonctions de forme
  - `compute_stiffness_matrix(k)`: Matrice de rigidité élémentaire

- **Classe `QuadElement`**: Élément quadrilatéral à 4 noeuds (Q1)
  - Fonctions de forme bilinéaires

#### `solver.py` - Résolution du système / System Solution
- **Classe `FESolver`**: Solveur principal
  - `assemble_system(k)`: Assemble la matrice de rigidité globale
  - `apply_dirichlet_bc(nodes, values)`: Applique les conditions de Dirichlet
  - `apply_neumann_bc(nodes, flux)`: Applique les conditions de Neumann
  - `apply_source_term(f)`: Applique un terme source
  - `solve()`: Résout le système linéaire
  - `export_to_vtk(filename)`: Export au format VTK

## Utilisation / Usage

### Exemple simple / Simple Example

```python
from src.mesh import Mesh
from src.solver import FESolver

# Créer un maillage / Create a mesh
mesh = Mesh()
mesh.generate_rectangular_mesh(lx=1.0, ly=1.0, nx=10, ny=10)

# Créer le solveur / Create solver
solver = FESolver(mesh)
solver.assemble_system(k=1.0)

# Appliquer les conditions aux limites / Apply boundary conditions
# ... (voir exemples / see examples)

# Résoudre / Solve
solution = solver.solve()

# Exporter / Export
solver.export_to_vtk("solution.vtk")
```

## Équations résolues / Equations Solved

### Équation de Laplace
```
-Δu = 0  dans Ω
```

### Équation de Poisson
```
-Δu = f  dans Ω
```

où `Δ = ∂²/∂x² + ∂²/∂y²` est l'opérateur Laplacien.

### Conditions aux limites / Boundary Conditions

1. **Dirichlet**: `u = g` sur le bord (valeur imposée)
2. **Neumann**: `∂u/∂n = h` sur le bord (flux imposé)

## Formulation faible / Weak Formulation

Trouver u ∈ V tel que:
```
∫_Ω ∇u · ∇v dx = ∫_Ω f v dx + ∫_∂Ω h v ds
```
pour tout v ∈ V (fonction test).

## Discrétisation / Discretization

- **Éléments**: Triangles P1 (linéaires)
- **Fonctions de forme**: Linéaires par morceaux
- **Intégration**: Points de Gauss ou approximation au centre

## Références / References

1. Zienkiewicz, O.C., Taylor, R.L. - The Finite Element Method
2. Hughes, T.J.R. - The Finite Element Method: Linear Static and Dynamic
3. Dhatt, G., Touzot, G. - Une présentation de la méthode des éléments finis
