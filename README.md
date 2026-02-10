# CODE_EF_2D_UPPA

Code d'éléments finis 2D pour les cours à l'UPPA (Université de Pau et des Pays de l'Adour)

2D Finite Element Code for courses at UPPA (University of Pau and Pays de l'Adour)

## Description

Ce repository contient un code Python pour la méthode des éléments finis en 2D, développé pour l'enseignement. Il permet de résoudre des équations aux dérivées partielles elliptiques comme l'équation de Laplace et de Poisson.

This repository contains a Python code for the 2D finite element method, developed for teaching. It allows solving elliptic partial differential equations such as Laplace and Poisson equations.

## 📁 Structure du repository / Repository Structure

```
CODE_EF_2D_UPPA/
├── src/                    # Code source / Source code
│   ├── __init__.py        # Module principal / Main module
│   ├── mesh.py            # Gestion du maillage / Mesh management
│   ├── element.py         # Types d'éléments finis / Finite element types
│   └── solver.py          # Solveur EF / FE solver
├── examples/              # Exemples / Examples
│   ├── example1_laplace.py    # Équation de Laplace
│   └── example2_poisson.py    # Équation de Poisson
├── docs/                  # Documentation
│   ├── guide_utilisation.md   # Guide d'utilisation / User guide
│   └── exercices.md          # Exercices pour étudiants / Student exercises
├── tests/                 # Tests unitaires / Unit tests
├── README.md             # Ce fichier / This file
├── LICENSE               # Licence MIT
└── .gitignore           # Fichiers à ignorer

```

## 🚀 Installation

### Prérequis / Prerequisites

- Python 3.7 ou supérieur / Python 3.7 or higher
- pip (gestionnaire de paquets Python)

### Installation des dépendances / Installing Dependencies

```bash
pip install numpy scipy matplotlib
```

Pour la visualisation VTK (optionnel) / For VTK visualization (optional):
```bash
# Installer ParaView pour visualiser les fichiers .vtk
# Install ParaView to visualize .vtk files
# https://www.paraview.org/
```

## 📖 Utilisation / Usage

### Exemple rapide / Quick Example

```bash
# Se placer dans le répertoire / Navigate to directory
cd CODE_EF_2D_UPPA

# Exécuter un exemple / Run an example
python examples/example1_laplace.py
python examples/example2_poisson.py
```

### Utilisation dans votre code / Using in Your Code

```python
from src.mesh import Mesh
from src.solver import FESolver

# Créer un maillage 10x10 sur [0,1]x[0,1]
mesh = Mesh()
mesh.generate_rectangular_mesh(1.0, 1.0, 10, 10)

# Initialiser le solveur
solver = FESolver(mesh)
solver.assemble_system(k=1.0)

# Appliquer les conditions aux limites
# (exemple: u=0 sur tous les bords)
boundary_nodes = [...]  # Identifier les noeuds du bord
solver.apply_dirichlet_bc(boundary_nodes, [0.0]*len(boundary_nodes))

# Résoudre
solution = solver.solve()

# Exporter
solver.export_to_vtk("solution.vtk")
```

## 📚 Documentation

- **[Guide d'utilisation](docs/guide_utilisation.md)**: Documentation complète du code
- **[Exercices](docs/exercices.md)**: Exercices pour les étudiants

## 🎓 Pour les étudiants / For Students

### Commencer / Getting Started

1. Cloner ce repository / Clone this repository
2. Installer les dépendances / Install dependencies
3. Lire le [guide d'utilisation](docs/guide_utilisation.md)
4. Exécuter les exemples / Run the examples
5. Faire les [exercices](docs/exercices.md)

### Exemples fournis / Provided Examples

1. **example1_laplace.py**: Résolution de l'équation de Laplace
2. **example2_poisson.py**: Résolution de l'équation de Poisson avec terme source

## 🔬 Fonctionnalités / Features

- ✅ Maillage triangulaire structuré / Structured triangular mesh
- ✅ Éléments triangulaires P1 / P1 triangular elements
- ✅ Assemblage de matrice de rigidité / Stiffness matrix assembly
- ✅ Conditions aux limites de Dirichlet / Dirichlet boundary conditions
- ✅ Conditions aux limites de Neumann / Neumann boundary conditions
- ✅ Termes sources / Source terms
- ✅ Export VTK pour visualisation / VTK export for visualization
- ✅ Visualisation avec Matplotlib / Matplotlib visualization

## 📝 Licence / License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 👨‍🏫 Auteur / Author

stordeux - UPPA

## 🤝 Contribution

Les contributions sont les bienvenues! N'hésitez pas à ouvrir une issue ou une pull request.

Contributions are welcome! Feel free to open an issue or pull request.
