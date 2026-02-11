# bib-code-ef

Bibliothèque Python pour l’**analyse numérique par éléments finis en dimension 2**  
(méthodes DG et CG, quadrature, maillages, matrices de référence, visualisation).

Cette bibliothèque est utilisée dans le cadre du cours  
**Analyse Numérique – Éléments Finis**.

---

## Prérequis

- **Python ≥ 3.9** (Python 3.14 supporté)
- Windows / Linux / macOS
- VS Code recommandé

---

## Structure du projet

```

CODE_EF_2D/
├── BIB/
│ ├── pyproject.toml
│ └── mes_packages/
├── examples/
├── notebooks/
├── .venv/
└── README.md

```

---

## Création de l’environnement virtuel (recommandée)

Depuis la racine du projet :

```bash
python -m venv .venv
```

### Activation

- **Windows (PowerShell)** :

```powershell
.\.venv\Scripts\Activate.ps1
```

- **Linux / macOS** :

```bash
source .venv/bin/activate
```

Le prompt doit commencer par :

```
(.venv)
```

---

## Installation de la bibliothèque

⚠️ Cette étape doit être faite **dans le `.venv` activé**.

Depuis la racine du projet :

```bash
python -m pip install -e BIB
```

Cette commande installe automatiquement toutes les dépendances
(`numpy`, `scipy`, `matplotlib`, `sympy`, `meshio`, `pygmsh`, `gmsh`).

---

## Vérification de l’installation

```bash
python -c "import mes_packages; print('OK mes_packages')"
```

Vérification complète :

```bash
python -c "import meshio, pygmsh, mes_packages; print('OK tout est installé')"
```

---

## Utilisation dans VS Code

1. `Ctrl + Shift + P`
2. **Python: Select Interpreter**
3. Choisir :

   ```
   Python 3.x (.venv)
   ```

4. Recharger la fenêtre :

   ```
   Ctrl + Shift + P → Reload Window
   ```

---

## Utilisation des notebooks Jupyter

1. Ouvrir un notebook (`.ipynb`)
2. En haut à droite, cliquer sur le kernel
3. Sélectionner :

   ```
   Python 3.x (.venv)
   ```

4. Redémarrer le kernel si nécessaire

### Vérification dans un notebook

```python
import sys
print(sys.executable)
```

Le chemin affiché doit contenir :

```
.../CODE_EF_2D/.venv/...
```

---

## Imports usuels

```python
from mes_packages.sparse import COOMatrix
from mes_packages.methode_DG import build_nodal_vector_DG
from mes_packages.methode_CG import build_masse_CG
from mes_packages.mesh import create_mesh_circle_in_square
```

---

## Développement

- Les modifications dans `BIB/mes_packages/` sont prises en compte immédiatement
- En cas de modification de `pyproject.toml`, relancer :

```bash
python -m pip install -e BIB
```

---

## Problèmes courants

### `ModuleNotFoundError` ou `meshio is not installed`

- Vérifier que le `.venv` est activé
- Vérifier que le bon interpréteur / kernel est sélectionné
- Réinstaller la bibliothèque :

```bash
python -m pip install -e BIB
```

---

## Bonnes pratiques

- Ne pas modifier `sys.path`
- Ne pas définir `PYTHONPATH`
- Ne pas copier la bibliothèque dans `.venv`
- Utiliser un seul environnement Python par projet

---

## Auteur

- **Sébastien Tordeux**

---

## Licence

Usage pédagogique et académique.

```

```
