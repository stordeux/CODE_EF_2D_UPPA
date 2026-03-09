# Diffraction aéroacoustique — Méthode CG

Cet exemple montre comment résoudre un problème de propagation d’onde
de type **Helmholtz convecté simplifié** à l’aide de la méthode des
**éléments finis continus (Continuous Galerkin — CG)**.

Le domaine de calcul est un **carré contenant un obstacle circulaire**.
Une **source volumique gaussienne** excite le système et une
**condition de Fourier absorbante** est imposée sur le bord extérieur.

---

# Programme Python

```python
import numpy as np
import sympy as sp
from mes_packages import *

# Données physiques
lambda_waves = 1/2
kappa = 2 * np.pi / lambda_waves
alpha = 1j * kappa

mesh = create_mesh_circle_in_square(radius=0.3, square_size=3, mesh_size=0.05)
ordre = 2

func = lambda x, y: -kappa**2
MASSE_CG = assemble_volume(mesh, ordre, func, "u", "v", methode="CG")

func = lambda x, y: alpha * .5
MIXTE_CG = assemble_volume(mesh, ordre, func, "dxu", "v", methode="CG")

func = lambda x, y: 1
RIGIDITE_CGx = assemble_volume(mesh, ordre, func, "dxu", "dxv", methode="CG")
RIGIDITE_CGy = assemble_volume(mesh, ordre, func, "dyu", "dyv", methode="CG")

func = lambda x, y: alpha
MASSE_BORD_CG = assemble_surface(mesh, ordre, func, "u", "v", methode="CG", domaine="FOURIER")

taille_MAT = nombre_dof_CG(mesh, ordre)

Nnz = (
    RIGIDITE_CGx.nnz
    + RIGIDITE_CGy.nnz
    + MASSE_CG.nnz
    + MASSE_BORD_CG.nnz
    + MIXTE_CG.nnz
)

MAT_EF_CG = COOMatrix(taille_MAT, taille_MAT, Nnz)

MAT_EF_CG += RIGIDITE_CGx
MAT_EF_CG += RIGIDITE_CGy
MAT_EF_CG += MIXTE_CG
MAT_EF_CG += MASSE_CG
MAT_EF_CG += MASSE_BORD_CG

x0 = 0
y0 = .6

fsource = lambda x, y: np.exp(-25 * (x-x0)**2 - 25 * (y-y0)**2)

F_CG = assemble_rhs_volume(mesh, ordre, fsource, operatorv="v", methode="CG")

U_sol = MAT_EF_CG.solve(F_CG)

Fsource = build_nodal_vector_CG(fsource, mesh, ordre)
plot_nodal_vector_CG(Fsource, mesh, ordre, "La source", secondes=2)

plot_nodal_vector_CG(U_sol, mesh, ordre, "Solution numérique CG")
```

---

# Domaine de calcul

Le domaine est un carré contenant un obstacle circulaire

$$
\Omega = \text{carré} \setminus \text{disque}.
$$

Le maillage est triangulaire et l’espace d’approximation utilise
des **éléments finis continus d’ordre 2**.

---

# Problème mathématique

On cherche une fonction complexe

$$
u : \Omega \rightarrow \mathbb{C}
$$

telle que

$$
-\Delta u + \alpha \frac{\partial u}{\partial x}
- \kappa^2 u = f
\quad \text{dans } \Omega.
$$

Une condition de Fourier est imposée sur le bord extérieur

$$
\frac{\partial u}{\partial n} + \alpha u = 0
\quad \text{sur } \Gamma_F.
$$

---

# Formulation variationnelle

On introduit l’espace

$$
V = H^1(\Omega).
$$

On cherche

$$
u \in V
$$

tel que

$$
a(u,v) = \ell(v)
\quad \forall v \in V
$$

avec

$$
a(u,v) =
\int_\Omega \nabla u \cdot \nabla v
+ \int_\Omega \alpha \frac{\partial u}{\partial x} v
- \int_\Omega \kappa^2 u v
+ \int_{\Gamma_F} \alpha u v
$$

et

$$
\ell(v) = \int_\Omega f v.
$$

---

# Système linéaire

Après discrétisation éléments finis on obtient

$$
A U = F
$$

où

- $A$ est la matrice éléments finis,
- $U$ le vecteur des degrés de liberté,
- $F$ le second membre.

La résolution est effectuée par

```
U_sol = MAT_EF_CG.solve(F_CG)
```

---

# Visualisation

La source volumique est représentée par

```
plot_nodal_vector_CG(Fsource, mesh, ordre, "La source")
```

La solution numérique est visualisée par

```
plot_nodal_vector_CG(U_sol, mesh, ordre, "Solution numérique CG")
```

---

# Résumé

Cet exemple montre comment :

- résoudre un problème de propagation d’onde avec la méthode **CG**
- assembler les matrices **masse, rigidité et convection**
- imposer une **condition de Fourier absorbante**
- visualiser la solution numérique.