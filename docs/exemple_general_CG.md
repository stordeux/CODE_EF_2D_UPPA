
# Diffraction acoustique avec coefficients variables — Méthode CG

Cet exemple montre comment résoudre un problème d’onde de type Helmholtz
avec des **coefficients variables** à l’aide de la méthode des **éléments finis continus (CG)**.

Le domaine de calcul est un **carré contenant un obstacle circulaire**.
Une **source gaussienne localisée** excite le système et une **condition de Fourier**
est imposée sur une partie du bord afin de modéliser un comportement absorbant.

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

# Définition des coefficients pour les matrices
a = lambda x, y: -kappa**2 * (1 + x**2)
b1 = lambda x, y: alpha * .5
b2 = lambda x, y: alpha * .1 * y
c = lambda x, y: 1 / (1 + x**2 + y**2)

alpha_F = lambda x, y: 1j * kappa * (1 + .1 * np.cos(x + y))

# Assemblage des matrices
MASSE_CG = assemble_volume(mesh, ordre, c, "u", "v", methode="CG")
MIXTE_CGx = assemble_volume(mesh, ordre, b1, "dxu", "v", methode="CG")
MIXTE_CGy = assemble_volume(mesh, ordre, b2, "dyu", "v", methode="CG")

RIGIDITE_CGx = assemble_volume(mesh, ordre, a, "dxu", "dxv", methode="CG")
RIGIDITE_CGy = assemble_volume(mesh, ordre, a, "dyu", "dyv", methode="CG")

MASSE_BORD_CG_Fourier = assemble_surface(
    mesh,
    ordre,
    alpha_F,
    "u",
    "v",
    methode="CG",
    domaine="FOURIER"
)

# Taille du système
taille_MAT = nombre_dof_CG(mesh, ordre)

Nnz = (
    RIGIDITE_CGx.nnz
    + RIGIDITE_CGy.nnz
    + MASSE_CG.nnz
    + MASSE_BORD_CG_Fourier.nnz
    + MIXTE_CGx.nnz
    + MIXTE_CGy.nnz
)

MAT_EF_CG = COOMatrix(taille_MAT, taille_MAT, Nnz)

# Assemblage global
MAT_EF_CG += RIGIDITE_CGx
MAT_EF_CG += RIGIDITE_CGy
MAT_EF_CG += MIXTE_CGx
MAT_EF_CG += MIXTE_CGy
MAT_EF_CG += MASSE_CG
MAT_EF_CG += MASSE_BORD_CG_Fourier

# Terme source
x0 = 0
y0 = .6

fsource = lambda x, y: np.exp(-25 * (x-x0)**2 - 25 * (y-y0)**2)

F_CG = assemble_rhs_volume(mesh, ordre, fsource, operatorv="v", methode="CG")

# Résolution
U_sol = MAT_EF_CG.solve(F_CG)

# Visualisation de la source
Fsource = build_nodal_vector_CG(fsource, mesh, ordre)
plot_nodal_vector_CG(Fsource, mesh, ordre, "La source", secondes=2)

# Visualisation de la solution
plot_nodal_vector_CG(U_sol, mesh, ordre, "Solution numérique CG")
```

---

# Domaine de calcul

Le domaine est un carré contenant un trou circulaire :

Ω = carré \ disque.

Le maillage est triangulaire et l’espace d’approximation utilise des
éléments finis continus d’ordre 2.

---

# Problème mathématique

On cherche une fonction complexe u telle que

$$
-\hbox{div}(a\nabla u) + b\cdot\nabla u +c u =f\quad\hbox{ dans }\Omega
$$
dans Ω.

Une condition de Fourier est imposée sur une partie du bord :
$$
\frac{\partial u}{\partial n} + \alpha_F u = 0
\quad
\hbox{ sur }\Gamma_F.
$$
Une condition de Neumann sur une autre partie du bord
$$
\frac{\partial u}{\partial n} + \alpha_F u = 0
\quad
\hbox{ sur }\Gamma_F.
$$

---

# Formulation variationnelle

On introduit l’espace

$$V = H^1(\Omega)$$

On cherche u ∈ V tel que
$$
a(u,v) = \ell(v) \quad\hbox{ pour tout }v \in V
$$
avec

$$
a(u,v) =
\int_{\Omega} a \frac{\partial u}{\partial x}  \frac{\partial v}{\partial x}
+ \int_{\Omega} a \frac{\partial u}{\partial y}  \frac{\partial v}{\partial y}
+ \int_{\Omega} b_1 \frac{\partial u}{\partial x} v
+ \int_{\Omega} b_2 \frac{\partial u}{\partial y} v
+ \int_{\Omega} c u v
+ \int_{\Gamma_F} \alpha_F u v
$$
et
$$
\ell(v) = \int_{\Omega} f v.
$$
---

# Système linéaire

Après discrétisation éléments finis on obtient
$$
A U = F
$$
où
$A$ est la matrice éléments finis,
$U$ le vecteur des degrés de liberté,
$F$ le second membre.

La résolution est effectuée par
`U_sol = MAT_EF_CG.solve(F_CG)`
