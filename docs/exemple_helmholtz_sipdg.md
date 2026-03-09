# Diffraction acoustique — Méthode DG (SIPG)

Cet exemple montre comment résoudre un problème de Helmholtz
avec la **méthode des éléments finis discontinus (DG)** utilisant
le schéma **SIPG (Symmetric Interior Penalty Galerkin)**.

Le domaine est un **carré contenant un obstacle circulaire**.
Une **source volumique gaussienne** excite le système et une
**condition de Fourier absorbante** est imposée sur le bord extérieur.

La solution obtenue par la méthode DG est ensuite comparée
à une solution obtenue par la **méthode CG**.

---

# Programme Python

```python
import numpy as np
import sympy as sp
from mes_packages import *

# --------------------------------------------------
# Construction du maillage
# --------------------------------------------------
mesh = create_mesh_circle_in_square(radius=0.3, square_size=3, mesh_size=0.05)

# --------------------------------------------------
# Données physiques et numériques
# --------------------------------------------------
sigma = 20
ordre = 2

lambda_waves = 1 / 2
kappa = 2 * np.pi / lambda_waves

hmin = compute_h_min(mesh)

alpha_pen = 1.j * sigma * (ordre + 1)**2 / hmin
alpha_FOURIER = 1j * kappa

# --------------------------------------------------
# Assemblage volumique
# --------------------------------------------------

func = lambda x, y: -kappa**2
MASSE_DG = assemble_volume(mesh, ordre, func, "u", "v", methode="DG")

func = lambda x, y: 1
RIGIDITE_DGx = assemble_volume(mesh, ordre, func, "dxu", "dxv", methode="DG")
RIGIDITE_DGy = assemble_volume(mesh, ordre, func, "dyu", "dyv", methode="DG")

# --------------------------------------------------
# Terme de bord de Fourier
# --------------------------------------------------

func = lambda x, y: alpha_FOURIER
MASSE_BORD_DG = assemble_surface(
    mesh, ordre, func, "u", "v", methode="DG", domaine="FOURIER"
)

# --------------------------------------------------
# Termes SIPG sur les faces
# --------------------------------------------------

A_SIPG1 = assemble_skeleton_par_face(
    mesh, ordre, -1,
    operatoru="moynablau",
    operatorv="sautDGv",
    methode="DG"
)

A_SIPG2 = assemble_skeleton_par_face(
    mesh, ordre, -1,
    operatoru="sautDGu",
    operatorv="moynablav",
    methode="DG"
)

A_saut_saut = assemble_skeleton_par_face(
    mesh, ordre, alpha_pen,
    operatoru="sautu",
    operatorv="sautv",
    methode="DG"
)

# --------------------------------------------------
# Construction de la matrice globale
# --------------------------------------------------

taille_MAT = nombre_dof_DG(mesh, ordre)

Nnz = (
    RIGIDITE_DGx.nnz
    + RIGIDITE_DGy.nnz
    + MASSE_DG.nnz
    + MASSE_BORD_DG.nnz
    + A_SIPG1.nnz
    + A_SIPG2.nnz
    + A_saut_saut.nnz
)

MAT_EF_DG = COOMatrix(taille_MAT, taille_MAT, Nnz)

MAT_EF_DG += RIGIDITE_DGx
MAT_EF_DG += RIGIDITE_DGy
MAT_EF_DG += A_SIPG1
MAT_EF_DG += A_SIPG2
MAT_EF_DG += A_saut_saut
MAT_EF_DG += MASSE_DG
MAT_EF_DG += MASSE_BORD_DG

# --------------------------------------------------
# Second membre
# --------------------------------------------------

x0 = 0
y0 = 0.6

fsource = lambda x, y: np.exp(-25 * (x - x0)**2 - 25 * (y - y0)**2)

F_DG = assemble_rhs_volume(mesh, ordre, fsource, operatorv="v", methode="DG")

# --------------------------------------------------
# Résolution
# --------------------------------------------------

U_sol_DG = MAT_EF_DG.solve(F_DG)

# --------------------------------------------------
# Visualisation
# --------------------------------------------------

Fsource = build_nodal_vector_DG(fsource, mesh, ordre)
plot_nodal_vector_DG(Fsource, mesh, ordre, "Source volumique")

plot_nodal_vector_DG(U_sol_DG, mesh, ordre, "Solution numérique DG")
```

---

# Domaine de calcul

Le domaine est un carré contenant un trou circulaire

$$
\Omega = \text{carré} \setminus \text{disque}.
$$

Le maillage est triangulaire et l’espace d’approximation utilise
des **éléments finis discontinus d’ordre 2**.

---

# Problème mathématique

On cherche une fonction complexe

$$
u : \Omega \to \mathbb{C}
$$

telle que

$$
-\Delta u - \kappa^2 u = f
\quad\text{dans }\Omega
$$

avec la condition de Fourier

$$
\frac{\partial u}{\partial n} + i\kappa u = 0
\quad\text{sur }\Gamma_F.
$$

---

# Formulation variationnelle DG

On introduit l’espace discontinu

$$
V_h = \{v_h : v_h|_K \in \mathbb{P}_p(K)\}.
$$

La formulation SIPG consiste à chercher

$$
u_h \in V_h
$$

tel que

$$
a_h(u_h,v_h) = \ell(v_h)
\quad \forall v_h \in V_h
$$

avec

$$
a_h(u,v)
=
\sum_K \int_K \nabla u \cdot \nabla v
- \kappa^2 \int_K uv
$$

et les contributions de squelette

$$
-\sum_F \int_F \{\nabla u\cdot n\}[v]
-\sum_F \int_F [u]\{\nabla v\cdot n\}
$$

et

$$
+\sum_F \int_F \alpha_{pen}[u][v].
$$

---

# Système linéaire

Après discrétisation on obtient

$$
A_{DG} U_{DG} = F.
$$

La résolution est effectuée par

```
U_sol_DG = MAT_EF_DG.solve(F_DG)
```

---

# Vérification avec la méthode CG

Pour vérifier la cohérence du résultat,
le même problème est résolu avec une **méthode CG**.

```python
MASSE_CG = assemble_volume(mesh, ordre, func, "u", "v", methode="CG")

RIGIDITE_CGx = assemble_volume(mesh, ordre, func, "dxu", "dxv", methode="CG")
RIGIDITE_CGy = assemble_volume(mesh, ordre, func, "dyu", "dyv", methode="CG")

MASSE_BORD_CG = assemble_surface(
    mesh, ordre, func, "u", "v", methode="CG", domaine="FOURIER"
)
```

La solution CG est ensuite comparée à la solution DG.

```python
U_sol_CG_in_DG = nodal_CG_to_DG(U_sol_CG, mesh, ordre)

plot_nodal_vector_DG(
    U_sol_CG_in_DG - U_sol_DG,
    mesh,
    ordre,
    "Comparaison CG et DG"
)
```

---

# Résumé

Cet exemple montre comment

- résoudre un problème de Helmholtz avec la **méthode DG SIPG**
- assembler les **termes de squelette**
- imposer une **condition de Fourier**
- comparer la solution DG à une solution CG.