# Exemple : résolution de l'équation de Helmholtz par éléments finis CG

Cet exemple illustre l'utilisation de la bibliothèque `mes_packages`
pour résoudre numériquement une équation de Helmholtz avec une condition
de Fourier (Robin) sur le bord du domaine.

---

# 1. Domaine et maillage

On considère un domaine constitué d'un carré contenant un trou circulaire :

$$
\Omega = [-4,4]^2 \setminus B(0,1)
$$

Le maillage est généré par

```python
mesh = create_mesh_circle_in_square(radius=1, square_size=8, mesh_size=0.25)
plot_mesh(mesh)
```

---

# 2. Problème mathématique

On cherche

$$
u : \Omega \rightarrow \mathbb{C}
$$

solution de

$$
-\Delta u - \kappa^2 u = f
\qquad \text{dans } \Omega
$$

avec la condition de Fourier

$$
\partial_n u + i\kappa u = 0
\qquad \text{sur } \Gamma
$$

où

- $\kappa = \frac{2\pi}{\lambda}$ est le nombre d'onde
- $f$ est une source volumique.

---

# 3. Formulation variationnelle

On cherche

$$
u \in H^1(\Omega)
$$

tel que

$$
a(u,v) = \ell(v)
\qquad \forall v \in H^1(\Omega)
$$

avec

$$
a(u,v)
=
\int_\Omega \nabla u \cdot \nabla v
-
\kappa^2 \int_\Omega uv
+
i\kappa \int_\Gamma uv
$$

et

$$
\ell(v)
=
\int_\Omega f v.
$$

---

# 4. Discrétisation éléments finis

On introduit l'espace EF

$$
V_h \subset H^1(\Omega)
$$

constitué de polynômes de degré \(p\) sur chaque triangle.

Dans le code

```python
ordre = 3
```

---

# 5. Matrices éléments finis

## Matrice de masse

$$
M_{ij}
=
\int_\Omega \phi_i \phi_j
$$

```python
MASSE_CG =
assemble_volume(mesh, ordre, func, "u", "v", methode="CG")
```

---

## Matrices de rigidité

$$
K_x(i,j)
=
\int_\Omega
\partial_x \phi_i
\partial_x \phi_j
$$

$$
K_y(i,j)
=
\int_\Omega
\partial_y \phi_i
\partial_y \phi_j
$$

```python
RIGIDITE_CGx =
assemble_volume(mesh, ordre, func, "dxu", "dxv", methode="CG")

RIGIDITE_CGy =
assemble_volume(mesh, ordre, func, "dyu", "dyv", methode="CG")
```

La matrice de Laplace est

$$
K = K_x + K_y
$$

---

## Matrice de bord (condition de Fourier)

$$
B_{ij}
=
\int_{\Gamma} \phi_i \phi_j
$$

```python
MASSE_BORD_CG_FOURIER =
assemble_surface(mesh, ordre, func, "u", "v", methode="CG", domaine="FOURIER")
```

---

# 6. Matrice globale

La matrice discrète du problème est

$$
A =
K_x + K_y
-
\kappa^2 M
+
i\kappa B
$$

dans le code :

```python
MAT_EF_CG += RIGIDITE_CGx
MAT_EF_CG += RIGIDITE_CGy
MAT_EF_CG -= kappa**2 * MASSE_CG
MAT_EF_CG += alpha * MASSE_BORD_CG_FOURIER
```

---

# 7. Second membre

La source est une gaussienne

$$
f(x,y) =
\exp\left(
-10(x-x_0)^2
-10(y-y_0)^2
\right)
$$

```python
f_source =
lambda x,y: np.exp(-10*(x-x_0)**2 -10*(y-y_0)**2)
```

Le second membre discret est

$$
F_i =
\int_\Omega f \phi_i
$$

```python
F_volume =
assemble_rhs_volume(mesh, ordre, f_source, "v", methode="CG")
```

---

# 8. Système linéaire

On résout

$$
A U = F
$$

```python
U_sol = MAT_EF_CG.solve(F_volume)
```

---

# 9. Visualisation

```python
plot_nodal_vector_CG(U_sol, mesh, ordre)
```

La solution est visualisée sur le maillage avec une interpolation
linéaire sur les sous-triangles.

---

# 10. Résumé

Le script réalise les étapes classiques d'une simulation EF :

1. génération du maillage  
2. assemblage des matrices locales  
3. construction de la matrice globale  
4. assemblage du second membre  
5. résolution du système linéaire  
6. visualisation de la solution