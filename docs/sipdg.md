# Théorie des éléments finis discontinus (SIPDG)

## Le problème aux limites

On considère le problème aux limites suivant : trouver $u$ tel que

$$
-\operatorname{div}\big(a\nabla u\big)
+ b_1 \frac{\partial u}{\partial x}
+ b_2 \frac{\partial u}{\partial y}
+ cu
=
f
\qquad \text{dans } \Omega .
$$

Le domaine $\Omega$ possède une frontière $\partial\Omega$ qui se décompose en

$$
\partial\Omega = \Gamma_D \cup \Gamma_N \cup \Gamma_F .
$$

On impose :

- **Condition de Dirichlet**

$$
u = u_D \qquad \text{sur } \Gamma_D
$$

- **Condition de Neumann**

$$
a\frac{\partial u}{\partial n} = g_N
\qquad \text{sur } \Gamma_N
$$

- **Condition de Fourier (Robin)**

$$
a\frac{\partial u}{\partial n} + \alpha_F u = g_F
\qquad \text{sur } \Gamma_F
$$

---

# La formulation DG

Dans la méthode DG, la solution approchée n’est **pas continue** entre les éléments.

On introduit un maillage

$$
\mathcal{T}_h
$$

et l’espace discret

$$
V_h =
\{ v_h \; ;\; v_h|_K \in \mathbb{P}_p(K)
\quad \forall K \in \mathcal{T}_h \}.
$$

Les fonctions de $V_h$ sont **polynomiales dans chaque élément**
mais **peuvent être discontinues entre éléments**.

---

# Sauts et moyennes

Soit $F$ une face commune à deux éléments $K^+$ et $K^-$.

On définit :

### saut

$$
[u] = u^+ n^+ + u^- n^-
$$

### moyenne

$$
\{q\} =
\frac{1}{2}(q^+ + q^-)
$$

où $n^+$ et $n^-$ sont les normales sortantes.

Ces quantités apparaissent dans les **termes de flux numériques**.

---

# Formulation variationnelle SIPDG

La formulation SIPDG consiste à chercher

$$
u_h \in V_h
$$

tel que

$$
a_h(u_h,v_h) = \ell_h(v_h)
\qquad
\forall v_h \in V_h .
$$

avec

## Terme volumique

$$
\sum_{K\in\mathcal{T}_h}
\int_K
\left(
a\nabla u_h\cdot\nabla v_h
+
(b\cdot\nabla u_h)v_h
+
c u_h v_h
\right)
dx
$$

## Flux symétriques

$$
-\sum_{F}
\int_F
\{a\nabla u_h\cdot n\}[v_h]\,ds
$$

$$
-\sum_{F}
\int_F
[u_h]\{a\nabla v_h\cdot n\}\,ds
$$

## Terme de pénalisation

$$
\sum_F
\int_F
\alpha_{pen}\,[u_h][v_h]\,ds
$$

où

$$
\alpha_{pen}
=
\sigma
\frac{(p+1)^2}{h_F}.
$$

Ce terme assure la **stabilité et la coercivité** de la méthode.

---

# Conditions aux limites

Les conditions aux limites sont incorporées dans les flux.

### Dirichlet

Traitée faiblement par le flux numérique.

### Neumann

Contribue directement au second membre.

### Fourier

Produit un terme de bord

$$
\int_{\Gamma_F} \alpha_F u_h v_h ds .
$$

---

# Formulation discrète

Le problème DG s’écrit :

Trouver $u_h \in V_h$ tel que

$$
a_h(u_h,v_h)=\ell_h(v_h)
\qquad
\forall v_h\in V_h .
$$

Après discrétisation on obtient le système linéaire

$$
A U = F
$$

où

- $A$ est la matrice DG
- $U$ le vecteur des degrés de liberté
- $F$ le second membre.

---

# Exemple de résolution

Des exemples d’implémentation DG sont disponibles dans le projet :

- Exemple Helmholtz DG (SIPG)

    - `examples/exemple_helmholtz_DG.py`
    - [`docs/exemple_helmholtz_DG.md`](exemple_helmholtz_DG.md)

- Exemple comparaison CG / DG

    - `examples/exemple_comparaison_CG_DG.py`
    - [`docs/exemple_comparaison_CG_DG.md`](exemple_comparaison_CG_DG.md)