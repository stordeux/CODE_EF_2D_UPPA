# Théorie des éléments finis continus

## Le problème aux limites

On considère le problème aux limites suivant : trouver $u\in H^1(\Omega)$ tel que

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

# La formulation variationnelle

On introduit l’espace

$$
V = \{v \in H^1(\Omega) \; ;\; v = 0 \text{ sur } \Gamma_D\}.
$$

On écrit la solution sous la forme

$$
u = u_D + w
$$

avec $w \in V$.

On multiplie l’équation différentielle par une fonction test $v \in V$ et on intègre sur $\Omega$ :

$$
\int_\Omega
\left(
-\operatorname{div}(a\nabla u)
+ b_1 \frac{\partial u}{\partial x}
+ b_2 \frac{\partial u}{\partial y}
+ cu
\right)
v
\,dx
=
\int_\Omega fv\,dx .
$$

On applique ensuite une intégration par parties au terme diffusif :

$$
\int_\Omega -\operatorname{div}(a\nabla u)\,v\,dx
=
\int_\Omega a\nabla u\cdot\nabla v\,dx
-
\int_{\partial\Omega} a\frac{\partial u}{\partial n} v\,ds .
$$

En utilisant la décomposition du bord

$$
\partial\Omega=\Gamma_D\cup\Gamma_N\cup\Gamma_F
$$

et les conditions aux limites, on obtient

$$
\int_\Omega a\nabla u\cdot\nabla v\,dx
+
\int_\Omega (b\cdot\nabla u)\,v\,dx
+
\int_\Omega cuv\,dx
+
\int_{\Gamma_F}\alpha_F uv\,ds
=
\int_\Omega fv\,dx
+
\int_{\Gamma_N} g_N v\,ds
+
\int_{\Gamma_F} g_F v\,ds .
$$

---

# Formulation faible

La **formulation variationnelle** du problème s’écrit :

Trouver $u\in H^1(\Omega)$ tel que $u=u_D$ sur $\Gamma_D$ et

$$
a(u,v)=\ell(v)
\qquad \forall v\in V
$$

avec

$$
a(u,v)=
\int_\Omega a\nabla u\cdot\nabla v\,dx
+
\int_\Omega (b\cdot\nabla u)v\,dx
+
\int_\Omega cuv\,dx
+
\int_{\Gamma_F}\alpha_F uv\,ds
$$

et

$$
\ell(v)=
\int_\Omega fv\,dx
+
\int_{\Gamma_N} g_N v\,ds
+
\int_{\Gamma_F} g_F v\,ds .
$$

Cette formulation est appelée **formulation faible** du problème.  
Elle constitue le point de départ de la **méthode des éléments finis**, qui consiste à rechercher une approximation de la solution dans un sous-espace de dimension finie de $V$.

---

# Exemple de résolution

Des exemples d’implémentation sont disponibles dans le projet :

- Exemple général avec coefficients variables

    - `examples/exemple_CG_general.py`
    - [`docs/exemple_general_CG.md`](exemple_general_CG.md)

- Exemple de résolution de problème de Helmoltz avec terme source

    - `examples/exemple_helmholtz_CG.py`
    - [`docs/exemple_helmholtz_CG.md`](exemple_helmholtz_CG.md)

- Exemple de diffraction aéroacoustique

    - `examples/exemple_aeroac_CG.py`
    - [`docs/exemple_aeroac_CG.md`](exemple_aeroac_CG.md)
