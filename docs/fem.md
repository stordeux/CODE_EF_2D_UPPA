# Théorie des éléments finis continus

## Le problème aux limites
On considère le problème aux limites : trouver $u\in H^1(\Omega)$
$$
-div\Big(a\nabla u\Big) + b_1 \frac{\partial u}{\partial x}+ b_2 \frac{\partial u}{\partial x} + c u = f\quad\hbox{ dans }\Omega
$$
avec $\Omega$ de frontière $\partial\Omega$ qui se décompose en 
- une partie sur laquelle est posée une condition de Dirichlet
$$
    u=u_{D}\hbox{ dans }\Gamma_{D}
$$
- une partie sur laquelle est posée une condition de Neumann
$$
    a\frac{\partial u}{\partial n}=g_{N}\hbox{ dans }\Gamma_{N}
$$
- une partie sur laquelle est posée une condition de Fourier
$$
    a\frac{\partial u}{\partial n}+\alpha_F u=g_{F}\hbox{ dans }\Gamma_{F}
$$
## La formulation variationnelle

On introduit l’espace
$$
V=\{v\in H^1(\Omega)\; ;\; v=0 \text{ sur }\Gamma_D\}.
$$

On cherche une solution sous la forme
$$
u = u_D + w
$$
où $w\in V$.

On multiplie l’équation différentielle par une fonction test $v\in V$ et on intègre sur $\Omega$ :
$$
\int_\Omega \Big(-\operatorname{div}(a\nabla u) 
+ b_1 \frac{\partial u}{\partial x}
+ b_2 \frac{\partial u}{\partial y}
+ cu\Big)v \,dx
=
\int_\Omega f v \,dx.
$$

On applique ensuite une intégration par parties au terme diffusif :
$$
\int_\Omega -\operatorname{div}(a\nabla u)\,v\,dx
=
\int_\Omega a\nabla u\cdot\nabla v\,dx
-
\int_{\partial\Omega} a\frac{\partial u}{\partial n} v\,ds.
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

La **formulation variationnelle** du problème s’écrit donc :

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
\int_{\Gamma_F} g_F v\,ds.
$$

Cette formulation est appelée **formulation faible** du problème. Elle constitue le point de départ de la méthode des éléments finis, qui consiste à rechercher une approximation de la solution dans un sous-espace de dimension finie de $V$.
