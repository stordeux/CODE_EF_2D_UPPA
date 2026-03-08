import numpy as np
import sympy as sp
from mes_packages import *

# --------------------------------------------------
# Création du maillage
# --------------------------------------------------
# Domaine : carré de côté 2 contenant un trou circulaire de rayon 0.7.
# Le paramètre mesh_size contrôle la finesse du maillage.
mesh = create_mesh_circle_in_square(radius=0.7, square_size=2, mesh_size=0.05)

# Affichage du maillage pendant 2 secondes
plot_mesh(mesh, secondes=2)

# --------------------------------------------------
# Choix de l'ordre polynomial de la méthode EF continue
# --------------------------------------------------
ordre = 3

# --------------------------------------------------
# Assemblage des matrices élémentaires
# --------------------------------------------------
# Fonction coefficient constante égale à 1.
# Elle sert ici pour construire les formes bilinéaires standards.
func = lambda x, y: 1.0

# Matrice de masse :
# \int_\Omega u v dx
MASSE_CG = assemble_volume(mesh, ordre, func, "u", "v", methode="CG")

# Matrices de rigidité selon x et y :
# \int_\Omega \partial_x u \partial_x v dx
# \int_\Omega \partial_y u \partial_y v dx
RIGIDITE_CGx = assemble_volume(mesh, ordre, func, "dxu", "dxv", methode="CG")
RIGIDITE_CGy = assemble_volume(mesh, ordre, func, "dyu", "dyv", methode="CG")

# Matrice de masse sur la frontière de Fourier :
# \int_{\Gamma_F} u v ds
MASSE_BORD_CG_FOURIER = assemble_surface(
    mesh, ordre, func, "u", "v", methode="CG", domaine="FOURIER"
)

# --------------------------------------------------
# Paramètres physiques du problème d'Helmholtz
# --------------------------------------------------
# Longueur d'onde
lambda_waves = 0.1

# Nombre d'onde : k = 2 pi / lambda
kappa = 2 * np.pi / lambda_waves

# Coefficient de la condition de Fourier
alpha = 1j * kappa

# --------------------------------------------------
# Construction de la matrice globale
# --------------------------------------------------
# Nombre total de degrés de liberté en CG
taille_MAT = nombre_dof_CG(mesh, ordre)

# Nombre total de coefficients non nuls à réserver dans la matrice COO.
# On additionne les nnz des différentes contributions.
Nnz = (
    RIGIDITE_CGx.nnz
    + RIGIDITE_CGy.nnz
    + MASSE_CG.nnz
    + MASSE_BORD_CG_FOURIER.nnz
)

# Réservation mémoire pour la matrice globale
MAT_EF_CG = COOMatrix(taille_MAT, taille_MAT, Nnz)

# Assemblage de l'opérateur :
# \int_\Omega \nabla u \cdot \nabla v
# - kappa^2 \int_\Omega u v
# + alpha \int_{\Gamma_F} u v
MAT_EF_CG += RIGIDITE_CGx
MAT_EF_CG += RIGIDITE_CGy

kappa2_MASSE_CG = kappa**2 * MASSE_CG
MAT_EF_CG -= kappa2_MASSE_CG

ik_MASSE_BORD_CG_FOURIER = alpha * MASSE_BORD_CG_FOURIER
MAT_EF_CG += ik_MASSE_BORD_CG_FOURIER

# --------------------------------------------------
# Définition de la donnée incidente / source
# --------------------------------------------------
# On définit symboliquement une onde plane :
# f(x, y) = exp(i kappa x)
x, y = sp.symbols('x y')
f_source_sym = sp.exp(1j * kappa * x)

# Conversion en fonctions Python + calcul des dérivées partielles
f_source, dfx_source, dfy_source = build_f_and_grads(f_source_sym, (x, y))

# --------------------------------------------------
# Assemblage du second membre sur la frontière de Fourier
# --------------------------------------------------
# Contribution :
# \int_{\Gamma_F} f v ds
F_CG_FOURIER = assemble_rhs_surface(
    mesh, ordre,
    f_source,
    op_f="f", op_v="v",
    methode="CG",
    domaine="FOURIER"
)

# Construction du champ vectoriel gradient de f :
# grad(f) = (df/dx, df/dy)
grad_f_source = make_vector_field(dfx_source, dfy_source)

# Contribution :
# \int_{\Gamma_F} (\nabla f \cdot n) v ds
dn_F_CG_FOURIER = assemble_rhs_surface(
    mesh, ordre,
    grad_f_source,
    op_f="f.n", op_v="v",
    methode="CG",
    domaine="FOURIER"
)

# --------------------------------------------------
# Résolution du système linéaire
# --------------------------------------------------
# Second membre total :
# (\partial_n f + alpha f, v)_{\Gamma_F}
U_sol_FOURIER = MAT_EF_CG.solve(dn_F_CG_FOURIER + alpha * F_CG_FOURIER)

# --------------------------------------------------
# Visualisation de la solution numérique
# --------------------------------------------------
plot_nodal_vector_CG(
    U_sol_FOURIER,
    mesh,
    ordre,
    "Solution numérique CG",
    secondes=2
)