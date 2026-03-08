import numpy as np
import sympy as sp
from mes_packages import *

# --------------------------------------------------
# Construction du maillage
# --------------------------------------------------
# Domaine : carré de côté 3 contenant un trou circulaire de rayon 0.3.
# Le paramètre mesh_size contrôle la finesse du maillage.
mesh = create_mesh_circle_in_square(radius=0.3, square_size=3, mesh_size=0.05)

# --------------------------------------------------
# Données physiques et numériques
# --------------------------------------------------
# Paramètre de pénalisation SIPG
sigma = 20

# Ordre polynomial local de la méthode DG
ordre = 2

# Longueur d'onde
lambda_waves = 1 / 2

# Nombre d'onde
kappa = 2 * np.pi / lambda_waves

# Taille minimale du maillage
hmin = compute_h_min(mesh)

# Paramètre de pénalisation sur les sauts
# Utilisé dans le terme SIPG de stabilisation
alpha_pen = 1.j * sigma * (ordre + 1)**2 / hmin

# Coefficient de la condition de Fourier sur le bord extérieur
alpha_FOURIER = 1j * kappa

# Taille minimale du maillage (recalculée ici, mais ce n'est pas indispensable)
hmin = compute_h_min(mesh)

# --------------------------------------------------
# Assemblage des contributions volumiques
# --------------------------------------------------
# Terme de masse avec coefficient -kappa^2 :
# \int_\Omega (-kappa^2) u v dx
func = lambda x, y: -kappa**2
MASSE_DG = assemble_volume(mesh, ordre, func, "u", "v", methode="DG")

# Terme de rigidité :
# \int_\Omega \partial_x u \partial_x v dx
# \int_\Omega \partial_y u \partial_y v dx
func = lambda x, y: 1
RIGIDITE_DGx = assemble_volume(mesh, ordre, func, "dxu", "dxv", methode="DG")
RIGIDITE_DGy = assemble_volume(mesh, ordre, func, "dyu", "dyv", methode="DG")

# --------------------------------------------------
# Assemblage du terme de bord de Fourier
# --------------------------------------------------
# Contribution :
# \int_{\Gamma_F} alpha_FOURIER u v ds
func = lambda x, y: alpha_FOURIER
MASSE_BORD_DG = assemble_surface(
    mesh, ordre, func, "u", "v", methode="DG", domaine="FOURIER"
)

# --------------------------------------------------
# Nombre total de degrés de liberté DG
# --------------------------------------------------
taille_MAT = nombre_dof_DG(mesh, ordre)

# --------------------------------------------------
# Assemblage des termes de squelette SIPG
# --------------------------------------------------
# Premier terme symétrique :
# - \int_{\mathcal{F}_h} {grad u · n} [v]
A_SIPG1 = assemble_skeleton_par_face(
    mesh, ordre, -1,
    operatoru="moynablau",
    operatorv="sautDGv",
    methode="DG"
)

# Second terme symétrique :
# - \int_{\mathcal{F}_h} [u] {grad v · n}
A_SIPG2 = assemble_skeleton_par_face(
    mesh, ordre, -1,
    operatoru="sautDGu",
    operatorv="moynablav",
    methode="DG"
)

# Terme de pénalisation sur les sauts :
# \int_{\mathcal{F}_h} alpha_pen [u][v]
A_saut_saut = assemble_skeleton_par_face(
    mesh, ordre, alpha_pen,
    operatoru="sautu",
    operatorv="sautv",
    methode="DG"
)

# --------------------------------------------------
# Réservation mémoire pour la matrice globale
# --------------------------------------------------
# Nombre total de coefficients non nuls
Nnz = (
    RIGIDITE_DGx.nnz
    + RIGIDITE_DGy.nnz
    + MASSE_DG.nnz
    + MASSE_BORD_DG.nnz
    + A_SIPG1.nnz
    + A_SIPG2.nnz
    + A_saut_saut.nnz
)

# Création de la matrice globale DG au format COO
MAT_EF_DG = COOMatrix(taille_MAT, taille_MAT, Nnz)

# --------------------------------------------------
# Assemblage de l'opérateur global
# --------------------------------------------------
MAT_EF_DG += RIGIDITE_DGx
MAT_EF_DG += RIGIDITE_DGy
MAT_EF_DG += A_SIPG1
MAT_EF_DG += A_SIPG2
MAT_EF_DG += A_saut_saut
MAT_EF_DG += MASSE_DG
MAT_EF_DG += MASSE_BORD_DG

# --------------------------------------------------
# Définition du second membre
# --------------------------------------------------
# Source gaussienne centrée en (x0, y0)
x0 = 0
y0 = 0.6
fsource = lambda x, y: np.exp(-25 * (x - x0)**2 - 25 * (y - y0)**2)

# Assemblage du second membre volumique :
# \int_\Omega f v dx
F_DG = assemble_rhs_volume(mesh, ordre, fsource, operatorv="v", methode="DG")

# --------------------------------------------------
# Résolution du système linéaire
# --------------------------------------------------
# Résolution du problème discret :
# MAT_EF_DG * U_sol = F_DG
U_sol_DG = MAT_EF_DG.solve(F_DG)

# --------------------------------------------------
# Visualisation de la source
# --------------------------------------------------
# Représentation nodale DG de la source pour visualiser son profil
Fsource = build_nodal_vector_DG(fsource, mesh, ordre)
plot_nodal_vector_DG(Fsource, mesh, ordre, "Source volumique")

# --------------------------------------------------
# Visualisation de la solution numérique
# --------------------------------------------------
plot_nodal_vector_DG(U_sol_DG, mesh, ordre, "Solution numérique DG")



############################################
# Verification par une methode CG
############################################


# --------------------------------------------------
# Données physiques et numériques
# --------------------------------------------------
ordre = 2
lambda_waves = 1 / 2
kappa = 2 * np.pi / lambda_waves
alpha_FOURIER = 1j * kappa

# --------------------------------------------------
# Assemblage des contributions volumiques
# --------------------------------------------------
# Terme de masse : \int_\Omega (-kappa^2) u v dx
func = lambda x, y: -kappa**2
MASSE_CG = assemble_volume(mesh, ordre, func, "u", "v", methode="CG")

# Termes de rigidité :
# \int_\Omega \partial_x u \partial_x v dx
# \int_\Omega \partial_y u \partial_y v dx
func = lambda x, y: 1.0
RIGIDITE_CGx = assemble_volume(mesh, ordre, func, "dxu", "dxv", methode="CG")
RIGIDITE_CGy = assemble_volume(mesh, ordre, func, "dyu", "dyv", methode="CG")

# --------------------------------------------------
# Assemblage du terme de bord de Fourier
# --------------------------------------------------
# \int_{\Gamma_F} alpha_FOURIER u v ds
func = lambda x, y: alpha_FOURIER
MASSE_BORD_CG = assemble_surface(
    mesh, ordre, func, "u", "v", methode="CG", domaine="FOURIER"
)

# --------------------------------------------------
# Taille de la matrice globale
# --------------------------------------------------
taille_MAT = nombre_dof_CG(mesh, ordre)

Nnz = (
    RIGIDITE_CGx.nnz
    + RIGIDITE_CGy.nnz
    + MASSE_CG.nnz
    + MASSE_BORD_CG.nnz
)

MAT_EF_CG = COOMatrix(taille_MAT, taille_MAT, Nnz)

# --------------------------------------------------
# Assemblage de l'opérateur global
# --------------------------------------------------
MAT_EF_CG += RIGIDITE_CGx
MAT_EF_CG += RIGIDITE_CGy
MAT_EF_CG += MASSE_CG
MAT_EF_CG += MASSE_BORD_CG

# --------------------------------------------------
# Définition du second membre
# --------------------------------------------------
x0 = 0
y0 = 0.6
fsource = lambda x, y: np.exp(-25 * (x - x0)**2 - 25 * (y - y0)**2)

# \int_\Omega f v dx
F_CG = assemble_rhs_volume(mesh, ordre, fsource, operatorv="v", methode="CG")

# --------------------------------------------------
# Résolution
# --------------------------------------------------
U_sol_CG = MAT_EF_CG.solve(F_CG)

# --------------------------------------------------
# Visualisation
# --------------------------------------------------
plot_nodal_vector_CG(U_sol_CG, mesh, ordre, "Solution numérique CG",secondes=2)

U_sol_CG_in_DG = nodal_CG_to_DG(U_sol_CG, mesh, ordre)
plot_nodal_vector_DG(U_sol_CG_in_DG-U_sol_DG, mesh, ordre, "Comparaison CG et DG")