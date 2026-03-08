# Construction du maillage
import numpy as np
import sympy as sp
from mes_packages import *

mesh = create_mesh_circle_in_square(radius=1, square_size=8, mesh_size=0.25)
plot_mesh(mesh,secondes=2)
# Définition de l'ordre de la méthode 
ordre =3
# Assemblage des différentes matrices

func = lambda x, y: 1.0  # Fonction de poids (constante)
# Matrice de masse
MASSE_CG = assemble_volume(mesh, ordre, func, "u", "v", methode="CG")
# ou 
# MASSE_CG = build_masse_CG(mesh, ordre, verbose=False)
# Matrices de rigidité
RIGIDITE_CGx = assemble_volume(mesh, ordre, func, "dxu", "dxv", methode="CG")
RIGIDITE_CGy = assemble_volume(mesh, ordre, func, "dyu", "dyv", methode="CG")
# Matrice mixte 
CONVECTION_CGy = assemble_volume(mesh, ordre, func, "dxu", "v", methode="CG")
# Construction des matrices de frontière
func = lambda x, y: 1.0  # Fonction de test (constante)
MASSE_BORD_CG = assemble_surface(mesh, ordre, func, "u", "v", methode="CG",domaine="all")
# ou
# MASSE_BORD_CG = build_masse_frontiere_CG(mesh, ordre) 
MASSE_BORD_CG_FOURIER = assemble_surface(mesh, ordre, func, "u", "v", methode="CG",domaine="FOURIER")
# ou
# MASSE_BORD_CG_FOURIER = build_masse_frontiere_CG(mesh, ordre,"Fourier") 

# Assemblage de la matrice éléments finis 
# Résolution du problème de Fourier sur les frontières intérieures et extérieures
lambda_waves = 1
kappa = 2 * np.pi / lambda_waves
alpha = 1j * kappa
# Création de la matrice globale C0
taille_MAT = nombre_dof_CG(mesh, ordre)
Nnz = RIGIDITE_CGx.nnz + RIGIDITE_CGy.nnz + MASSE_CG.nnz + MASSE_BORD_CG_FOURIER.nnz
MAT_EF_CG = COOMatrix(taille_MAT, taille_MAT, Nnz)


# Assemblage des matrices de masse et rigidité dans la matrice globale
MAT_EF_CG += RIGIDITE_CGx
MAT_EF_CG += RIGIDITE_CGy
kappa2_MASSE_CG = kappa**2 * MASSE_CG
MAT_EF_CG -= kappa2_MASSE_CG
ik_MASSE_BORD_CG_FOURIER = alpha * MASSE_BORD_CG_FOURIER
MAT_EF_CG += ik_MASSE_BORD_CG_FOURIER
# Construction du terme source
x_0=1.0
y_0=1.0
f_source = lambda x, y: np.exp(-10*(x-x_0)**2 -10*(y-y_0)**2)  # Source gaussienne centrée en (x0, y0)
F_volume = assemble_rhs_volume(mesh, ordre, f_source, "v", methode="CG")
# Représentation du terme source
f_vec_source = build_nodal_vector_CG(f_source,mesh, ordre)
plot_nodal_vector_CG(f_vec_source, mesh, ordre, "Solution numérique CG")




# Résolution du système linéaire K_CG * U_sol = F_CG
U_sol_FOURIER = MAT_EF_CG.solve(F_volume)
plot_nodal_vector_CG(U_sol_FOURIER, mesh, ordre, "Solution numérique CG")