import numpy as np
import sympy as sp
from mes_packages import *

# Construction du maillage
mesh = create_mesh_circle_in_square(radius=0.3, square_size=3, mesh_size=0.05)


 # Données physiques 
sigma =20
ordre =2
lambda_waves = 1/2
kappa = 2 * np.pi / lambda_waves
hmin = compute_h_min(mesh)
alpha_pen = 1.j*sigma*(ordre+1)**2 / hmin
alpha_FOURIER = 1j * kappa
hmin = compute_h_min(mesh)


func = lambda x, y: -kappa**2  # Fonction de test (constante)
MASSE_DG = assemble_volume(mesh, ordre, func, "u", "v", methode="DG")
# ou 
# MASSE_CG = build_masse_CG(mesh, ordre, verbose=False)

func = lambda x, y: 1  # Fonction de test (constante)
RIGIDITE_DGx = assemble_volume(mesh, ordre, func, "dxu", "dxv", methode="DG")
RIGIDITE_DGy = assemble_volume(mesh, ordre, func, "dyu", "dyv", methode="DG")
func = lambda x, y: alpha_FOURIER  # Fonction de test (constante)  
MASSE_BORD_DG = assemble_surface(mesh, ordre, func, "u", "v", methode="DG",domaine="FOURIER")
taille_MAT = nombre_dof_DG(mesh, ordre)

A_SIPG1 = assemble_skeleton_par_face(mesh, ordre, -1,
                                     operatoru="moynablau",
                                     operatorv="sautDGv",
                                     methode="DG")
A_SIPG2 = assemble_skeleton_par_face(mesh, ordre, -1,
                                     operatoru="sautDGu",
                                     operatorv="moynablav",
                                     methode="DG")
A_saut_saut = assemble_skeleton_par_face(mesh, ordre, alpha_pen,
                                     operatoru="sautu",operatorv="sautv",
                                     methode="DG")


Nnz = RIGIDITE_DGx.nnz + RIGIDITE_DGy.nnz + MASSE_DG.nnz + MASSE_BORD_DG.nnz + A_SIPG1.nnz + A_SIPG2.nnz + A_saut_saut.nnz
MAT_EF_DG = COOMatrix(taille_MAT, taille_MAT, Nnz)
# Assemblage des matrices de masse et rigidité dans la matrice globale
MAT_EF_DG = MAT_EF_DG + RIGIDITE_DGx
MAT_EF_DG = MAT_EF_DG + RIGIDITE_DGy
MAT_EF_DG = MAT_EF_DG + A_SIPG1
MAT_EF_DG = MAT_EF_DG + A_SIPG2
MAT_EF_DG = MAT_EF_DG + A_saut_saut
MAT_EF_DG =MAT_EF_DG + MASSE_DG
MAT_EF_DG = MAT_EF_DG + MASSE_BORD_DG
x0=0
y0=.6
fsource = lambda x, y: np.exp(- 25* (x-x0)**2- 25*(y-y0)**2)

F_DG = assemble_rhs_volume(mesh, ordre, fsource, operatorv="v", methode="DG")

U_sol = MAT_EF_DG.solve(F_DG)

Fsource = build_nodal_vector_DG(fsource, mesh, ordre)
plot_nodal_vector_DG(Fsource, mesh, ordre, "Solution numérique DG")

# Visualisation du résultat
plot_nodal_vector_DG(U_sol, mesh, ordre, "Solution numérique DG")

