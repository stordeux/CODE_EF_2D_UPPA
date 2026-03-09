import numpy as np
import sympy as sp
from mes_packages import *

 # Données physiques 
lambda_waves = 1/2
kappa = 2 * np.pi / lambda_waves
alpha = 1j * kappa

mesh = create_mesh_circle_in_square(radius=0.3, square_size=3, mesh_size=0.05)
ordre =2

func = lambda x, y: -kappa**2  # Fonction de test (constante)
MASSE_CG = assemble_volume(mesh, ordre, func, "u", "v", methode="CG")
# ou 
# MASSE_CG = build_masse_CG(mesh, ordre, verbose=False)

func = lambda x, y: alpha * .5  # Fonction de test (constante)
MIXTE_CG = assemble_volume(mesh, ordre, func, "dxu", "v", methode="CG")

func = lambda x, y: 1  # Fonction de test (constante)
RIGIDITE_CGx = assemble_volume(mesh, ordre, func, "dxu", "dxv", methode="CG")
RIGIDITE_CGy = assemble_volume(mesh, ordre, func, "dyu", "dyv", methode="CG")
func = lambda x, y: alpha  
MASSE_BORD_CG = assemble_surface(mesh, ordre, func, "u", "v", methode="CG",domaine="FOURIER")


taille_MAT = nombre_dof_CG(mesh, ordre)
Nnz = RIGIDITE_CGx.nnz + RIGIDITE_CGy.nnz + MASSE_CG.nnz + MASSE_BORD_CG.nnz + MIXTE_CG.nnz
MAT_EF_CG = COOMatrix(taille_MAT, taille_MAT, Nnz)
# Assemblage des matrices de masse et rigidité dans la matrice globale
MAT_EF_CG += RIGIDITE_CGx
MAT_EF_CG += RIGIDITE_CGy
MAT_EF_CG += MIXTE_CG
MAT_EF_CG += MASSE_CG
MAT_EF_CG += MASSE_BORD_CG
x0=0
y0=.6
fsource = lambda x, y: np.exp(- 25* (x-x0)**2- 25*(y-y0)**2)

F_CG = assemble_rhs_volume(mesh, ordre, fsource, operatorv="v", methode="CG")

U_sol = MAT_EF_CG.solve(F_CG)

Fsource = build_nodal_vector_CG(fsource, mesh, ordre)
plot_nodal_vector_CG(Fsource, mesh, ordre, "La source",secondes=2)

# Visualisation du résultat
plot_nodal_vector_CG(U_sol, mesh, ordre, "Solution numérique CG")

