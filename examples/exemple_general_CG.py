import numpy as np
import sympy as sp
from mes_packages import *

 # Données physiques 
lambda_waves = 1/2
kappa = 2 * np.pi / lambda_waves
alpha = 1j * kappa

mesh = create_mesh_circle_in_square(radius=0.3, square_size=3, mesh_size=0.05)
ordre =2

# Définition des coefficients pour les matrices
a = lambda x, y: -kappa**2 *(1+x**2) 
b1 = lambda x, y: alpha * .5  
b2 = lambda x, y: alpha * .1*y
c = lambda x, y: 1/(1+x**2+y**2)
alpha_F = lambda x, y: 1j * kappa * (1+.1*np.cos(x+y))
# Matrice de masse, de rigidité et mixte pour les éléments de type CG
MASSE_CG = assemble_volume(mesh, ordre, c, "u", "v", methode="CG")
# ou 
# MASSE_CG = build_masse_CG(mesh, ordre, verbose=False)
MIXTE_CGx = assemble_volume(mesh, ordre, b1, "dxu", "v", methode="CG")
MIXTE_CGy = assemble_volume(mesh, ordre, b2, "dyu", "v", methode="CG")
RIGIDITE_CGx = assemble_volume(mesh, ordre, a, "dxu", "dxv", methode="CG")
RIGIDITE_CGy = assemble_volume(mesh, ordre, a, "dyu", "dyv", methode="CG")
MASSE_BORD_CG_Fourier = assemble_surface(mesh, ordre, alpha_F, "u", "v", methode="CG",domaine="FOURIER")
# Réservation de la mémoire de la matrice globale
taille_MAT = nombre_dof_CG(mesh, ordre)
Nnz = RIGIDITE_CGx.nnz + RIGIDITE_CGy.nnz + MASSE_CG.nnz + MASSE_BORD_CG_Fourier.nnz + MIXTE_CGx.nnz + MIXTE_CGy.nnz
MAT_EF_CG = COOMatrix(taille_MAT, taille_MAT, Nnz)
# Assemblage des matrices de masse et rigidité dans la matrice globale
MAT_EF_CG += RIGIDITE_CGx
MAT_EF_CG += RIGIDITE_CGy
MAT_EF_CG += MIXTE_CGx
MAT_EF_CG += MIXTE_CGy
MAT_EF_CG += MASSE_CG
MAT_EF_CG += MASSE_BORD_CG_Fourier

# Construction du terme soure
x0=0
y0=.6
fsource = lambda x, y: np.exp(- 25* (x-x0)**2- 25*(y-y0)**2)
F_CG = assemble_rhs_volume(mesh, ordre, fsource, operatorv="v", methode="CG")

# Résolution du système linéaire
U_sol = MAT_EF_CG.solve(F_CG)

# Visualisation de la source
Fsource = build_nodal_vector_CG(fsource, mesh, ordre)
plot_nodal_vector_CG(Fsource, mesh, ordre, "La source",secondes=2)

# Visualisation du résultat
plot_nodal_vector_CG(U_sol, mesh, ordre, "Solution numérique CG")

