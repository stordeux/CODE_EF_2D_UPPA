import numpy as np

# Importe toutes les fonctions exposées dans mes_packages
# (COOMatrix, create_mesh_circle_in_square, build_masse_CG, etc.)
from mes_packages import *


# Création d’un maillage 2D :
# domaine = carré de côté 0.3 contenant un trou circulaire de rayon 0.1
# mesh_size contrôle la finesse du maillage
mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
ordre = 2  # Ordre polynomial des éléments finis
# Assemblage de la matrice de masse
func = lambda x,y: 1
MAT_M=assemble_volume(mesh, ordre, func, "u", "v", methode="CG")
# Assemblage de la matrice de rigidité
func = lambda x,y: 1
MAT_Kx=assemble_volume(mesh, ordre, func, "dxu", "dxv", methode="CG")
MAT_Ky=assemble_volume(mesh, ordre, func, "dyu", "dyv", methode="CG")

taille_MAT = nombre_dof_CG(mesh, ordre)
Nnz = MAT_Kx.nnz + MAT_Ky.nnz
Rigidite = COOMatrix(taille_MAT, taille_MAT, Nnz)
Rigidite += MAT_Kx 
Rigidite += MAT_Ky





