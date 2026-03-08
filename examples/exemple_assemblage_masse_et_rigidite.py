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
# Nombre total de degrés de liberté du problème éléments finis CG
# (Continuous Galerkin) pour le maillage donné et l'ordre du polynôme.
taille_MAT = nombre_dof_CG(mesh, ordre)
# Nombre total d’entrées non nulles attendu dans la matrice finale.
# On suppose que la matrice de rigidité finale sera la somme
# des contributions MAT_Kx et MAT_Ky (par exemple dérivées selon x et y),
# donc son nombre maximal d’entrées non nulles est la somme des deux.
Nnz = MAT_Kx.nnz + MAT_Ky.nnz
# Création d’une matrice creuse au format COO.
# Elle est de taille (taille_MAT × taille_MAT) et on réserve
# de la place pour Nnz coefficients non nuls.
Rigidite = COOMatrix(taille_MAT, taille_MAT, Nnz)

# Ajout des contributions de rigidité associées à la dérivée en x.
# Grâce à l’opérateur +=, les coefficients de MAT_Kx sont
# ajoutés directement dans la matrice Rigidite.
Rigidite += MAT_Kx 

# Ajout des contributions de rigidité associées à la dérivée en y.
# La matrice finale Rigidite correspond donc à
# Rigidite = MAT_Kx + MAT_Ky.
Rigidite += MAT_Ky






