import numpy as np
from mes_packages import *

# Création d’un maillage 2D :
# domaine = carré de côté 0.3 contenant un trou circulaire de rayon 0.1
# mesh_size contrôle la finesse du maillage
mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
ordre = 2 # Ordre polynomial des éléments finis

# Assemblage de la matrice de la surface
func = lambda x, y: 1
# Assemblage de la matrice de bord
# Matrice de représentation 
# \int_{\Gamma_F} u v dS
M_bord = assemble_surface(mesh, ordre, func, "u", "v", methode="CG",domaine="FOURIER")
# Détermination du type de M_bord
print("M_bord est de type ", type(M_bord))
ok, lam_min, eigvals = M_bord.check_positive_definite()
print(ok)
print(lam_min)
print(eigvals)


func = lambda x, y: 1
# Assemblage générique
MAT_1 = assemble_surface(mesh, ordre, func, "(M.n)u", "(M.n)v", methode="CG",domaine="FOURIER")
# Détermination du type de MAT_1
print("MAT_1 est de type ", type(MAT_1))
ok, lam_min, eigvals = MAT_1.check_positive_definite()
print(ok)
print(lam_min)
print(eigvals)

ordre = 1
mesh = create_mesh_circle_in_square(0.1, 0.3, 0.1)
func = lambda x, y: 1
# Assemblage générique
MAT_1 = assemble_surface(mesh, ordre, func, "u", "v", methode="CG",domaine="FOURIER")
# Détermination du type de MAT_1
print("MAT_1 est de type ", type(MAT_1))
ok, lam_min, eigvals = MAT_1.is_positive_slow()
print(ok)
print(lam_min)
print(eigvals)
assert ok, "probleme de positivité"
