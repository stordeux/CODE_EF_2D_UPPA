import numpy as np
from mes_packages import *

# Création d’un maillage 2D :
# domaine = carré de côté 0.3 contenant un trou circulaire de rayon 0.1
# mesh_size contrôle la finesse du maillage
mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
ordre = 2  # Ordre polynomial des éléments finis

# Assemblage de la matrice de surface
func = lambda x, y: 1

# Assemblage de la matrice de bord associée à la forme :
# \int_{\Gamma_F} u v \, dS
M_bord = assemble_surface(mesh, ordre, func, "u", "v", methode="CG", domaine="FOURIER")

# Affichage du type de la matrice obtenue
print("M_bord est de type ", type(M_bord))

# Test de positivité de la matrice
ok, lam_min, eigvals = M_bord.check_positive_definite()
print("Les valeurs propres sont positives",ok)


# Définition d’un champ vectoriel M = (Mx, My)
Mx = lambda x, y: x**2
My = lambda x, y: x + y
M = make_vector_field(Mx, My)

# Assemblage de la matrice associée à la forme :
# \int_{\Gamma_F} (M.n)^2 u v \, dS
# car "(M.n)u" contre "(M.n)v" donne un facteur (M.n)(M.n)
MAT_1 = assemble_surface(mesh, ordre, M, "(M.n)u", "(M.n)v", methode="CG", domaine="FOURIER")
# Test de positivité de la matrice
ok, lam_min, eigvals = MAT_1.check_positive_definite()
print("Les valeurs propres sont positives",ok)

# Assemblage de la matrice associée à la forme :
# \int_{\Gamma_F} (M.n) u v \, dS
# ici seul le facteur de gauche porte (M.n)
MAT_1 = assemble_surface(mesh, ordre, M, "(M.n)u", "v", methode="CG", domaine="FOURIER")
# Test lent de positivité
ok, lam_min, eigvals = MAT_1.is_positive_slow()
print("Les valeurs propres sont positives", ok,"Attendu False")


