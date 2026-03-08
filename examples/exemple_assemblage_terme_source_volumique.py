import numpy as np
from mes_packages import *

# Génération d’un maillage 2D :
# on construit un carré de côté 0.3 contenant un cercle de rayon 0.1,
# avec une taille caractéristique de maille égale à 0.025.
mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)

# Ordre polynomial des éléments finis utilisés pour la discrétisation.
# Ici, on travaille avec des éléments d’ordre 2.
ordre = 2

# Définition d’un second membre volumique constant :
# f(x,y) = 1
func = lambda x, y: 1.0

# Assemblage du vecteur second membre associé à f sur le volume.
# L’option operatorv="v" indique que l’on assemble le terme linéaire
# classique de type ∫_Ω f v.
# L’option methode="CG" précise que l’on utilise un espace
# Continuous Galerkin.
F = assemble_rhs_volume(mesh, ordre, func, operatorv="v", methode="CG")

# Définition d’un second membre volumique non constant :
# g(x,y) = x² + y²
gunc = lambda x, y: x**2 + y**2

# Assemblage du vecteur second membre associé à g sur le volume.
# On obtient donc ici le vecteur correspondant au terme
# ∫_Ω (x² + y²) v.
G = assemble_rhs_volume(mesh, ordre, gunc, operatorv="v", methode="CG")