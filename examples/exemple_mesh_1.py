import numpy as np
from mes_packages import *


# Création d'un maillage d'un cercle dans un carré
mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
nb_corriges = verifier_et_corriger_orientation(mesh)
plot_mesh(mesh,secondes=2)
plot_mesh_with_bc(mesh,secondes=2)
