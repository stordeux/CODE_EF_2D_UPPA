import numpy as np
from mes_packages import *


# Création d'un polygone
points = [
    (-4,0),
    (0,0),
    (0,1.5),
    (2,1.5),
    (2,0),
    (6,0),
    (6,4),
    (2,4),
    (2,2.5),
    (0,2.5),
    (0,4),
    (-4,4)
]
mesh = create_mesh_from_polygon(points, mesh_size=0.2)
plot_mesh(mesh,secondes=2)
plot_mesh_with_bc(mesh,secondes=2)


