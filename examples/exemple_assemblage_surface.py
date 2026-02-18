import numpy as np
from mes_packages import *

mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
plot_mesh(mesh)

ordre = 3

func = lambda x,y: 1
assemble_surface(mesh, ordre, func, "u", "v", methode="CG", domaine = "all")
