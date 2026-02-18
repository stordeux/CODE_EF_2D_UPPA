import numpy as np
from mes_packages import *

mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
ordre = 2

func = lambda x, y: 1.0
F = assemble_rhs_volume(mesh, ordre, func, operatorv="v", methode="CG")

gunc = lambda x, y: x**2+y**2
G = assemble_rhs_volume(mesh, ordre, gunc, operatorv="v", methode="CG")
