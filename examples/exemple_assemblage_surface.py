import numpy as np
from mes_packages import *

mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
ordre = 2

neighbors, neighbor_faces, edges_to_triangles, reference_BC, bc_name= build_neighborhood_structure_with_bc(mesh)
refer = reference_BC("FOURIER")
print("reference de Fourier",refer)



func = lambda x, y: 1
# Assemblage générique
MAT_1 = assemble_surface(mesh, ordre, func, "u", "v", methode="CG",domaine="FOURIER")

