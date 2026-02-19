import numpy as np
from mes_packages import *
from mes_packages.assemblage_general import precompute_ref, grad_base_ref, loc_to_glob_general,assemble_volume

mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)

ordre = 2

func = lambda x,y: 1
MAT_1=assemble_volume(mesh, ordre, func, "u", "v", methode="CG")
MAT_2 = build_masse_CG(mesh, ordre)

N_glob = MAT_1.shape[0]
nnz = len(MAT_1.data)
MAT = COOMatrix(N_glob, N_glob, 2*nnz)
MAT = MAT + MAT_1
MAT = MAT - MAT_2

print(MAT.is_zero())





