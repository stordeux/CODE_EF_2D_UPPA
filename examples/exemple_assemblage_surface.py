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
# Détermination du type de MAT_1
print("MAT_1 est de type ", type(MAT_1))
ok, lam_min, eigvals = MAT_1.check_positive_definite()
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
ok, lam_min, eigvals = MAT_1.check_positive_definite()
print(ok)
print(lam_min)
print(eigvals)
assert ok, "probleme de positivité"
