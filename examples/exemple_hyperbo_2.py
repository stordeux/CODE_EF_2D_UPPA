import numpy as np
from mes_packages import *
from mes_packages.methode_hyperbolique import build_exemple_1_F0, build_exemple_1_F1, give_F_format,assemble_surface

ordre =3


# Fin du programme de test de give_F_format
mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
plot_mesh(mesh,secondes=2)

ZERO = lambda x, y: 0.0

Lambda = .1
kappa = 2*np.pi/Lambda

f = lambda x, y: 1 
g = lambda x, y: 1 
h = lambda x, y: -1.j * kappa


F0=np.array([
        [h, ZERO, ZERO],
        [ZERO, h, ZERO],
        [ZERO, ZERO, h]
    ], dtype=object)

Fx = np.array([
        [ZERO, f, ZERO],
        [f, ZERO, ZERO],
        [ZERO, ZERO, ZERO]
    ], dtype=object)
Fy = np.array([
        [ZERO, ZERO,g],
        [ZERO, ZERO, ZERO],
        [g, ZERO, ZERO]
    ], dtype=object)

func_vec = exemple_fonction_vectorielle(theta=np.pi/12, kappa=kappa)
vec_nodal = build_vecteur_nodal_hyperbolique(mesh, ordre, func_vec=func_vec, methode="DG")
print("vecteur nodal construit.")   

FpartialOmeganx = assemble_hyperbo(mesh, ordre, opu="nxu", opv="v", F=Fx, kind="frontiere",  methode="DG", domaine="all")
FpartialOmegany = assemble_hyperbo(mesh, ordre, opu="nyu", opv="v", F=Fy, kind="frontiere",  methode="DG", domaine="all")
Finterieurnx = assemble_hyperbo(mesh, ordre, opu="nxu", opv="v", F=Fx, kind="squelette_element",  methode="DG")
Finterieurny = assemble_hyperbo(mesh, ordre, opu="nyu", opv="v", F=Fy, kind="squelette_element",  methode="DG")

taille = FpartialOmeganx.shape[0]
nnz = FpartialOmeganx.nnz+FpartialOmegany.nnz+Finterieurnx.nnz+Finterieurny.nnz
MATfront = COOMatrix(taille, taille,nnz)  # Estimation du nombre d'entrées non nulles

# plot_nodal_vector_hyperbolique(mesh, vec_nodal, d=3, ordre=ordre, methode="DG")

MATfront = MATfront + FpartialOmeganx
MATfront = MATfront + FpartialOmegany
MATfront = MATfront + Finterieurnx
MATfront = MATfront + Finterieurny


# print(np.linalg.norm(MAT@vec_nodal))

