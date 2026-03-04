import numpy as np
import pytest
from mes_packages import *
from mes_packages.methode_hyperbolique import build_exemple_1_F0, build_exemple_1_F1, give_F_format


mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.1)
# plot_mesh(mesh,secondes=2)

ZERO = lambda x, y: 0.0

Lambda = 1.3
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

@pytest.mark.parametrize("ordre", [2, 3, 4])
def test_sol_volume_acoustique(ordre):
    MASSE_H = assemble_hyperbo(mesh, ordre, opu="u", opv="v", F=F0, kind="volume", methode="DG", domaine="all")
    F1D1 = assemble_hyperbo(mesh, ordre, opu="dxu", opv="v", F=Fx, kind="volume", methode="DG", domaine="all")
    F2D2 = assemble_hyperbo(mesh, ordre, opu="dyu", opv="v", F=Fy, kind="volume", methode="DG", domaine="all")

    func_vec = exemple_fonction_vectorielle(theta=np.pi/12, kappa=kappa)
    vec_nodal = build_vecteur_nodal_hyperbolique(mesh, ordre, func_vec=func_vec, methode="DG")
    print("vecteur nodal construit.")   
    # plot_nodal_vector_hyperbolique(mesh, vec_nodal, d=3, ordre=ordre, methode="DG")

    MAT = COOMatrix(MASSE_H.shape[0], MASSE_H.shape[1], MASSE_H.nnz + F1D1.nnz + F2D2.nnz)  # Estimation du nombre d'entrées non nulles
    MAT = MAT + MASSE_H 
    MAT = MAT + F1D1 
    MAT = MAT + F2D2

    assert np.linalg.norm(MAT@vec_nodal)<10**(-ordre+1), "Le vecteur nodal n'est pas une solution approchée du système hyperbolique."

