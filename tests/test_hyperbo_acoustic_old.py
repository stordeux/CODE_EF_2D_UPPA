import numpy as np
import pytest
from mes_packages import *
from mes_packages import mesh

ZERO = lambda x, y: 0.0
UN = lambda x, y: 1.0

Lambda = 1.3
kappa = 2 * np.pi / Lambda


def build_acoustic_matrices(kappa):
    h = lambda x, y: -1.j * kappa

    F0 = np.array([
        [h, ZERO, ZERO],
        [ZERO, h, ZERO],
        [ZERO, ZERO, h]
    ], dtype=object)

    Fx = np.array([
        [ZERO, UN, ZERO],
        [UN, ZERO, ZERO],
        [ZERO, ZERO, ZERO]
    ], dtype=object)

    Fy = np.array([
        [ZERO, ZERO, UN],
        [ZERO, ZERO, ZERO],
        [UN, ZERO, ZERO]
    ], dtype=object)

    return F0, Fx, Fy


@pytest.fixture
def mesh_volume():
    return create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.1)


@pytest.fixture
def mesh_ipp():
    return create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.03)


@pytest.mark.parametrize("ordre", [2, 3, 4])
def test_sol_volume_acoustique(mesh_volume, ordre):
    F0, Fx, Fy = build_acoustic_matrices(kappa)

    MASSE_H = assemble_hyperbo(mesh_volume, ordre, opu="u", opv="v", F=F0, kind="volume", methode="DG", domaine="all")
    F1D1 = assemble_hyperbo(mesh_volume, ordre, opu="dxu", opv="v", F=Fx, kind="volume", methode="DG", domaine="all")
    F2D2 = assemble_hyperbo(mesh_volume, ordre, opu="dyu", opv="v", F=Fy, kind="volume", methode="DG", domaine="all")


    MAT = COOMatrix(MASSE_H.shape[0], MASSE_H.shape[1], MASSE_H.nnz + F1D1.nnz + F2D2.nnz)
    MAT += MASSE_H
    MAT += F1D1
    MAT += F2D2

    func_vec = exemple_fonction_vectorielle(theta=np.pi/12, kappa=kappa)
    vec_nodal = build_vecteur_nodal_hyperbolique(mesh_volume, ordre, func_vec=func_vec, methode="DG")

    res = np.linalg.norm(MAT @ vec_nodal)
    assert res < 10 ** (-ordre + 1), f"Résidu trop grand : {res}"


def test_integration_by_parts_hyperbolic(mesh_ipp):
    ordre = 3
    _, Fx, Fy = build_acoustic_matrices(kappa)

    Dx = assemble_hyperbo(mesh_ipp, ordre, opu="dxu", opv="v", F=Fx, kind="volume", methode="DG")
    Dy = assemble_hyperbo(mesh_ipp, ordre, opu="dyu", opv="v", F=Fy, kind="volume", methode="DG")

    DxT = assemble_hyperbo(mesh_ipp, ordre, opu="u", opv="dxv", F=Fx, kind="volume", methode="DG")
    DyT = assemble_hyperbo(mesh_ipp, ordre, opu="u", opv="dyv", F=Fy, kind="volume", methode="DG")

    Bextx = assemble_hyperbo(mesh_ipp, ordre, opu="nxu", opv="v", F=Fx, kind="frontiere", methode="DG")
    Bexty = assemble_hyperbo(mesh_ipp, ordre, opu="nyu", opv="v", F=Fy, kind="frontiere", methode="DG")

    Bintx = assemble_hyperbo(mesh_ipp, ordre, opu="nxu", opv="v", F=Fx, kind="squelette_element", methode="DG")
    Binty = assemble_hyperbo(mesh_ipp, ordre, opu="nyu", opv="v", F=Fy, kind="squelette_element", methode="DG")
    T= Dx+Dy+ DxT+DyT - Bextx - Bexty - Bintx - Binty
    assert T.is_zero()

def test_penalisation_interieure(mesh_ipp):
    ordre = 3
    func_F_p = np.array([
        [UN,   ZERO, ZERO],
        [ZERO, ZERO, ZERO],
        [ZERO, ZERO, ZERO]], dtype=object)
    func_F_vx = np.array([
        [ZERO, ZERO, ZERO],
        [ZERO, UN,   ZERO],
        [ZERO, ZERO, ZERO]], dtype=object)
    func_F_vy = np.array([
        [ZERO, ZERO, ZERO],
        [ZERO, ZERO, ZERO],
        [ZERO, ZERO, UN]], dtype=object)
    func_F_cross = np.array([
        [ZERO, ZERO, ZERO],
        [ZERO, ZERO, UN  ],
        [ZERO, UN,   ZERO]], dtype=object)
    A_p_TT = assemble_hyperbo(
        mesh_ipp, ordre,
        opu="uT", opv="vT",
        F=func_F_p,
        kind="squelette_element",
        methode="DG",
        domaine="all")
    A_p_TV = assemble_hyperbo(
        mesh_ipp, ordre,
        opu="uT", opv="vV",
        F=func_F_p,
        kind="squelette_element",
        methode="DG",
        domaine="all")
    A_xx_TT = assemble_hyperbo(
        mesh_ipp, ordre,
        opu="nxuT", opv="nxvT",
        F=func_F_vx,
        kind="squelette_element",
        methode="DG",
        domaine="all")
    A_xx_TV = assemble_hyperbo(
        mesh_ipp, ordre,
        opu="nxuT", opv="nxvV",
        F=func_F_vx,
        kind="squelette_element",
        methode="DG",
        domaine="all")
    A_yy_TT = assemble_hyperbo(
        mesh_ipp, ordre,
        opu="nyuT", opv="nyvT",
        F=func_F_vy,
        kind="squelette_element",
        methode="DG",
        domaine="all")
    A_yy_TV = assemble_hyperbo(
        mesh_ipp, ordre,
        opu="nyuT", opv="nyvV",
        F=func_F_vy,
        kind="squelette_element",
        methode="DG",
        domaine="all")
    A_yx_TT = assemble_hyperbo(
        mesh_ipp, ordre,
        opu="nyuT", opv="nxvT",
        F=func_F_cross,
        kind="squelette_element",
        methode="DG",
        domaine="all")
    A_yx_TV = assemble_hyperbo(
        mesh_ipp, ordre,
        opu="nyuT", opv="nxvV",
        F=func_F_cross,
        kind="squelette_element",
        methode="DG",
        domaine="all")
    nx = A_xx_TT.shape[0]
    ny = A_xx_TT.shape[1]
    nnz = (
        A_p_TT.nnz + A_p_TV.nnz
        + A_xx_TT.nnz + A_xx_TV.nnz
        + A_yy_TT.nnz + A_yy_TV.nnz
        + A_yx_TT.nnz + A_yx_TV.nnz)
    # Assemblage de la matrice de penalisation
    PENALISATION = COOMatrix(nx,ny,nnz)
    PENALISATION += A_p_TT
    PENALISATION -= A_p_TV
    PENALISATION += A_xx_TT
    PENALISATION += A_xx_TV
    PENALISATION += A_yy_TT
    PENALISATION += A_yy_TV
    PENALISATION += A_yx_TT
    PENALISATION += A_yx_TV
    assert PENALISATION.is_positive(), "La matrice de pénalisation n'est pas positive"
    func_vec = exemple_fonction_vectorielle(theta=np.pi/12, kappa=kappa)
    vec_nodal = build_vecteur_nodal_hyperbolique(mesh_ipp, ordre, func_vec=func_vec, methode="DG")
    res = PENALISATION @ vec_nodal
    assert np.all(np.abs(res) < 10 ** -8), f"Résidu trop grand : {np.linalg.norm(res)}"