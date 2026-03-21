import numpy as np
import pytest
from mes_packages import *


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


def build_penalty_matrices():
    func_F_p = np.array([
        [UN,   ZERO, ZERO],
        [ZERO, ZERO, ZERO],
        [ZERO, ZERO, ZERO]
    ], dtype=object)

    func_F_vx = np.array([
        [ZERO, ZERO, ZERO],
        [ZERO, UN,   ZERO],
        [ZERO, ZERO, ZERO]
    ], dtype=object)

    func_F_vy = np.array([
        [ZERO, ZERO, ZERO],
        [ZERO, ZERO, ZERO],
        [ZERO, ZERO, UN]
    ], dtype=object)

    func_F_cross = np.array([
        [ZERO, ZERO, ZERO],
        [ZERO, ZERO, UN],
        [ZERO, UN,   ZERO]
    ], dtype=object)

    return func_F_p, func_F_vx, func_F_vy, func_F_cross



@pytest.fixture
def mesh_volume():
    return create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.1)


@pytest.fixture
def mesh_ipp():
    return create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.1)

@pytest.fixture
def mesh_rhs():
    return create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.01)


@pytest.mark.parametrize("ordre", [2, 3, 4])
def test_sol_volume_acoustique(mesh_volume, ordre):
    MAT = assemble_volume_hyperbolique(mesh_volume, ordre, kappa,modele="acoustique", formulation="forte")
    func_vec = onde_plane_acoustique(theta=np.pi / 12, kappa=kappa)
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

    T = Dx + Dy + DxT + DyT - Bextx - Bexty - Bintx - Binty
    assert T.is_zero()


def test_penalisation_interieure(mesh_ipp):
    ordre = 3
    Lambda=1.3
    kappa = 1.j * 2 * np.pi / Lambda
    PENALISATION = assemble_squelette_hyperbolique(mesh_ipp, ordre, kappa,Z=2,formulation="Fplus",modele="acoustique")
    assert PENALISATION.is_positive(), "La matrice de pénalisation n'est pas positive"

    func_vec = onde_plane_acoustique(theta=np.pi / 12, kappa=kappa)
    vec_nodal = build_vecteur_nodal_hyperbolique(mesh_ipp, ordre, func_vec=func_vec, methode="DG")
    res = PENALISATION @ vec_nodal
    assert np.all(np.abs(res) < 10 ** -8), f"Résidu trop grand : {np.linalg.norm(res)}"

def test_ipp_hyperbolic(mesh_ipp):
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
    IPP = Dx + Dy + DxT + DyT - Bextx - Bexty - Bintx - Binty
    assert IPP.is_zero(), "L'identité d'intégration par parties discrète n'est pas vérifiée"

def test_saut_reciprocite(mesh_ipp):
    ordre = 3
    Lambda=1.3
    kappa = 1.j * 2 * np.pi / Lambda
    Saut_reciprocite =  assemble_squelette_hyperbolique(mesh_ipp, ordre, kappa, Z=2, formulation="Fplus", modele="acoustique")
    func_vec = onde_plane_acoustique(theta=np.pi / 12, kappa=kappa)
    vec_nodal = build_vecteur_nodal_hyperbolique(mesh_ipp, ordre, func_vec=func_vec, methode="DG")
    assert np.linalg.norm(Saut_reciprocite@vec_nodal)<1e-10, "Le saut de réciprocité n'est pas nul"


def test_continuite_Fmoins(mesh_ipp):
    ordre = 3
    Lambda=1.3
    kappa = 1.j * 2 * np.pi / Lambda
    MATRICE_INTERIEURE = assemble_squelette_hyperbolique(mesh_ipp, ordre, kappa, Z=2, formulation="Fmoins", modele="acoustique")
    func_vec = onde_plane_acoustique(theta=np.pi / 12, kappa=kappa)
    vec_nodal = build_vecteur_nodal_hyperbolique(mesh_ipp, ordre, func_vec=func_vec, methode="DG")
    erreur = np.linalg.norm(MATRICE_INTERIEURE @ vec_nodal)
    assert erreur < 1e-8, f"Résidu de continuité trop grand : {erreur}"

def test_positivite_frontiere(mesh_ipp):
    ordre = 3
    Lambda=1.3
    kappa = 1.j * 2 * np.pi / Lambda
    MAT = assemble_frontiere_hyperbolique(mesh_ipp, ordre, kappa,modele="acoustique", formulation="positif")
    assert MAT.is_positive(), "test du caractère positif de la matrice"

def test_positivite_volume(mesh_volume):
    ordre = 3
    Lambda=1.3
    kappa = 1.j * 2 * np.pi / Lambda
    Mat_coercive = assemble_volume_hyperbolique(mesh_volume, ordre, kappa,modele="acoustique", formulation="positif")
    assert Mat_coercive.is_positive(), "test du caractère positif de la matrice"

def test_matrice_sauf_frontière(mesh_ipp):
    ordre = 4
    Lambda=1.3
    kappa = 1.j * 2 * np.pi / Lambda
    MAT = assemble_matrice_totale_sans_frontiere(mesh_ipp, ordre, kappa)


    func_vec = onde_plane_acoustique(theta=np.pi / 12, kappa=kappa)
    vec_nodal = build_vecteur_nodal_hyperbolique(mesh_ipp, ordre, func_vec=func_vec, methode="DG")
    erreur = np.linalg.norm(MAT @ vec_nodal)
    assert erreur < 1e-6, f"Résidu de continuité trop grand : {erreur}"


def test_frontiere_consistante(mesh_rhs):
    ordre = 4
    Lambda=1.3
    kappa = 1.j * 2 * np.pi / Lambda
    MAT = assemble_frontiere_hyperbolique(mesh_rhs, ordre, kappa,modele="acoustique", formulation="Fplus", Z=2)
    assert MAT.is_positive(), "La matrice de frontière consistante n'est pas positive"
    func_vec = onde_plane_acoustique(theta=np.pi / 12, kappa=kappa)
    vec_nodal = build_vecteur_nodal_hyperbolique(mesh_rhs, ordre, func_vec=func_vec, methode="DG")
    RHS = assemble_rhs_acoustique(mesh_rhs, ordre, func_vec=func_vec, Z=2.0)
    erreur = np.linalg.norm(MAT @ vec_nodal-RHS)
    assert erreur < 1e-6, f"Résidu de frontière consistante trop grand : {erreur}"



