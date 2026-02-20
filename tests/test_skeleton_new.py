import numpy as np
import pytest
from mes_packages import *

mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)


@pytest.mark.parametrize("ordre", [1,2,3,4])
def test_assemble_skeleton_1(ordre):
    # Nouveau : forme jump-jump
    A_new = assemble_skeleton_par_element(mesh, ordre, coef=1.0,
                              operatoru="uT", operatorv="vT",
                              methode="DG")
    # Ancien : build_jump_matrix_DG
    A_old = assemble_skeleton_par_element_old(mesh, ordre, coef=1.0,
                              operatoru="uT", operatorv="vT",
                              methode="DG")
    A_new.is_symmetric(tol=1e-8)
    MAT= COOMatrix(A_new.shape[0], A_new.shape[1], A_new.l + A_old.l)
    MAT = MAT + A_new
    MAT = MAT - A_old
    assert MAT.is_zero(tol=1e-8), "problème d'assemblage de la matrice de squelette (jump-jump)"

@pytest.mark.parametrize("ordre", [1,2,3,4])
def test_assemble_skeleton_2(ordre):
    # Nouveau : forme jump-jump
    A_new = assemble_skeleton_par_element(mesh, ordre, coef=1.0,
                              operatoru="uT", operatorv="vV",
                              methode="DG")
    # Ancien : build_jump_matrix_DG
    A_old = assemble_skeleton_par_element_old(mesh, ordre, coef=1.0,
                              operatoru="uT", operatorv="vV",
                              methode="DG")
    A_new.is_symmetric(tol=1e-8)
    MAT= COOMatrix(A_new.shape[0], A_new.shape[1], A_new.l + A_old.l)
    MAT = MAT + A_new
    MAT = MAT - A_old
    assert MAT.is_zero(tol=1e-8), "problème d'assemblage de la matrice de squelette (jump-jump)"


@pytest.mark.parametrize("ordre", [1,2,3,4])
def test_assemble_skeleton_3(ordre):
    # Nouveau : forme jump-jump
    A_new = assemble_skeleton_par_element(mesh, ordre, coef=1.0,
                              operatoru="uV", operatorv="vT",
                              methode="DG")
    # Ancien : build_jump_matrix_DG
    A_old = assemble_skeleton_par_element_old(mesh, ordre, coef=1.0,
                              operatoru="uV", operatorv="vT",
                              methode="DG")
    A_new.is_symmetric(tol=1e-8)
    MAT= COOMatrix(A_new.shape[0], A_new.shape[1], A_new.l + A_old.l)
    MAT = MAT + A_new
    MAT = MAT - A_old
    assert MAT.is_zero(tol=1e-8), "problème d'assemblage de la matrice de squelette (jump-jump)"


@pytest.mark.parametrize("ordre", [1,2,3,4])
def test_assemble_skeleton_4p(ordre):
    # Nouveau : forme jump-jump
    A_new = assemble_skeleton_par_element(mesh, ordre, coef=1.0,
                              operatoru="uV", operatorv="vV",
                              methode="DG")
    # Ancien : build_jump_matrix_DG
    A_old = assemble_skeleton_par_element_old(mesh, ordre, coef=1.0,
                              operatoru="uV", operatorv="vV",
                              methode="DG")
    A_new.is_symmetric(tol=1e-8)
    MAT= COOMatrix(A_new.shape[0], A_new.shape[1], A_new.l + A_old.l)
    MAT = MAT + A_new
    MAT = MAT - A_old
    assert MAT.is_zero(tol=1e-8), "problème d'assemblage de la matrice de squelette (jump-jump)"


@pytest.mark.parametrize("ordre", [1,2,3,4])
@pytest.mark.parametrize("beta", [-1.0, 1.0 ])
def test_assemble_skeleton_4(ordre,beta):
    # Nouveau : forme jump-jump
    A_new = assemble_skeleton_par_element(mesh, ordre, coef=1.0,
                              operatoru="uT", operatorv="vT",
                              methode="DG")
    # Ancien : build_jump_matrix_DG
    A_old = assemble_skeleton_par_element_old(mesh, ordre, coef=1.0,
                              operatoru="uV", operatorv="vV",
                              methode="DG")
    A_new.is_symmetric(tol=1e-8)
    MAT= COOMatrix(A_new.shape[0], A_new.shape[1], A_new.l + A_old.l)
    MAT = MAT + A_new
    MAT = MAT - A_old
    assert MAT.is_zero(tol=1e-8), "problème d'assemblage de la matrice de squelette (jump-jump)"

@pytest.mark.parametrize("ordre", [1,2,3,4])
@pytest.mark.parametrize("beta", [-1.0, 1.0 ])
def test_assemble_skeleton_5(ordre,beta):
    opuT = "uT"
    opvT = "vT"
    opuV = "uV"
    opvV = "vV"
    ATT = assemble_skeleton_par_element(mesh, ordre, 1.0,opuT,opvT,"DG")
    ATV = assemble_skeleton_par_element(mesh, ordre, beta,opuT,opvV,"DG")
    AVT = assemble_skeleton_par_element(mesh, ordre, beta,opuV,opvT,"DG")
    AVV = assemble_skeleton_par_element(mesh, ordre, 1.0,opuV,opvV,"DG")
    A = COOMatrix(ATT.shape[0], ATT.shape[1], ATT.l + ATV.l + AVT.l + AVV.l)
    A += ATT 
    A += ATV 
    A += AVT 
    A += AVV
    assert A.is_symmetric(tol=1e-8), "A nest pas symétrique"
    assert A.is_positive(tol=1e-8), "A n est pas définie positive"

@pytest.mark.parametrize("ordre", [1,2,3,4])
@pytest.mark.parametrize("beta", [-1.0, 1.0 ])
def test_assemble_skeleton_6(ordre,beta):
    opuT = "dnuT"
    opvT = "dnvT"
    opuV = "dnuV"
    opvV = "dnvV"
    ATT = assemble_skeleton_par_element(mesh, ordre, 1.0,opuT,opvT,"DG")
    ATV = assemble_skeleton_par_element(mesh, ordre, beta,opuT,opvV,"DG")
    AVT = assemble_skeleton_par_element(mesh, ordre, beta,opuV,opvT,"DG")
    AVV = assemble_skeleton_par_element(mesh, ordre, 1.0,opuV,opvV,"DG")
    A = COOMatrix(ATT.shape[0], ATT.shape[1], ATT.l + ATV.l + AVT.l + AVV.l)
    A += ATT 
    A += ATV 
    A += AVT 
    A += AVV
    assert A.is_symmetric(tol=1e-8), "A nest pas symétrique"
    assert A.is_positive(tol=1e-8), "A n est pas définie positive"


@pytest.mark.parametrize("ordre", [1,2,3,4])
def test_assemble_skeleton_7(ordre):
    beta=-1.0
    opuT = "uT"
    opvT = "vT"
    opuV = "uV"
    opvV = "vV"
    ATT = assemble_skeleton_par_element(mesh, ordre, 1.0,opuT,opvT,"DG")
    ATV = assemble_skeleton_par_element(mesh, ordre, beta,opuT,opvV,"DG")
    AVT = assemble_skeleton_par_element(mesh, ordre, beta,opuV,opvT,"DG")
    AVV = assemble_skeleton_par_element(mesh, ordre, 1.0,opuV,opvV,"DG")
    A = COOMatrix(ATT.shape[0], ATT.shape[1], ATT.l + ATV.l + AVT.l + AVV.l)
    A += ATT 
    A += ATV 
    A += AVT 
    A += AVV
    B = assemble_skeleton_par_element_old(mesh, ordre, 1.0,"sautu","sautv","DG")
    assert A.is_equal(B, tol=1e-8), "problème d'assemblage de la matrice de squelette (jump-jump)"

@pytest.mark.parametrize("ordre", [1,2,3,4])
def test_assemble_skeleton_8(ordre):
    beta=1.0
    opuT = "dnuT"
    opvT = "dnvT"
    opuV = "dnuV"
    opvV = "dnvV"
    ATT = assemble_skeleton_par_element(mesh, ordre, 1.0,opuT,opvT,"DG")
    ATV = assemble_skeleton_par_element(mesh, ordre, beta,opuT,opvV,"DG")
    AVT = assemble_skeleton_par_element(mesh, ordre, beta,opuV,opvT,"DG")
    AVV = assemble_skeleton_par_element(mesh, ordre, 1.0,opuV,opvV,"DG")
    A = COOMatrix(ATT.shape[0], ATT.shape[1], ATT.l + ATV.l + AVT.l + AVV.l)
    A += ATT 
    A += ATV 
    A += AVT 
    A += AVV
    B = assemble_skeleton_par_element_old(mesh, ordre, 1.0,"sautdnu","sautdnv","DG")
    assert A.is_equal(B, tol=1e-8), "problème d'assemblage de la matrice de squelette (jump-jump)"

@pytest.mark.parametrize("opu", ["uT","uV","dnuT","dnuV"])
@pytest.mark.parametrize("opv", ["vT","vV","dnvT","dnvV"])
@pytest.mark.parametrize("func", [lambda x, y: x**2 + y**2, lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)])
@pytest.mark.parametrize("ordre", [1,2,3])
def test_assemble_skeleton_9(ordre, opu, opv, func):
    A_new = assemble_skeleton_par_element(mesh, ordre, func,opu,opv,"DG")
    A_old = assemble_skeleton_par_element_old(mesh, ordre, func,opu,opv,"DG")
    assert A_new.is_equal(A_old, tol=1e-8), "problème d'assemblage de la matrice de squelette (jump-jump)"



@pytest.mark.parametrize("ordre", [1,2,4])
def test_assemble_skeleton_10(ordre):

    opuT = "(M.n)uT"
    opvT = "(M.n)vT"
    opuV = "(M.n)uV"
    opvV = "(M.n)vV"
    Mx = lambda x, y: x**2+y**2
    My = lambda x, y: x+y  # Vector field M=(1,0)
    M = lambda x, y: np.array([Mx(x,y), My(x,y)])    
    ATT = assemble_skeleton_par_element(mesh, ordre, M,opuT,opvT,"DG")
    ATV = assemble_skeleton_par_element(mesh, ordre, M,opuT,opvV,"DG")
    AVT = assemble_skeleton_par_element(mesh, ordre, M,opuV,opvT,"DG")
    AVV = assemble_skeleton_par_element(mesh, ordre, M,opuV,opvV,"DG")
    A = COOMatrix(ATT.shape[0], ATT.shape[1], ATT.l + ATV.l + AVT.l + AVV.l)
    A += ATT
    A += ATV
    A += AVT
    A += AVV
    assert A.is_positive(), "A n est pas positive"

@pytest.mark.parametrize("ordre", [1,2,3])
def test_assemble_skeleton_11(ordre):

    opuT = "(M.n)uT"
    opvT = "(M.n)vT"
    opuV = "(M.n)uV"
    opvV = "(M.n)vV"
    Mx = lambda x, y: x**2+y**2
    My = lambda x, y: x+y  # Vector field M=(1,0)
    M = lambda x, y: np.array([Mx(x,y), My(x,y)])    
    ATT = assemble_skeleton_par_element(mesh, ordre, M,opuT,opvT,"DG")
    AVV = assemble_skeleton_par_element(mesh, ordre, M,opuV,opvV,"DG")
    assert ATT.is_equal(AVV, tol=1e-8), "A n est pas positive"

@pytest.mark.parametrize("ordre", [1,2,3])
def test_assemble_skeleton_12(ordre):
    opuT = "(M.n)uT"
    opvT = "vT"
    opuV = "(M.n)uV"
    opvV = "vV"
    Mx = lambda x, y: x**2+y**2
    My = lambda x, y: x+y  # Vector field M=(1,0)
    M = lambda x, y: np.array([Mx(x,y), My(x,y)])    
    ATT = assemble_skeleton_par_element(mesh, ordre, M,opuT,opvT,"DG")
    AVV = assemble_skeleton_par_element(mesh, ordre, M,opuV,opvV,"DG")
    assert ATT.is_equal(AVV, tol=1e-8), "A n est pas positive"
