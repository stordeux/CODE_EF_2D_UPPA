import numpy as np
import pytest
from mes_packages import *

@pytest.mark.parametrize("ordre", [1,2,3,4])
def test_assemble_skeleton_1(ordre):
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    func = lambda x, y: (x**2+y**2)/2
    f1 = lambda x,y: 1
    fx = lambda x,y: x
    fy = lambda x,y: y
    V = build_nodal_vector_CG(func,mesh, ordre)
    V1 = build_nodal_vector_CG(f1,mesh, ordre)
    Vx = build_nodal_vector_CG(fx,mesh, ordre)
    Vy = build_nodal_vector_CG(fy,mesh, ordre)
    # Nouveau : forme jump-jump
    A_new = assemble_skeleton_par_element_old(mesh, ordre, coef=1.0,
                              operatoru="sautu", operatorv="vT",
                              methode="DG")
    # Ancien : build_jump_matrix_DG
    A_old = build_jump_matrix_DG(mesh, ordre, verbose=False)
    MAT= COOMatrix(A_new.shape[0], A_new.shape[1], A_new.l + A_old.l)
    MAT += A_new
    MAT -= A_old
    assert MAT.is_zero(tol=1e-12), "problème d'assemblage de la matrice de squelette (jump-jump)"

@pytest.mark.parametrize("ordre", [1,2,3,4])
def test_assemble_skeleton_2(ordre):
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    func = lambda x, y: (x**2+y**2)/2
    f1 = lambda x,y: 1
    fx = lambda x,y: x
    fy = lambda x,y: y
    V = build_nodal_vector_CG(func,mesh, ordre)
    V1 = build_nodal_vector_CG(f1,mesh, ordre)
    Vx = build_nodal_vector_CG(fx,mesh, ordre)
    Vy = build_nodal_vector_CG(fy,mesh, ordre)
    # Nouveau : forme jump-jump
    A_new = assemble_skeleton_par_element_old(mesh, ordre, coef=1.0,
                              operatoru="sautDGu", operatorv="vTnT",
                              methode="DG")
    # Ancien : build_jump_matrix_DG
    A_old = build_jump_matrix_DG(mesh, ordre, verbose=False)
    MAT= COOMatrix(A_new.shape[0], A_new.shape[1], A_new.l + A_old.l)
    MAT += A_new
    MAT -= A_old
    assert MAT.is_zero(tol=1e-12), "problème d'assemblage de la matrice de squelette (jump-jump)"
    
@pytest.mark.parametrize("ordre", [1,2,3,4])
def test_assemble_skeleton_par_face_1(ordre):
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    func = lambda x, y: (x**2+y**2)/2
    f1 = lambda x,y: 1
    fx = lambda x,y: x
    fy = lambda x,y: y
    V = build_nodal_vector_CG(func,mesh, ordre)
    V1 = build_nodal_vector_CG(f1,mesh, ordre)
    Vx = build_nodal_vector_CG(fx,mesh, ordre)
    Vy = build_nodal_vector_CG(fy,mesh, ordre)
    # Nouveau : forme jump-jump
    A_new = assemble_skeleton_par_face(mesh, ordre, coef=1.0,
                              operatoru="sautu", operatorv="sautv",
                              methode="DG")
    # Ancien : build_jump_matrix_DG
    A_old = build_jump_matrix_DG(mesh, ordre, verbose=False)
    MAT= COOMatrix(A_new.shape[0], A_new.shape[1], A_new.l + A_old.l)
    MAT +=   A_new
    MAT -= A_old
    assert MAT.is_zero(tol=1e-12), "problème d'assemblage de la matrice de squelette (jump-jump)"
    
@pytest.mark.parametrize("ordre", [1,2,3,4])
def test_assemble_skeleton_par_face_2(ordre):
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    func = lambda x, y: (x**2+y**2)/2
    f1 = lambda x,y: 1
    fx = lambda x,y: x
    fy = lambda x,y: y
    V = build_nodal_vector_CG(func,mesh, ordre)
    V1 = build_nodal_vector_CG(f1,mesh, ordre)
    Vx = build_nodal_vector_CG(fx,mesh, ordre)
    Vy = build_nodal_vector_CG(fy,mesh, ordre)
    # Nouveau : forme jump-jump
    A_new = assemble_skeleton_par_face(mesh, ordre, coef=1.0,
                              operatoru="sautDGu", operatorv="sautDGv",
                              methode="DG")
    # Ancien : build_jump_matrix_DG
    A_old = build_jump_matrix_DG(mesh, ordre, verbose=False)
    MAT= COOMatrix(A_new.shape[0], A_new.shape[1], A_new.l + A_old.l)
    MAT += A_new
    MAT -= A_old
    assert MAT.is_zero(tol=1e-12), "problème d'assemblage de la matrice de squelette (jump-jump)"