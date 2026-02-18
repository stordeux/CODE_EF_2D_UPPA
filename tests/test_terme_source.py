import numpy as np
from mes_packages import *

def test_assemble_rhs_volume_CG_1():
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 5

    f1 = lambda x, y: 1
    Fx = lambda x, y: x
    fy = lambda x, y: y
    fxy= lambda x, y: x*y
    fx2= lambda x, y: x**2
    fy2= lambda x, y: y**2
    V1 = build_nodal_vector_CG(f1,mesh, ordre)
    Vx = build_nodal_vector_CG(Fx,mesh, ordre)
    Vy = build_nodal_vector_CG(fy,mesh, ordre)
    Vxy = build_nodal_vector_CG(fxy,mesh, ordre)
    Vx2 = build_nodal_vector_CG(fx2,mesh, ordre)
    Vy2 = build_nodal_vector_CG(fy2,mesh, ordre)

    func = lambda x, y: x**2+y**2
    F = assemble_rhs_volume(mesh, ordre, func, operatorv="v", methode="CG")
    Mat = assemble_volume(mesh, ordre, func, "u", "v", methode="CG")
    val1 = np.vdot(F, Vx)
    val2 = Mat.sesquilinear_form(V1, Vx)
    assert np.isclose(val1, val2, atol=1e-12), "problème d'assemblage du terme source volumique CG (v)"
    val1 = np.vdot(F, Vy)
    val2 = Mat.sesquilinear_form(Vy, V1)
    assert np.isclose(val1, val2, atol=1e-12), "problème d'assemblage du terme source volumique CG (v)"

def test_assemble_rhs_volume_CG_2():
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 5

    f1 = lambda x, y: 1
    Fx = lambda x, y: x
    fy = lambda x, y: y
    fxy= lambda x, y: x*y
    fx2= lambda x, y: x**2
    fy2= lambda x, y: y**2
    V1 = build_nodal_vector_CG(f1,mesh, ordre)
    Vx = build_nodal_vector_CG(Fx,mesh, ordre)
    Vy = build_nodal_vector_CG(fy,mesh, ordre)
    Vxy = build_nodal_vector_CG(fxy,mesh, ordre)
    Vx2 = build_nodal_vector_CG(fx2,mesh, ordre)
    Vy2 = build_nodal_vector_CG(fy2,mesh, ordre)

    func = lambda x, y: x**2
    F = assemble_rhs_volume(mesh, ordre, func, operatorv="dxv", methode="CG")
    Mat = assemble_volume(mesh, ordre, func, "u", "dxv", methode="CG")
    val1 = np.vdot(F, Vx)
    val2 = Mat.sesquilinear_form(Vx, V1)
    assert np.isclose(val1, val2, atol=1e-12), "problème d'assemblage du terme source volumique CG (v)"
    val1 = np.vdot(F, Vy)
    val2 = Mat.sesquilinear_form(Vy, V1)
    assert np.isclose(val1, val2, atol=1e-12), "problème d'assemblage du terme source volumique CG (v)"

    func = lambda x, y: x**2
    F = assemble_rhs_volume(mesh, ordre, func, operatorv="dyv", methode="CG")
    Mat = assemble_volume(mesh, ordre, func, "u", "dyv", methode="CG")
    val1 = np.vdot(F, Vx)
    val2 = Mat.sesquilinear_form(Vx, V1)
    assert np.isclose(val1, val2, atol=1e-12), "problème d'assemblage du terme source volumique CG (v)"
    val1 = np.vdot(F, Vy)
    val2 = Mat.sesquilinear_form(Vy, V1)
    assert np.isclose(val1, val2, atol=1e-12), "problème d'assemblage du terme source volumique CG (v)"

##################################################
### Comparaison CG et DG #########################
##################################################

def test_assemble_rhs_volume_CG_vs_DG_consistency():
    """
    Vérifie que l'intégrale assemblée est identique en CG et DG
    lorsqu'on teste avec une fonction reconstruite identique.
    """

    mesh = create_mesh_circle_in_square(0.1, 0.3, 0.025)
    ordre = 4

    func = lambda x, y: x**2 + y**2
    testf = lambda x, y: x + y

    Vcg = build_nodal_vector_CG(testf, mesh, ordre)
    Vdg = build_nodal_vector_DG(testf, mesh, ordre)

    Fcg = assemble_rhs_volume(mesh, ordre, func, operatorv="v", methode="CG")
    Fdg = assemble_rhs_volume(mesh, ordre, func, operatorv="v", methode="DG")

    val_cg = np.vdot(Fcg, Vcg)
    val_dg = np.vdot(Fdg, Vdg)

    assert np.isclose(val_cg, val_dg, atol=1e-12), \
        "Incohérence CG/DG dans l’assemblage RHS volumique"

    