import numpy as np
from mes_packages import *


def test_assemble_volume_equals_build_masse_CG():
    """
    Vérifie que :
        assemble_volume(..., "u","v","CG")
    reproduit exactement la matrice de masse CG classique.
    """

    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 2

    func = lambda x, y: 1.0

    # Assemblage générique
    MAT_1 = assemble_volume(mesh, ordre, func, "u", "v", methode="CG")

    # Assemblage historique spécialisé
    MAT_2 = build_masse_CG(mesh, ordre)

    # Comparaison via COO combiné (même logique que ton script)
    N_glob = MAT_1.shape[0]
    nnz_est = MAT_1.l + MAT_2.l

    MAT = COOMatrix(N_glob, N_glob, nnz_est)
    MAT = MAT + MAT_1.copy()
    MAT = MAT - MAT_2.copy()

    assert MAT.is_zero(tol=1e-12), \
        "assemble_volume (u,v) ne reproduit pas build_masse_CG"
    
def test_assemble_volume_equals_build_rigidite_CG():
    """
    Vérifie que :
        assemble_volume(..., "dxu","dxv","CG")
    reproduit exactement la matrice de rigidité CG classique.
    """

    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 2

    func = lambda x, y: 1.0

    # Assemblage générique
    MAT_1xx = assemble_volume(mesh, ordre, func, "dxu", "dxv", methode="CG")
    MAT_1yy = assemble_volume(mesh, ordre, func, "dyu", "dyv", methode="CG")

    # Assemblage historique spécialisé
    MAT_2 = build_rigidite_CG(mesh, ordre)

    # Comparaison via COO combiné (même logique que ton script)
    N_glob = MAT_1xx.shape[0]
    nnz_est = MAT_1xx.l + MAT_1yy.l + MAT_2.l

    MAT = COOMatrix(N_glob, N_glob, nnz_est)
    MAT = MAT + MAT_1xx
    MAT = MAT + MAT_1yy
    MAT = MAT - MAT_2

    assert MAT.is_zero(tol=1e-12), "problème avec la matrice de rigidité CG"

####################################################################################
#### TEST DG constants #############################################################
####################################################################################

def test_assemble_volume_equals_build_masse_DG():
    """
    Vérifie que :
        assemble_volume(..., "u","v","DG")
    reproduit exactement la matrice de masse DG classique.
    """

    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 2

    func = lambda x, y: 1.0

    # Assemblage générique
    MAT_1 = assemble_volume(mesh, ordre, func, "u", "v", methode="DG")

    # Assemblage historique spécialisé
    MAT_2 = build_masse_DG(mesh, ordre)

    # Comparaison via COO combiné (même logique que ton script)
    N_glob = MAT_1.shape[0]
    nnz_est = MAT_1.l + MAT_2.l

    MAT = COOMatrix(N_glob, N_glob, nnz_est)
    MAT = MAT + MAT_1.copy()
    MAT = MAT - MAT_2.copy()

    assert MAT.is_zero(tol=1e-12), \
        "assemble_volume (u,v) ne reproduit pas build_masse_DG"
    
def test_assemble_volume_equals_build_mixte_DG():
    """
    Vérifie que :
        assemble_volume(..., "dxu","v","DG")
        assemble_volume(..., "dyu","v","DG")
    reproduit exactement la matrice de masse DG classique.
    """

    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 2

    func = lambda x, y: 1.0

    # Assemblage générique
    MAT_1x = assemble_volume(mesh, ordre, func, "dxu", "v", methode="DG")
    MAT_1y = assemble_volume(mesh, ordre, func, "dyu", "v", methode="DG")
    # Assemblage historique spécialisé
    MAT_2x,MAT_2y = build_mixte_DG(mesh, ordre)

    # Comparaison via COO combiné (même logique que ton script)
    N_glob = MAT_1x.shape[0]
    nnz_est = MAT_1x.l + MAT_2x.l

    MATx = COOMatrix(N_glob, N_glob, nnz_est)
    MATx = MATx + MAT_1x.copy()
    MATx = MATx - MAT_2x.copy()

    assert MATx.is_zero(tol=1e-12), \
        "assemble_volume (dxu,v) ne reproduit pas build_mixte_DG"
    

    MATy = COOMatrix(N_glob, N_glob, nnz_est)
    MATy = MATy + MAT_1y.copy()
    MATy = MATy - MAT_2y.copy()

    assert MATy.is_zero(tol=1e-12), \
        "assemble_volume (dyu,v) ne reproduit pas build_mixte_DG"
##############################################################################
### TEST DG VARIABLES ########################################################
##############################################################################
    
def test_masse_variable_DG():
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 2
    func = lambda x, y: x**2+y**2
    MAT_1 = assemble_volume(mesh, ordre, func, "u", "v", methode="DG")
    # Assemblage historique spécialisé
    MAT_2 = build_masse_variable_DG(func,mesh, ordre)
    # Comparaison via COO combiné (même logique que ton script)
    N_glob = MAT_1.shape[0]
    nnz_est = MAT_1.l + MAT_2.l
    MAT = COOMatrix(N_glob, N_glob, nnz_est)
    MAT = MAT + MAT_1
    MAT = MAT - MAT_2
    assert MAT.is_zero(tol=1e-12), "problème d'assemblage de la matrice variable DG"

def test_mixte_variable_DG():
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 2
    func = lambda x, y: x**2+y**2
    MAT_1x = assemble_volume(mesh, ordre, func, "dxu", "v", methode="DG")
    MAT_1y = assemble_volume(mesh, ordre, func, "dyu", "v", methode="DG")
    # Assemblage historique spécialisé
    MAT_2x,MAT_2y = build_mixte_variable_DG(func,mesh, ordre)
    # Comparaison via COO combiné (même logique que ton script)
    N_glob = MAT_1x.shape[0]
    nnz_est = MAT_1x.l + MAT_2x.l
    MATx = COOMatrix(N_glob, N_glob, nnz_est)
    MATx = MATx + MAT_1x
    MATx = MATx - MAT_2x
    assert MATx.is_zero(tol=1e-12), "problème d'assemblage de la matrice variable mixte DG (dxu,v)"
    
    N_glob = MAT_1y.shape[0]
    nnz_est = MAT_1y.l + MAT_2y.l
    MAty = COOMatrix(N_glob, N_glob, nnz_est)
    MAty = MAty + MAT_1y
    MAty = MAty - MAT_2y
    assert MAty.is_zero(tol=1e-12), "problème d'assemblage de la matrice variable mixte DG (dyu,v)"


#########################################################
### Test mixte variable CG ###############################
#########################################################

def test_mixte_variable_CG():
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 2
    func = lambda x, y: (x**2+y**2)/2
    f1 = lambda x,y: 1
    fx = lambda x,y: x
    fy = lambda x,y: y
    V = build_nodal_vector_CG(func,mesh, ordre)
    V1 = build_nodal_vector_CG(f1,mesh, ordre)
    Vx = build_nodal_vector_CG(fx,mesh, ordre)
    Vy = build_nodal_vector_CG(fy,mesh, ordre)
    MAT_1x = assemble_volume(mesh, ordre, func, "dxu", "v", methode="CG")
    MAT_1y = assemble_volume(mesh, ordre, func, "dyu", "v", methode="CG")
    # Assemblage historique spécialisé
    Masse_CG = build_masse_CG(mesh, ordre)

    val1x = Masse_CG.sesquilinear_form(Vx, V)
    val2x = MAT_1x.sesquilinear_form(Vx, Vx)
    assert np.isclose(val1x, val2x, atol=1e-12), "problème d'assemblage de la matrice variable mixte CG (dxu,v)"
    val1y = Masse_CG.sesquilinear_form(Vy, V)
    val2y = MAT_1y.sesquilinear_form(Vy, Vy)
    assert np.isclose(val1y, val2y, atol=1e-12), "problème d'assemblage de la matrice variable mixte CG (dyu,v)"


#def test_positivite():
#    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.1)
#    ordre = 2
#    A = assemble_surface(mesh, ordre, M, "(func.n)u", "(func.n)v")  # (M·n)^2