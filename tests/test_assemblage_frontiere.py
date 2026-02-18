import numpy as np
from mes_packages import *


def test_assemble_surface_equals_build_masse_frontiere_CG():
    """
    Vérifie que l’assemblage frontière générique

        assemble_surface(mesh, ordre, f, "u","v", methode="CG")

    reproduit exactement la routine spécialisée historique

        build_masse_frontiere_CG

    pour une fonction poids constante f = 1.
    """

    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 2

    func = lambda x, y: 1.0

    # Assemblage générique
    MAT_1 = assemble_surface(mesh, ordre, func, "u", "v", methode="CG", domaine="all")

    # Assemblage historique spécialisé
    MAT_2 = build_masse_frontiere_CG(mesh, ordre)

    # Comparaison via COO combiné
    N_glob = MAT_1.shape[0]
    nnz_est = MAT_1.l + MAT_2.l

    MAT = COOMatrix(N_glob, N_glob, nnz_est)
    MAT = MAT + MAT_1.copy()
    MAT = MAT - MAT_2.copy()

    assert MAT.is_zero(tol=1e-6), \
        "assemble_surface CG (masse frontière constante) diffère de build_masse_frontiere_CG"


def test_surface_matrix_is_symmetric_CG():
    """
    Vérifie que la matrice de masse frontière CG est symétrique
    pour un poids constant puis polynomial.
    Cela correspond à la symétrie de la forme bilinéaire :

        ∫_{∂Ω} f u v.
    """
    mesh = create_mesh_circle_in_square(0.1, 0.3, 0.05)
    ordre = 3

    A = assemble_surface(mesh, ordre, lambda x, y: 1, "u", "v", methode="CG")
    assert A.is_symmetric(), "La matrice frontière CG doit être symétrique pour f constant"

    A = assemble_surface(mesh, ordre, lambda x, y: x**2 + y**3, "u", "v", methode="CG")
    assert A.is_symmetric(), "La matrice frontière CG doit être symétrique pour f polynomial"


def test_assemble_surface_FOURIER():
    """
    Vérifie que l’assemblage frontière générique restreint au domaine FOURIER
    coïncide avec la routine spécialisée build_masse_frontiere_CG(...,'FOURIER').
    """

    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 2

    func = lambda x, y: 1

    MAT_1 = assemble_surface(mesh, ordre, func, "u", "v", methode="CG", domaine="FOURIER")
    MAT_2 = build_masse_frontiere_CG(mesh, ordre, "FOURIER")

    N_glob = MAT_1.shape[0]
    nnz_est = MAT_1.l + MAT_2.l

    MAT = COOMatrix(N_glob, N_glob, nnz_est)
    MAT = MAT + MAT_1.copy()
    MAT = MAT - MAT_2.copy()

    assert MAT.is_zero(tol=1e-6), \
        "assemble_surface CG (FOURIER) diffère de build_masse_frontiere_CG"


def test_assemble_surface_NEUMANN():
    """
    Vérifie que l’assemblage frontière générique restreint au domaine NEUMANN
    coïncide avec la routine spécialisée build_masse_frontiere_CG(...,'NEUMANN').
    """

    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 2

    func = lambda x, y: 1

    MAT_1 = assemble_surface(mesh, ordre, func, "u", "v", methode="CG", domaine="NEUMANN")
    MAT_2 = build_masse_frontiere_CG(mesh, ordre, "NEUMANN")

    N_glob = MAT_1.shape[0]
    nnz_est = MAT_1.l + MAT_2.l

    MAT = COOMatrix(N_glob, N_glob, nnz_est)
    MAT = MAT + MAT_1.copy()
    MAT = MAT - MAT_2.copy()

    assert MAT.is_zero(tol=1e-6), \
        "assemble_surface CG (NEUMANN) diffère de build_masse_frontiere_CG"


def test_surface_matrix_is_symmetric_DG():
    """
    Vérifie la symétrie de la matrice de masse frontière en DG.
    Même en DG, la forme locale

        ∫_{∂K ∩ ∂Ω} f u v

    reste symétrique élément par élément.
    """
    mesh = create_mesh_circle_in_square(0.1, 0.3, 0.05)
    ordre = 3

    A = assemble_surface(mesh, ordre, lambda x, y: 1, "u", "v", methode="DG")
    assert A.is_symmetric(), "La matrice frontière DG doit être symétrique pour f constant"

    A = assemble_surface(mesh, ordre, lambda x, y: x**2 + y**3, "u", "v", methode="DG")
    assert A.is_symmetric(), "La matrice frontière DG doit être symétrique pour f polynomial"


def test_surface_matrix_variable_DG():
    """
    Teste la cohérence algébrique de la masse frontière DG avec poids variable.

    On vérifie des identités du type :

        ⟨M(f)·1,1⟩ = ⟨M(1)·f,1⟩

    et plusieurs cas polynomiaux pour valider l’insertion correcte
    du poids f(x,y) dans l’intégration de frontière.
    """

    func = lambda x, y: 1 + x**2 + 2*y**2
    f1 = lambda x, y: 1
    fx = lambda x, y: x
    fy = lambda x, y: y
    fxy = lambda x, y: x*y
    fx2 = lambda x, y: x**2
    fy2 = lambda x, y: y**2

    mesh = create_mesh_circle_in_square(0.1, 0.3, 0.05)
    ordre = 3

    V_func = build_nodal_vector_DG(func, mesh, ordre)
    V_1 = build_nodal_vector_DG(f1, mesh, ordre)
    V_x = build_nodal_vector_DG(fx, mesh, ordre)
    V_y = build_nodal_vector_DG(fy, mesh, ordre)
    V_xy = build_nodal_vector_DG(fxy, mesh, ordre)
    V_x2 = build_nodal_vector_DG(fx2, mesh, ordre)

    Mat_func = assemble_surface(mesh, ordre, func, "u", "v", methode="DG", domaine="all")
    Mat_f1 = assemble_surface(mesh, ordre, f1, "u", "v", methode="DG", domaine="all")
    Mat_fx = assemble_surface(mesh, ordre, fx, "u", "v", methode="DG", domaine="all")

    val1 = Mat_func.sesquilinear_form(V_1, V_1)
    val2 = Mat_f1.sesquilinear_form(V_func, V_1)
    assert np.isclose(val1, val2), \
        "Incohérence entre masse frontière pondérée et masse constante appliquée à f"

    val1 = Mat_fx.sesquilinear_form(V_x, V_x)
    val2 = Mat_f1.sesquilinear_form(V_x2, V_x)
    assert np.isclose(val1, val2), \
        "Échec du test de cohérence pour le poids f(x,y)=x"

    val1 = Mat_fx.sesquilinear_form(V_y, V_y)
    val2 = Mat_f1.sesquilinear_form(V_xy, V_y)
    assert np.isclose(val1, val2), \
        "Échec du test de cohérence pour le poids croisé sur la frontière"
