import numpy as np
import sympy as sp
from mes_packages import create_mesh_circle_in_square
from mes_packages import build_masse_CG, build_rigidite_CG, build_nodal_vector_CG,build_masse_frontiere_CG, termes_source_frontiere_CG, termes_source_frontiere_gradn_CG
from mes_packages.calcul_symbolique import build_f_and_grads


def test_build_masse_CG():
    mesh = create_mesh_circle_in_square(0.1, 0.3,0.05)
    ordre = 3
    M_CG = build_masse_CG(mesh, ordre, verbose=False)

    U_const = build_nodal_vector_CG(lambda x, y: 1.0, mesh,ordre)
    # Construction du vecteur nodal pour f(x,y) = x
    U_x = build_nodal_vector_CG(lambda x, y: x, mesh,ordre)
    # Construction du vecteur nodal pour f(x,y) = y
    U_y = build_nodal_vector_CG(lambda x, y: y, mesh,ordre)
    # Construction du vecteur nodal pour f(x,y) = x*y
    U_xy = build_nodal_vector_CG(lambda x, y: x*y, mesh,ordre)
    # Calcul de l'aire du domaine
    Aire_analytique = 0.3**2 - np.pi * 0.1**2
    Aire_numerique = M_CG.sesquilinear_form(U_const, U_const)
    assert abs(Aire_numerique - Aire_analytique) < 1e-3, "Aire numerique incorrecte"
    # Test centre de gravité
    assert abs(M_CG.sesquilinear_form(U_x, U_const)) < 1e-3, "Centre de gravité en x incorrect"
    assert abs(M_CG.sesquilinear_form(U_y, U_const)) < 1e-3, "Centre de gravité en y incorrect"
    val1 = M_CG.sesquilinear_form(U_y, U_x)
    val2 = M_CG.sesquilinear_form(U_xy, U_const)
    assert abs(val1-val2) < 1e-6, "Centre de gravité en xy incorrect"
    assert M_CG.is_symmetric(), "La matrice de masse doit être symétrique"
    K_CG = build_rigidite_CG(mesh, ordre, verbose=False)
    val1 = K_CG.sesquilinear_form(U_const, U_const)
    assert abs(val1) < 1e-6, "K(1,1) doit être proche de 0"
    val2 = K_CG.sesquilinear_form(U_const, U_x)
    assert abs(val2) < 1e-6, "K(1,x) doit être proche de 0"
    val3 = K_CG.sesquilinear_form(U_x, U_const)
    assert abs(val3) < 1e-6, "K(x,1) doit être proche de 0"
    val4 = K_CG.sesquilinear_form(U_x, U_x)
    val5 = M_CG.sesquilinear_form(U_const, U_const)
    assert abs(val4-val5) < 1e-6, "K(x,x) doit être proche de l'aire du domaine"
    val6 = K_CG.sesquilinear_form(U_y, U_y)
    assert abs(val6-val5) < 1e-6, "K(y,y) doit être proche de l'aire du domaine"

    M_bord_CG = build_masse_frontiere_CG(mesh, ordre)
    Perimetre = 4 * 0.3 + 2 * np.pi * 0.1
    U1_CG = build_nodal_vector_CG(lambda x, y: 1, mesh,ordre)
    Ux_CG = build_nodal_vector_CG(lambda x, y: x, mesh,ordre)
    Uy_CG = build_nodal_vector_CG(lambda x, y: y, mesh,ordre)
    TEST = abs(M_bord_CG.sesquilinear_form(U1_CG,U1_CG)-Perimetre) < 1e-2   
    assert TEST, "∫_Γ 1 1 ds doit être proche du périmètre du domaine"
    TEST = abs(M_bord_CG.sesquilinear_form(Ux_CG,U1_CG)) < 1e-2
    assert TEST, "∫_Γ x 1 ds doit être proche de 0 (symétrie)"
    TEST = abs(M_bord_CG.sesquilinear_form(Uy_CG,U1_CG)) < 1e-2
    assert TEST, "∫_Γ y 1 ds doit être proche de 0 (symétrie)"


def test_termes_source_frontiere_CG():
    mesh = create_mesh_circle_in_square(0.1, 0.3,0.05)
    ordre =2
    M_bord_CG = build_masse_frontiere_CG(mesh, ordre)
    # Définir la fonction source
    f_source = lambda x, y: 1
    F_CG = termes_source_frontiere_CG(f_source, mesh,ordre)
    # Construire le vecteur nodal du second membre (projection de f_source sur les DDL C0)
    U_fsource_CG = build_nodal_vector_CG(f_source,mesh,ordre)

    # Produit avec la matrice de masse pour obtenir l'intégrale pondérée
    Perimetre = 4 * 0.3 + 2 * np.pi * 0.1
    valeur_integrale1 = M_bord_CG.sesquilinear_form(U_fsource_CG, U_fsource_CG)
    valeur_integrale2 = np.dot(F_CG, U_fsource_CG)

    # Le perimetre est approché car les elements ne sont pas courbes
    TEST = abs(valeur_integrale1 - Perimetre) < 1e-2 and abs(valeur_integrale2 - valeur_integrale1) < 1e-5
    assert TEST, "∫_Γ f_source f_source ds doit être proche du périmètre du domaine"

    f_source = lambda x, y: np.cos(x + y)
    F_CG = termes_source_frontiere_CG(f_source, mesh, ordre)
    # Construire le vecteur nodal du second membre (projection de f_source sur les DDL C0)
    U_fsource_CG = build_nodal_vector_CG(f_source,mesh,ordre)

    # Produit avec la matrice de masse pour obtenir l'intégrale pondérée
    Perimetre = 4 * 0.3 + 2 * np.pi * 0.1
    valeur_integrale1 = M_bord_CG.sesquilinear_form(U_fsource_CG, U_fsource_CG)
    valeur_integrale2 = np.dot(F_CG, U_fsource_CG)

    TEST = abs(valeur_integrale1 - valeur_integrale2) < 1e-6
    assert TEST, "∫_Γ f_source f_source ds doit être proche du périmètre du domaine"






def test_matrice_CG_masse_et_rigidite():
    mesh= create_mesh_circle_in_square(0.1, 0.3,0.05)
    ordre = 2
    f1 = lambda x, y: 1
    fx = lambda x, y: x
    fy = lambda x, y: y
    fx2 = lambda x, y: x**2
    fxy = lambda x, y: x*y
    fy2 = lambda x, y: y**2
    fex1 = lambda x, y: np.exp(x+y)
    fex2 = lambda x, y: np.cos(2*(x+y))
    fex2p = lambda x, y: -2 * np.cos(2*(x+y))
    # Consturction des vecteurs nodaux C0 
    U1_CG = build_nodal_vector_CG(f1, mesh,ordre)
    Ux_CG = build_nodal_vector_CG(fx, mesh,ordre)
    Uy_CG = build_nodal_vector_CG(fy, mesh,ordre)
    Ux2_CG = build_nodal_vector_CG(fx2, mesh,ordre)
    Uxy_CG = build_nodal_vector_CG(fxy, mesh,ordre)
    Uy2_CG = build_nodal_vector_CG(fy2, mesh,ordre)
    Uex1_CG = build_nodal_vector_CG(fex1, mesh,ordre)
    Uex2_CG = build_nodal_vector_CG(fex2, mesh,ordre)
    Uex2p_CG = build_nodal_vector_CG(fex2p, mesh,ordre)
    K_CG = build_rigidite_CG(mesh, ordre, verbose=False)
    M_CG = build_masse_CG(mesh, ordre, verbose=False)
    # On calcule de deux manières différentes des intégrales
    TEST = abs(K_CG.sesquilinear_form(U1_CG, U1_CG)) < 1e-6
    assert TEST, "K(1,1) doit être proche de 0"      # ∫_Ω ∇1 · ∇1 = 0
    TEST = abs(K_CG.sesquilinear_form(Ux_CG, Ux_CG) - M_CG.sesquilinear_form(U1_CG, U1_CG)) < 1e-6
    assert TEST, "K(x,x) doit être proche de M(1,1)"      # ∫_Ω ∇x · ∇x = ∫_Ω 1 · 1 dx dy
    TEST = abs(K_CG.sesquilinear_form(Ux_CG, Uy_CG)) < 1e-6
    assert TEST, "K(x,y) doit être proche de 0"      # ∫_Ω ∇x · ∇y = 0
    TEST = abs(K_CG.sesquilinear_form(Uy_CG, Uy_CG) - M_CG.sesquilinear_form(U1_CG, U1_CG)) < 1e-6
    assert TEST, "K(y,y) doit être proche de M(1,1)"      # ∫_Ω ∇y · ∇y = ∫_Ω 1 · 1 dx dy
    TEST = abs(K_CG.sesquilinear_form(Uy2_CG, Uy2_CG) - 4*M_CG.sesquilinear_form(Uy_CG, Uy_CG)) < 1e-6
    assert TEST, "K(y^2,y^2) doit être proche de 4*K(y,y)"      # ∫_Ω ∇(y^2) · ∇(y^2) = 4∫_Ω y · y dx dy
    TEST = abs(K_CG.sesquilinear_form(Uex1_CG, Uex1_CG) - 2*M_CG.sesquilinear_form(Uex1_CG, Uex1_CG)) < 1e-6
    assert TEST, "K(ex1,ex2) doit être proche de 2*M(ex1,ex2p)"      # ∫_Ω ∇(e^(x+y)) · ∇(e^(x+y)) = 2∫_Ω e^(x+y) e^(x+y) dx dy




def test_non_trivial():
    # Test non trivial
    # On verifie que :
    #   
    #   int_{dOmega} d_n u * v = int_{Omega} (Delta u) * v + int_{Omega} grad u . grad v
    #   
    # pour des fonctions dont on peut calculer le laplacien :
    #   Delta(x^2 - y^2) = 0
    #   Delta(x^3 + 2 y^2) = 6 x + 4
    mesh = create_mesh_circle_in_square(0.1, 0.3,0.05)
    ordre = 3
    K_CG = build_rigidite_CG(mesh, ordre, verbose=False)
    M_CG = build_masse_CG(mesh, ordre, verbose=False)   
    x,y = sp.symbols('x y')
    f_sym = x**3+2*y**2
    f,fx,fy = build_f_and_grads(f_sym, (x,y))
    g = lambda x,y: 6*x+4 
    dnF0 = termes_source_frontiere_gradn_CG(fx, fy, mesh,ordre)
    # Vecteur nodal de f
    U_f_CG = build_nodal_vector_CG(f,mesh,ordre)
    U_g_CG  = build_nodal_vector_CG(g, mesh,ordre) 
    dnF0bis = K_CG@U_f_CG + M_CG@U_g_CG
    assert np.linalg.norm(dnF0 - dnF0bis) < 1e-6, "Le terme source doit être égal à Kf + Mg"
    

