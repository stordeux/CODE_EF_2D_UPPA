from mes_packages import *
import numpy as np
from scipy.sparse.linalg import eigs
import pytest


from mes_packages.methode_DG import build_dof_coordinates_DG, build_jump_matrix_DG, build_masse_frontiere_elt_DG, build_masse_DG, build_mixte_DG, build_nodal_vector_DG
from mes_packages.sparse import COOMatrix
from mes_packages import (
    build_masse_ref_1D, 
    build_matrice_masse_frontière_DG,build_jump_matrix_DG)

def test_loctoglob_DG():
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    # Recuperation de la géométrie du maillage
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]
    for ordre in range(1,5):
        loctoglob_DG, n_glob_DG = build_loctoglob_DG(triangles, ordre)
        print_loctoglob_DG(loctoglob_DG,triangles, ordre,n_glob_DG)


        # Vérification de la formule pour tous les triangles
        print(f"\n=== Vérification ===")
        Nloc = (ordre+1)*(ordre+2)//2
        message = ""

        for ielt in range(len(triangles)):
            for iloc in range(Nloc):
                iglob_formule = Nloc * ielt + iloc
                iglob_table = loctoglob_DG[ielt, iloc]
                if iglob_formule != iglob_table:
                    message += f"Problème : Triangle {ielt}, iloc {iloc} : formule={iglob_formule}, table={iglob_table}\n"

        assert message == "", message

# Test de la fonction
# On calcule le max de f(x,y)=x^2 + y^2 de deux facons differentes.
#
# Calcul sur deux ensembles differents :
# 1) Sur les sommets du maillage (points) : on evalue f sur tous les sommets.
# 2) Sur les DDLs (dof_coords) : on evalue f sur tous les degres de liberte
#    (qui incluent les sommets et les points d interpolation interieurs).
#
# Remarque sur le minimum :
# On ne peut pas comparer le minimum car le centre du cercle (0,0) est present
# dans points (comme point geometrique de construction) sans etre present dans
# la triangulation. On ne peut donc pas faire de test sur le minimum.


def test_vecteur_nodal_DG():
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    # Recuperation de la géométrie du maillage
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]
    ordre = 2
    loctoglob_DG, n_glob_DG = build_loctoglob_DG(triangles, ordre)
    dof_coords_DG = build_dof_coordinates_DG(mesh, ordre)
    f1 = lambda x, y: x**2 + y**2
    U1 = build_nodal_vector_DG(f1, mesh,ordre)
    print("=== Test du vecteur nodal ===\n")
    F1 = points[:,0]**2 + points[:,1]**2
    TEST = np.isclose(F1.max(), U1.max())
    assert TEST, f"Problème : max(F1)={F1.max()} != max(U1)={U1.max()}"

def test_build_masse_mixte_globale_DG():
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    # Recuperation de la géométrie du maillage
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]
    ordre = 2
    # Declaration d'une matrice sparse pour stocker la matrice de masse globale
    Nloc = (ordre + 1) * (ordre + 2) // 2
    Nglob = Nloc * len(triangles)
    dof_coords_DG=build_dof_coordinates_DG(mesh, ordre)
    masse_globale_DG = build_masse_DG(mesh, ordre)    
    # Construction du vecteur nodal pour f(x,y) = 1
    U_const = build_nodal_vector_DG(lambda x, y: 1.0, mesh, ordre)
    # Construction du vecteur nodal pour f(x,y) = x
    U_x = build_nodal_vector_DG(lambda x, y: x, mesh, ordre)
    # Construction du vecteur nodal pour f(x,y) = y
    U_y = build_nodal_vector_DG(lambda x, y: y, mesh, ordre)
    # Construction du vecteur nodal pour f(x,y) = x*y
    U_xy = build_nodal_vector_DG(lambda x, y: x*y, mesh, ordre)
    # Aire analytique du domaine (carre - disque)
    aire_analytique = 0.3**2 - np.pi * 0.1**2
    aire_numerique = masse_globale_DG.sesquilinear_form(U_const, U_const)
    assert abs(aire_numerique - aire_analytique) < 1e-3, "Aire numerique incorrecte"

    # Centre de gravite (symetrie -> integrales de x et y nulles)
    cx = masse_globale_DG.sesquilinear_form(U_x, U_const)
    cy = masse_globale_DG.sesquilinear_form(U_y, U_const)
    assert abs(cx) < 1e-3, "Centre de gravite en x incorrect"
    assert abs(cy) < 1e-3, "Centre de gravite en y incorrect"
    cxy = masse_globale_DG.sesquilinear_form(U_y, U_x)
    assert abs(cxy) < 1e-3, "Centre de gravite en xy incorrect"
    assert np.isclose(masse_globale_DG.sesquilinear_form(U_y, U_x),masse_globale_DG.sesquilinear_form(U_xy,U_const))
    assert masse_globale_DG.is_symmetric(), "La matrice de masse globale doit etre symetrique"                  

    # Construction des matrices
    Kx_globale, Ky_globale = build_mixte_DG(mesh, ordre)
 
    tol = 1e-3

    # Kx / Ky sur constantes
    kx_const = Kx_globale.sesquilinear_form(U_const, U_const)
    ky_const = Ky_globale.sesquilinear_form(U_const, U_const)
    assert abs(kx_const) < tol, "Kx(1,1) doit etre proche de 0"
    assert abs(ky_const) < tol, "Ky(1,1) doit etre proche de 0"

    # Kx
    kx_1x = Kx_globale.sesquilinear_form(U_const, U_x)
    assert abs(kx_1x-aire_analytique) < tol, "Kx(1,x) doit etre proche de 0"

    kx_y1 = Kx_globale.sesquilinear_form(U_y, U_const)
    assert abs(kx_y1) < tol, "Kx(y,1) doit etre proche de 0"

    kx_x1 = Kx_globale.sesquilinear_form(U_x, U_const)
    assert abs(kx_x1) < tol, "Kx(x,1) doit etre proche de l aire"

    kx_xyx = Kx_globale.sesquilinear_form(U_xy, U_x)
    assert abs(kx_xyx) < tol, "Kx(xy,x) doit etre proche de 0"

    # Ky
    ky_1x = Ky_globale.sesquilinear_form(U_const, U_x)
    assert abs(ky_1x) < tol, "Ky(1,x) doit etre proche de 0"

    ky_x1 = Ky_globale.sesquilinear_form(U_x, U_const)
    assert abs(ky_x1) < tol, "Ky(x,1) doit etre proche de 0"

    ky_y1 = Ky_globale.sesquilinear_form(U_y, U_const)
    assert abs(ky_y1) < tol, "Ky(y,1) doit etre proche de l aire"

    ky_1y = Ky_globale.sesquilinear_form(U_const, U_y)
    assert abs(ky_1y-aire_analytique) < tol, "Ky(1,y) doit etre proche de 0"

    ky_xy1 = Ky_globale.sesquilinear_form(U_xy, U_const)
    assert abs(ky_xy1) < tol, "Ky(xy,1) doit etre proche de 0"

    ky_xyy = Ky_globale.sesquilinear_form(U_xy, U_y)
    assert abs(ky_xyy) < tol, "Ky(xy,y) doit etre proche de 0"
    f_x2 = lambda x, y: x**2
    U_x2 = build_nodal_vector_DG(f_x2, mesh, ordre)
    TEST= np.isclose(Kx_globale.sesquilinear_form(U_x2, U_x),masse_globale_DG.sesquilinear_form(U_x2, U_const))
    assert TEST, "Kx(x^2,x) doit etre proche de M(x^2,1)"
    f_y2 = lambda x, y: y**2
    U_y2 = build_nodal_vector_DG(f_y2, mesh, ordre)
    TEST= np.isclose(Ky_globale.sesquilinear_form(U_y2, U_y),masse_globale_DG.sesquilinear_form(U_y2, U_const))
    assert TEST, "Ky(y^2,y) doit etre proche de M(y^2,1)"

def test_build_masse_frontiere_elt_DG():
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    # Recuperation de la géométrie du maillage
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]
    ordre = 2
    loctoglob_DG,_ = build_loctoglob_DG(triangles, ordre)
    # Declaration d'une matrice sparse pour stocker la matrice de masse globale
    Nloc = (ordre + 1) * (ordre + 2) // 2
    # Test de la matrice locale de face
    ielt = 5
    iface = 2
    # Construction de la matrice de masse de frontière pour la face iface de l'élément ielt
    M_ref_1D = build_masse_ref_1D(ordre)
    MF = build_masse_frontiere_elt_DG(ielt, iface, ordre, loctoglob_DG, points, triangles, M_ref_1D)
    # Récupération des sommets du triangle
    pt1, pt2, pt3 = triangles[ielt]
    A0 = points[pt1]
    A1 = points[pt2]
    A2 = points[pt3]

    # Identification de la face
    if iface == 0:
        P_debut, P_fin = A1, A2
    elif iface == 1:
        P_debut, P_fin = A2, A0
    else:
        P_debut, P_fin = A0, A1
    longueur = np.linalg.norm(P_fin - P_debut)
    U_const = build_nodal_vector_DG(lambda x, y: 1.0, mesh, ordre)
    val1 = MF.sesquilinear_form(U_const, U_const)
    assert np.isclose(val1,longueur), f"MF(1,1) doit etre proche de la longueur de la face ({val1} vs {longueur})"


def test_matrice_frontiere_exterieure_DG():
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.001)
    ordre = 2
    M_Gamma = build_matrice_masse_frontière_DG(mesh,ordre)
    U1_DG = build_nodal_vector_DG(lambda x, y: 1, mesh,ordre)
    val = M_Gamma.sesquilinear_form(U1_DG,U1_DG)  # Doit être égal au Perimetre de la frontière du domaine
    attendu = 0.3*4 + 2*np.pi*0.1
    assert abs(val-attendu)<1e-3, f"M_Gamma(1,1) doit etre proche du perimetre de la frontière du domaine ({val} vs {attendu})"

def test_build_jump_matrix_DG():
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 2
    MAT_saut = build_jump_matrix_DG(mesh, ordre, verbose=True)

    
    f1 = build_nodal_vector_DG(lambda x, y: x * y, mesh, ordre)
    f2 = build_nodal_vector_DG(lambda x, y: x * y, mesh, ordre)

    U=MAT_saut@f1
    # Calcul de la norme L2 du vecteur U
    test =  np.linalg.norm(U)<1e-10
    assert test, "Le produit de la matrice de saut par un vecteur nodal doit être nul"

    print("Test de la continuité à droite :",test) 
    # Test de la continuité à gauche
    U=MAT_saut.produit_gauche(f1)
    # Calcul de la norme L2 du vecteur U
    test = np.linalg.norm(U)<1e-10
    assert test, "Le produit de la matrice de saut par un vecteur nodal doit être nul"


    U=MAT_saut@f2
    # Calcul de la norme L2 du vecteur U
    test =  np.linalg.norm(U)<1e-10
    assert test, "Le produit de la matrice de saut par un vecteur nodal doit être nul"
    
    print("Test de la continuité à droite :",test) 
    # Test de la continuité à gauche
    U=MAT_saut.produit_gauche(f2)
    # Calcul de la norme L2 du vecteur U
    test = np.linalg.norm(U)<1e-10
    assert test, "Le produit de la matrice de saut par un vecteur nodal doit être nul"

    assert MAT_saut.is_symmetric(), "La matrice de saut doit être symétrique"

    MAT_saut.create_COO()

    # Plus grande valeur propre (en module)
    vp_max =  np.array([-1])
    vp_min =  np.array([-1])
    try:
        vp_max, vec_max = eigs(MAT_saut.coo, k=1, which='LM')[:2]  # LM = Largest Magnitude
        vp_min, vec_min = eigs(MAT_saut.coo, k=1, which='SM')[:2]  # SM = Smallest Magnitude
    except Exception as exc:
        raise AssertionError(f"Calcul des valeurs propres échoué: {exc}")
    vp_max_val = vp_max[0].real
    vp_min_val = vp_min[0].real
    assert vp_max_val > 1e-6, f"La plus grande valeur propre en module doit être positive de 0 (vp_max={vp_max})"
    assert np.isclose(vp_min_val,0), f"La plus petite valeur propre en module doit être positive de 0 (vp_min={vp_min})"
    

def test_masse_volumique_DG():

    mesh = create_mesh_circle_in_square(0.1, 0.3, 0.05)
    ordre = 4

    # --- matrice de masse DG (nouvelle version optimisée) ---
    M_DG = build_masse_DG(mesh, ordre)

    # --- fonctions tests (polynômes => intégration exacte attendue) ---
    func = lambda x, y: x**2 + y**2
    f_1  = lambda x, y: 1.0
    f_x  = lambda x, y: x
    f_y  = lambda x, y: y

    # --- vecteurs nodaux DG ---
    V   = build_nodal_vector_DG(func, mesh, ordre)
    V_1 = build_nodal_vector_DG(f_1, mesh, ordre)
    V_x = build_nodal_vector_DG(f_x, mesh, ordre)
    V_y = build_nodal_vector_DG(f_y, mesh, ordre)

    # --- produit faible via la masse ---
    val_1 = M_DG.sesquilinear_form(V, V_1)
    val_x = M_DG.sesquilinear_form(V, V_x)
    val_y = M_DG.sesquilinear_form(V, V_y)

    # --- calcul direct avec le terme source DG ---
    F_vol = terme_source_DG(func, mesh, ordre)

    val_1_bis = np.vdot(F_vol, V_1)
    val_x_bis = np.vdot(F_vol, V_x)
    val_y_bis = np.vdot(F_vol, V_y)

    assert np.isclose(val_1, val_1_bis, atol=1e-10), \
        "DG: incohérence masse / terme source pour w=1"

    assert np.isclose(val_x, val_x_bis, atol=1e-10), \
        "DG: incohérence masse / terme source pour w=x"

    assert np.isclose(val_y, val_y_bis, atol=1e-10), \
        "DG: incohérence masse / terme source pour w=y"
    

def test_masse_variable_DG():
    mesh = create_mesh_circle_in_square(0.1, 0.3, 0.05)
    ordre = 4
    func = lambda x, y: x**2 + y**2
    fx = lambda x, y: x
    fy = lambda x, y: y
    fxy = lambda x, y: x*y

    # --- matrice de masse variable DG ---
    M_var_DG = build_masse_variable_DG(func,mesh, ordre)
    M_DG = build_masse_DG(mesh, ordre)
    # --- fonctions tests (polynômes => intégration exacte attendue) ---

    # --- vecteurs nodaux DG ---
    Vx   = build_nodal_vector_DG(fx,  mesh, ordre)
    Vxy  = build_nodal_vector_DG(fxy, mesh, ordre)
    Vy   = build_nodal_vector_DG(fy,  mesh, ordre)
    Vx2y2   = build_nodal_vector_DG(func, mesh, ordre)
    # --- produit faible via la masse variable ---
    val_var = M_var_DG.sesquilinear_form(Vx, Vy)
    val = M_DG.sesquilinear_form(Vx2y2, Vxy)
    assert np.isclose(val_var, val, atol=1e-11), "DG: incohérence masse variable / masse standard pour w=x et v=y"

@pytest.mark.parametrize("ordre", [4,5])
def test_mixte_xy_DG(ordre):

    mesh = create_mesh_circle_in_square(0.1, 0.3, 0.05)

    # coefficient variable
    rho = lambda x, y: x**2 + y**2

    # fonctions tests
    f1  = lambda x, y: 1.0
    fx  = lambda x, y: x
    fy  = lambda x, y: y
    fxy = lambda x, y: x*y
    fx2 = lambda x, y: x**2
    fy2 = lambda x, y: y**2

    funs = [f1, fx, fy, fxy, fx2, fy2]

    # matrices
    Kxvar, Kyvar = build_mixte_variable_DG(rho, mesh, ordre)
    Kx, Ky       = build_mixte_DG(mesh, ordre)

    # vecteur rho interpolé
    Urho = build_nodal_vector_DG(rho, mesh, ordre)

    for fu in funs:
        U = build_nodal_vector_DG(fu, mesh, ordre)

        for fv in funs:
            V = build_nodal_vector_DG(fv, mesh, ordre)

            Vrho = build_nodal_vector_DG(lambda x,y: rho(x,y)*fv(x,y), mesh, ordre)

            # --- direction x ---
            val_var = Kxvar.sesquilinear_form(V, U)
            val_ref = Kx.sesquilinear_form(Vrho, U)

            assert np.isclose(val_var, val_ref, atol=1e-10), \
                f"Mismatch in x-direction for u={fu.__name__ if hasattr(fu,'__name__') else fu}, v={fv.__name__ if hasattr(fv,'__name__') else fv}"

            # --- direction y ---
            val_var = Kyvar.sesquilinear_form(V, U)
            val_ref = Ky.sesquilinear_form(Vrho, U)

            assert np.isclose(val_var, val_ref, atol=1e-10), \
                f"Mismatch in y-direction for u={fu.__name__ if hasattr(fu,'__name__') else fu}, v={fv.__name__ if hasattr(fv,'__name__') else fv}"

