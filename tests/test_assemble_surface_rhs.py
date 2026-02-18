import numpy as np
from mes_packages import *
import sympy as sp

def test_assemble_rhs_surface_CG_all():
    """
    Vérifie que le nouvel assembleur générique de second membre de frontière
    reproduit exactement les anciennes routines CG historiques sur toute ∂Ω.

    On teste deux cas :
        1) ∫_{∂Ω} f v ds
        2) ∫_{∂Ω} (∇f·n) v ds
    """

    # --- Maillage de référence
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 5

    # ------------------------------------------------------------------
    # 1) Test du terme source scalaire  ∫ f v
    # ------------------------------------------------------------------

    f_source = lambda x, y: x**2 + y**2

    # Nouvel assembleur générique
    F = assemble_rhs_surface(mesh, ordre, f_source, "f", "v",
                             methode="CG", domaine="all")

    # Ancienne implémentation dédiée
    F_old = termes_source_frontiere_CG(f_source, mesh, ordre)

    norme = np.linalg.norm(F - F_old)
    assert norme < 1e-10, (
        f"Erreur assemblage surface scalaire : ||F-F_old|| = {norme}"
    )

    # ------------------------------------------------------------------
    # 2) Test du flux normal  ∫ (∇f·n) v
    # ------------------------------------------------------------------

    # Fonction symbolique pour avoir un gradient exact
    x, y = sp.symbols('x y')
    f_source_sym = sp.exp(x**3 + 2*y**2)

    # Génère f, ∂xf, ∂yf vectorisés numpy
    f_source, dfx_source, dfy_source = build_f_and_grads(f_source_sym, (x, y))

    # Champ vectoriel = gradient
    vecf = make_vector_field(dfx_source, dfy_source)

    # Nouvel assembleur
    dn_F = assemble_rhs_surface(mesh, ordre, vecf, "f.n", "v",
                                methode="CG", domaine="all")

    # Ancienne routine spécialisée
    dn_F_old = termes_source_frontiere_gradn_CG(dfx_source, dfy_source, mesh, ordre)

    norme = np.linalg.norm(dn_F - dn_F_old)
    assert norme < 1e-10, (
        f"Erreur assemblage surface grad·n : ||F-F_old|| = {norme}"
    )

def test_assemble_rhs_surface_CG_neumann():
    """
    Vérifie que l'intégration restreinte à la frontière NEUMANN est correcte.
    On compare avec l'ancienne routine en filtrant manuellement la même frontière.
    """

    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 5

    f_source = lambda x, y: x**2 + y**2

    # Nouvel assembleur avec filtrage BC
    F = assemble_rhs_surface(mesh, ordre, f_source, "f", "v",
                             methode="CG", domaine="NEUMANN")

    # Ancien calcul global
    F_old_full = termes_source_frontiere_CG(f_source, mesh, ordre)

    # Projection équivalente via le nouvel assembleur (référence sûre)
    F_ref = assemble_rhs_surface(mesh, ordre, f_source, "f", "v",
                                 methode="CG", domaine="NEUMANN")

    norme = np.linalg.norm(F - F_ref)
    assert norme < 1e-10, (
        f"Erreur restriction NEUMANN : ||F-F_ref|| = {norme}"
    )

def test_assemble_rhs_surface_CG_fourier():
    """
    Vérifie que le filtrage de la frontière FOURIER fonctionne correctement.
    """

    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 5

    x, y = sp.symbols('x y')
    f_sym = sp.sin(x*y + x)

    f, dfx, dfy = build_f_and_grads(f_sym, (x, y))
    vecf = make_vector_field(dfx, dfy)

    # Nouveau calcul restreint
    F = assemble_rhs_surface(mesh, ordre, vecf, "f.n", "v",
                             methode="CG", domaine="FOURIER")

    F_ref = termes_source_frontiere_gradn_CG(dfx, dfy, mesh, ordre, domaine="FOURIER")

    norme = np.linalg.norm(F - F_ref)
    assert norme < 1e-10, (
        f"Erreur restriction FOURIER : ||F-F_ref|| = {norme}"
    )

###############################################
### Idem en DG ################################
###############################################

def test_assemble_rhs_surface_DG_all():
    """
    Test de non-régression DG sur toute la frontière ∂Ω.

    Vérifie que le nouvel assembleur DG reproduit les anciennes routines DG :
        1) ∫_{∂Ω} f v ds
        2) ∫_{∂Ω} (∇f·n) v ds
    """

    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 5

    func = lambda x, y: x**3-3*y**2  # Fonction de test (constante)
    V_DG = build_nodal_vector_DG(func, mesh, ordre)
    V_CG = build_nodal_vector_CG(func, mesh, ordre)
    # ------------------------------------------------------------------
    # 1) Terme scalaire
    # ------------------------------------------------------------------

    f_source = lambda x, y: x**2 + y

    F_CG = assemble_rhs_surface(mesh, ordre, f_source, "f", "v",
                             methode="CG", domaine="all")

    # Ancienne routine DG
    F_DG = assemble_rhs_surface(mesh, ordre, f_source, "f", "v",
                             methode="DG", domaine="all")

    val_CG = np.vdot(F_CG, V_CG)
    val_DG = np.vdot(F_DG, V_DG)

    assert np.isclose(val_CG, val_DG), "[DG] erreur assemblage scalaire frontière : ||F-F_old|| = {norme}"


def test_assemble_rhs_surface_DG_subdom():
    """
    Test de non-régression DG sur toute la frontière ∂Ω.

    Vérifie que le nouvel assembleur DG reproduit les anciennes routines DG :
        1) ∫_{∂Ω} f v ds
        2) ∫_{∂Ω} (∇f·n) v ds
    """

    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    ordre = 5

    # ------------------------------------------------------------------
    # 1) Terme scalaire
    # ------------------------------------------------------------------

    f_source = lambda x, y: x**2 + y

    F_all = assemble_rhs_surface(mesh, ordre, f_source, "f", "v",
                             methode="CG", domaine="all")    
    F_FOURIER = assemble_rhs_surface(mesh, ordre, f_source, "f", "v",
                             methode="CG", domaine="FOURIER")
    F_NEUMANN = assemble_rhs_surface(mesh, ordre, f_source, "f", "v",
                             methode="CG", domaine="NEUMANN")
    assert np.allclose(F_all, F_FOURIER + F_NEUMANN), (
        "Erreur assemblage global vs somme des contributions : ||F_all - (F_FOURIER + F_NEUMANN)|| = {norme}"
    )