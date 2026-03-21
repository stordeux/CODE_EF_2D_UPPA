import numpy as np
from mes_packages.assemblage_general import *


ZERO = lambda x,y: 0.0
DEUX = lambda x, y: 2.0

def onde_plane_acoustique(theta, kappa):
    kx = kappa * np.cos(theta)
    ky = kappa * np.sin(theta)

    def phase(x, y):
        return np.exp(1j * (kx * x + ky * y))

    func0 = lambda x, y, f=phase: f(x, y)
    func1 = lambda x, y, f=phase: (kx / kappa) * f(x, y)
    func2 = lambda x, y, f=phase: (ky / kappa) * f(x, y)

    return func0, func1, func2

def build_acoustic_matrices(kappa):
    h = lambda x, y: -1.j * kappa
    ZERO = lambda x, y: 0.0
    UN = lambda x, y: 1.0
    F0 = np.array([
        [h, ZERO, ZERO],
        [ZERO, h, ZERO],
        [ZERO, ZERO, h]
    ], dtype=object)

    Fx = np.array([
        [ZERO, UN,   ZERO],
        [UN,   ZERO, ZERO],
        [ZERO, ZERO, ZERO]
    ], dtype=object)

    Fy = np.array([
        [ZERO, ZERO, UN],
        [ZERO, ZERO, ZERO],
        [UN,   ZERO, ZERO]
    ], dtype=object)

    return F0, Fx, Fy

def build_penalty_matrices():
    ZERO = lambda x, y: 0.0
    UN = lambda x, y: 1.0
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

def assemble_frontiere_acoustique(mesh, ordre, kappa,Z=1,formulation="positif"):

    r"""
    Assemble la matrice de frontière associée à la partie coercive du système
    acoustique.

    Cette matrice représente la forme bilinéaire

        Formulation = positif 
    $$
        \int_{\partial\Omega}
        \left(
            Z\;p\;p'
            + 1/Z\;(v\cdot n)(v'\cdot n)
        \right) ds.
    $$
        Formulation = Fplus
    $$
        \int_{\partial\Omega}
        \left(
            1/Z(v.n+Z\;p)(v'.n+Z\;p')
        \right) ds.
    $$
        Formulation = Fmoins
    $$
        \int_{\partial\Omega}
        \left(
            -1/Z(v.n-Z\;p)(v'.n-Z\;p')
        \right) ds.
    $$
        Formulation = reciprocite
    $$
        \int_{\partial\Omega}
        \left(
            p v'.n +  v.n p'
        \right) ds.
    
    Paramètres
    ----------
    mesh : Mesh
        Maillage.
    ordre : int
        Ordre des éléments finis.

    Retour
    ------
    COOMatrix
        Matrice creuse représentant la contribution coercive de frontière.
    """
    from .methode_hyperbolique import assemble_hyperbo
    if np.isclose(Z,0):
        raise ValueError("L'impédance Z doit être non nulle pour éviter les divisions par zéro.")

    _, Fx, Fy = build_acoustic_matrices(kappa)
    func_F_p, func_F_vx, func_F_vy, func_F_cross = build_penalty_matrices()

    if formulation == "positif":
        need_pp = True
        need_pv = False
        need_vv = True
    elif formulation == "Fplus":
        need_pp = True
        need_pv = True
        need_vv = True
    elif formulation == "Fmoins":
        need_pp = True
        need_pv = True
        need_vv = True
    elif formulation == "reciprocite":
        need_pp = False
        need_pv = True
        need_vv = False
    elif formulation == "Dirichlet":
        need_pp = True
        need_pv = False
        need_vv = False
    elif formulation == "Neumann":
        need_pp = False
        need_pv = False
        need_vv = True
    else:
        raise NotImplementedError("Formulation forte non encore implémentée pour la frontière.")
    if need_pp:A_p_TT = assemble_hyperbo(
        mesh, ordre, opu="u", opv="v",
        F=func_F_p, kind="frontiere", methode="DG", domaine="all")
    else:A_p_TT = None
    if formulation == "Dirichlet":
        F_dir_x= np.array([
            [ZERO, ZERO,   ZERO],
            [DEUX,   ZERO, ZERO],
            [ZERO, ZERO, ZERO]], dtype=object)
        A_dir_x = assemble_hyperbo(mesh, ordre, opu="nxu", opv="v",
            F=F_dir_x, kind="frontiere", methode="DG", domaine="all")
        F_dir_y= np.array([
            [ZERO, ZERO,   ZERO],
            [ZERO, ZERO, ZERO],
            [DEUX,   ZERO, ZERO]], dtype=object)
        A_dir_y = assemble_hyperbo(mesh, ordre, opu="nyu", opv="v",
            F=F_dir_y, kind="frontiere", methode="DG", domaine="all") 
    else:
        A_dir_x = None
        A_dir_y = None

    if formulation == "Neumann":
        F_neu_x= np.array([
            [ZERO, DEUX, ZERO],
            [ZERO, ZERO, ZERO],
            [ZERO, ZERO, ZERO]], dtype=object)
        A_neu_x = assemble_hyperbo(mesh, ordre, opu="nxu", opv="v",
            F=F_neu_x, kind="frontiere", methode="DG", domaine="all")
        F_neu_y= np.array([
            [ZERO, ZERO, DEUX],
            [ZERO, ZERO, ZERO],
            [DEUX, ZERO, ZERO]], dtype=object)
        A_neu_y = assemble_hyperbo(mesh, ordre, opu="nyu", opv="v",
            F=F_neu_y, kind="frontiere", methode="DG", domaine="all") 
    else:
        A_neu_x = None
        A_neu_y = None
    if need_vv:
        A_xx_TT = assemble_hyperbo(mesh, ordre, opu="nxu", opv="nxv",
            F=func_F_vx, kind="frontiere", methode="DG", domaine="all")
        A_yy_TT = assemble_hyperbo(mesh, ordre, opu="nyu", opv="nyv",
            F=func_F_vy, kind="frontiere", methode="DG", domaine="all")
        A_yx_TT = assemble_hyperbo(mesh, ordre, opu="nyu", opv="nxv",
            F=func_F_cross, kind="frontiere", methode="DG", domaine="all")
    else:
        A_xx_TT = None
        A_yy_TT = None
        A_yx_TT = None
    if need_pv:
        A_px_TT = assemble_hyperbo(mesh, ordre, opu="nxu", opv="v",
            F=Fx, kind="frontiere", methode="DG", domaine="all")
        A_py_TT = assemble_hyperbo(mesh, ordre, opu="nyu", opv="v",
            F=Fy, kind="frontiere", methode="DG", domaine="all")
    else:
        A_px_TT = None
        A_py_TT = None

    if formulation == "positif":
        blocs = [
            (+Z, A_p_TT),
            (+1./Z, A_xx_TT),
            (+1./Z, A_yy_TT),
            (+1./Z, A_yx_TT),]
    elif formulation == "Fplus":
        blocs = [
            (+Z, A_p_TT),
            (+1./Z, A_xx_TT),
            (+1./Z, A_yy_TT),
            (+1./Z, A_yx_TT),
            (+1, A_px_TT),
            (+1, A_py_TT),]
    elif formulation == "Fmoins":
        blocs = [
            (-Z, A_p_TT),
            (-1./Z, A_xx_TT),
            (-1./Z, A_yy_TT),
            (-1./Z, A_yx_TT),
            (+1, A_px_TT),
            (+1, A_py_TT),]
    elif formulation == "reciprocite":
        blocs = [
            (+1, A_px_TT),
            (+1, A_py_TT),]
    elif formulation == "Dirichlet":
        blocs = [
            (2, A_p_TT),
            (+1,A_dir_x),
            (+1,A_dir_y),]
    elif formulation == "Neumann":
        blocs = [
            (+1,A_neu_x),
            (+1,A_neu_y),
            (2, A_xx_TT),
            (2, A_yy_TT),
            (2, A_yx_TT),]
        
    else:
        raise NotImplementedError("Formulation forte non encore implémentée pour la frontière.")
    Mat_bord = assemble_signed_blocks(blocs)
    return Mat_bord

def assemble_rhs_acoustique(mesh,ordre, func_vec=None, Z=1.0, domaine="all", formulation = "Fplus"):
    """
    Assemble le second membre de frontière consistant associé à

        formulation = "Fplus"
        (1/Z) (v_inc.n + Z p_inc) (v'.n + Z p')

        formulation = "Fmoins"
        (-1/Z) (v_inc.n - Z p_inc) (v'.n - Z p')

        formulation = "positif"
        (1/Z) v_inc.n v'.n + Z p_inc p'


    Paramètres
    ----------
    func_vec : tuple de 3 callables
        func_vec = (p_inc, v_inc_x, v_inc_y)

    Retour
    ------
    rhs : ndarray complexe de taille 3*Nglob
        Ordre des composantes : [p, vx, vy]
    """

    if np.isclose(Z, 0):
       raise ValueError("Z ne doit pas être nul.")

    if func_vec is None:
        func_vec = onde_plane_acoustique(np.pi/3, 2*np.pi/1.3)

    if not isinstance(func_vec, (tuple, list)) or len(func_vec) != 3:
        raise TypeError(
            "func_vec doit être un tuple/list de 3 fonctions : (p_inc, v_inc_x, v_inc_y)."
        )

    p_inc, v_inc_x, v_inc_y = func_vec

    if not (callable(p_inc) and callable(v_inc_x) and callable(v_inc_y)):
        raise TypeError(
            "Les trois composantes de func_vec doivent être callables."
        )

    v_inc = make_vector_field(v_inc_x, v_inc_y)

    if formulation == "Fplus":
        alpha = Z
        gamma = 1/Z
        beta = 1.0
    elif formulation == "Fmoins":
        alpha = -Z
        gamma = -1/Z
        beta = 1.0
    elif formulation == "positif":
        alpha = Z
        gamma = 1/Z
        beta = 0
    else:
        raise NotImplementedError("Formulation non implémentée pour l'instant.")
    # Il y a quatre blocs à assembler pour le second membre de frontière consistant :

    # Bloc 1 : ∫ Z p_inc p' ds
    rhs_p = alpha*assemble_rhs_surface(
            mesh, ordre,f=p_inc,op_f="f",
            op_v="v",methode="DG",domaine=domaine
        )
    # Bloc 2 : ∫ v_inc.n p' ds
    rhs_p+= beta*assemble_rhs_surface(
            mesh, ordre,f=v_inc,op_f="f.n",
            op_v="v",methode="DG",domaine=domaine)

    # Bloc 3 : ∫ p_inc v'.n ds
    # Le bloc 3 se divise en deux parties :
    # ∫ p_inc nx v'_x ds
    # ∫ p_inc ny v'_y ds
    rhs_vx = beta*assemble_rhs_surface(
            mesh, ordre,f=p_inc,op_f="f",
            op_v="nxv",methode="DG",domaine=domaine)
    rhs_vy = beta * assemble_rhs_surface(
            mesh, ordre,f=p_inc,op_f="f",
            op_v="nyv",methode="DG",domaine=domaine)
    # Bloc 4 : ∫ (1/Z) v_inc.n v'.n ds
    # Le bloc 4 se divise en deux parties :
    # ∫ (1/Z) v_inc.n nx v'_x ds
    # ∫ (1/Z) v_inc.n ny v'_y ds
    rhs_vx += gamma * assemble_rhs_surface(
            mesh, ordre,f=v_inc,
            op_f="f.n",op_v="nxv",methode="DG",domaine=domaine)
    rhs_vy += gamma * assemble_rhs_surface(
            mesh, ordre,f=v_inc,op_f="f.n",
            op_v="nyv",methode="DG",domaine=domaine)

    Nglob = rhs_p.shape[0]
    rhs = np.zeros(3 * Nglob, dtype=np.complex128)
    rhs[0:Nglob] = rhs_p
    rhs[Nglob:2*Nglob] = rhs_vx
    rhs[2*Nglob:3*Nglob] = rhs_vy

    return rhs

def assemble_squelette_acoustique(mesh, ordre, kappa,Z=1,formulation="positif"):
    r"""
    Assemble la matrice de squelette associée à la partie coercive du système
    acoustique.

    Cette matrice représente la forme bilinéaire

        Formulation = positif 
    $$
        \sum_{T\in\mathcal{T}}
        \sum_{F\in\mathcal{F}_T\cap\mathcal{F}_{\text{int}}\int_{F} 
        \left( Z (p_T-p_V) p'_T
        + \frac{1}{Z} (n_T \cdot v_T+n_V \cdot v_V)(n_T \cdot v'_T) \right) ds
    $$
        Formulation = Fplus
    $$
        \sum_{T\in\mathcal{T}}
        \sum_{F\in\mathcal{F}_T\cap\mathcal{F}_{\text{int}}\int_{F} 
        \left(    1/Z(v_T.n_T+Z\;p_T)(v'_T.n_T+Z\;p'_T)
                + 1/Z(v_V.n_V+Z\;p_V)(v'_T.n_V+Z\;p'_T) \right) ds
    $$
        Formulation = Fmoins
    $$
        \sum_{T\in\mathcal{T}}  
        \sum_{F\in\mathcal{F}_T\cap\mathcal{F}_{\text{int}}\int_{F}
        \left(   -1/Z(v_T.n_T-Z\;p_T)(v'_T.n_T-Z\;p'_T)
                -1/Z(v_V.n_V-Z\;p_V)(v'_T.n_V-Z\;p'_T) \right) ds
    $$
        Formulation = reciprocite
    $$
        \sum_{T\in\mathcal{T}}
        \sum_{F\in\mathcal{F}_T\cap\mathcal{F}_{\text{int}}\int_{F}
        \left( p_T v'_T.n_T +  v_T.n_T p'_T \right) ds
    $$
    Paramètres
    ----------
    mesh : Mesh
        Maillage.
    ordre : int
        Ordre des éléments finis.

    Retour
    ------
    COOMatrix
        Matrice creuse représentant la contribution du squelette.
    """
    from .methode_hyperbolique import assemble_hyperbo
    if np.isclose(Z,0):
        raise ValueError("L'impédance Z doit être non nulle pour éviter les divisions par zéro.")
    _, Fx, Fy = build_acoustic_matrices(kappa)
    func_F_p, func_F_vx, func_F_vy, func_F_cross = build_penalty_matrices()
    if formulation=="Fplus":
        need_pp=True
        need_vv=True
        need_pv=True
    elif formulation=="Fmoins":
        need_pp=True
        need_vv=True
        need_pv=True
    elif formulation=="positif":
        need_pp=True
        need_vv=True
        need_pv=False
    elif formulation=="reciprocite": 
        need_pp=False
        need_vv=False
        need_pv=True
    else:
        raise NotImplementedError("Formulation non implémentée pour l'instant.")
    if need_pp:
        A_p_TT = assemble_hyperbo(
            mesh, ordre, opu="uT", opv="vT",
            F=func_F_p, kind="squelette_element", methode="DG", domaine="all")
        A_p_TV = assemble_hyperbo(
            mesh, ordre, opu="uT", opv="vV",
            F=func_F_p, kind="squelette_element", methode="DG", domaine="all")
    else:
        A_p_TT=None
        A_p_TV=None
    if need_vv:
        A_xx_TT = assemble_hyperbo(
            mesh, ordre, opu="nxuT", opv="nxvT",
            F=func_F_vx, kind="squelette_element", methode="DG", domaine="all")
        A_xx_TV = assemble_hyperbo(
            mesh, ordre, opu="nxuT", opv="nxvV",
            F=func_F_vx, kind="squelette_element", methode="DG", domaine="all")
        A_yy_TT = assemble_hyperbo(
            mesh, ordre, opu="nyuT", opv="nyvT",
            F=func_F_vy, kind="squelette_element", methode="DG", domaine="all")
        A_yy_TV = assemble_hyperbo(
            mesh, ordre, opu="nyuT", opv="nyvV",
            F=func_F_vy, kind="squelette_element", methode="DG", domaine="all")
        A_yx_TT = assemble_hyperbo(
            mesh, ordre, opu="nyuT", opv="nxvT",
            F=func_F_cross, kind="squelette_element", methode="DG", domaine="all")
        A_yx_TV = assemble_hyperbo(
            mesh, ordre, opu="nyuT", opv="nxvV",
            F=func_F_cross, kind="squelette_element", methode="DG", domaine="all")
    else:
        A_xx_TT=None
        A_xx_TV=None
        A_yy_TT=None
        A_yy_TV=None
        A_yx_TT=None
        A_yx_TV=None
    if need_pv:
        A_px_TT = assemble_hyperbo(
            mesh, ordre, opu="nxuT", opv="vT",
            F=Fx, kind="squelette_element", methode="DG", domaine="all")
        A_px_TV = assemble_hyperbo(
            mesh, ordre, opu="nxuV", opv="vT",
            F=Fx, kind="squelette_element", methode="DG", domaine="all")
        A_py_TT = assemble_hyperbo(
            mesh, ordre, opu="nyuT", opv="vT",
            F=Fy, kind="squelette_element", methode="DG", domaine="all")
        A_py_TV = assemble_hyperbo(
            mesh, ordre, opu="nyuV", opv="vT",
            F=Fy, kind="squelette_element", methode="DG", domaine="all")
    else:
        A_px_TT=None
        A_px_TV=None
        A_py_TT=None
        A_py_TV=None
    if formulation == "positif":
        blocs = [
            (+Z, A_p_TT),
            (-Z, A_p_TV),
            (+1./Z, A_xx_TT),
            (+1./Z, A_xx_TV),
            (+1./Z, A_yy_TT),
            (+1./Z, A_yy_TV),
            (+1./Z, A_yx_TT),
            (+1./Z, A_yx_TV),]
    elif formulation == "Fplus":
        blocs = [
            (+Z, A_p_TT),
            (-Z, A_p_TV),
            (+1./Z, A_xx_TT),
            (+1./Z, A_xx_TV),
            (+1./Z, A_yy_TT),
            (+1./Z, A_yy_TV),
            (+1./Z, A_yx_TT),
            (+1./Z, A_yx_TV),
            (+1, A_px_TT),
            (+1, A_px_TV),
            (+1, A_py_TT),
            (+1, A_py_TV),]
    elif formulation == "Fmoins":
        blocs = [
            (-Z, A_p_TT),
            (+Z, A_p_TV),
            (-1./Z, A_xx_TT),
            (-1./Z, A_xx_TV),
            (-1./Z, A_yy_TT),
            (-1./Z, A_yy_TV),
            (-1./Z, A_yx_TT),
            (-1./Z, A_yx_TV),
            (+1, A_px_TT),
            (+1, A_px_TV),
            (+1, A_py_TT),
            (+1, A_py_TV),]
    elif formulation == "reciprocite":
        blocs = [
            (+1, A_px_TT),
            (+1, A_py_TT),]
    else:
        raise NotImplementedError("Formulation non implémentée pour l'instant.")
    MAT = assemble_signed_blocks(blocs)
    return MAT

def assemble_volume_acoustique(mesh, ordre, kappa,formulation="forte"): 
    r"""
    Assemble la partie volumique cohérente du système acoustique.

        formulation = "forte"  :
        ∫_T (F0 U).V + (Fx ∂xU).V + (Fy ∂yU).V

        formulation = "faible" :
        ∫_T (F0 U).V - (Fx U).∂xV - (Fy U).∂yV
    
        formulation = "positif" :
        ∫_T (F0 U).V + 0.5 ∫_T (Fx ∂xU).V + 0.5 ∫_T (Fy ∂yU).V - 0.5 ∫_T (Fx U).∂xV - 0.5 ∫_T (Fy U).∂yV

    Paramètres
    ----------
        mesh : Mesh
            Maillage.
        ordre : int
            Ordre des éléments finis.
        kappa : float ou complex
            Nombre d'onde du problème acoustique.

    Retour
    ------
        COOMatrix
            Matrice creuse représentant le terme volumique.
    """
    from .methode_hyperbolique import assemble_hyperbo
    F0, Fx, Fy = build_acoustic_matrices(kappa)
    # Volume
    if formulation == "forte":
        needM0 = True
        needMx = True
        needMy = True
        needMxT = False
        needMyT = False
    elif formulation == "faible":
        needM0 = True
        needMx = False
        needMy = False
        needMxT = True
        needMyT = True
    elif formulation == "positif":
        needM0 = True
        needMx = True
        needMy = True
        needMxT = True
        needMyT = True
    else:
        raise NotImplementedError("formulation non reconnue pour l'assemblage volume.")
    if needM0:M0 = assemble_hyperbo(mesh, ordre, opu="u",   opv="v", F=F0, kind="volume", methode="DG", domaine="all")
    else:M0 = None
    if needMx:Mx = assemble_hyperbo(mesh, ordre, opu="dxu", opv="v", F=Fx, kind="volume", methode="DG", domaine="all")
    else:Mx = None
    if needMy:My = assemble_hyperbo(mesh, ordre, opu="dyu", opv="v", F=Fy, kind="volume", methode="DG", domaine="all")
    else:My = None
    if needMxT:MxT = assemble_hyperbo(mesh, ordre, opu="u", opv="dxv", F=Fx, kind="volume", methode="DG", domaine="all")
    else:MxT = None
    if needMyT:MyT = assemble_hyperbo(mesh, ordre, opu="u", opv="dyv", F=Fy, kind="volume", methode="DG", domaine="all")
    else:MyT = None
    
    blocs=[]
    if formulation == "forte":
        blocs=[
            (+1, M0),
            (+1, Mx),
            (+1, My)
        ]
    elif formulation == "faible":
        blocs=[
            (+1, M0),
            (-1, MxT),
            (-1, MyT)
        ]
    elif formulation == "positif":
        blocs=[
            (+1, M0),
            (+.5, Mx),
            (+.5, My),
            (-.5, MxT),
            (-.5, MyT)
        ]
    MATRICE_VOLUME = assemble_signed_blocks(blocs)
    return MATRICE_VOLUME
