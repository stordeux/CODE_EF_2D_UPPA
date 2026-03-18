import numpy as np
from mes_packages.assemblage_general import *

ZERO = lambda x,y: 0.0

def normalize_hyperbo_operator(op: str, kind: str) -> str:
    """
    Normalise les alias d'opérateurs pour rendre cohérents :
    - squelette : uTnx, uVnx, ...
    - frontière : nxu, nyu, ...
    - anciens alias : n1/n2
    """
    op = op.strip().replace(" ", "")

    # anciens alias n1/n2 -> nx/ny
    aliases = {
        "uTn1": "uTnx",
        "uVn1": "uVnx",
        "vTn1": "vTnx",
        "vVn1": "vVnx",
        "uTn2": "uTny",
        "uVn2": "uVny",
        "vTn2": "vTny",
        "vVn2": "vVny",
    }
    op = aliases.get(op, op)

    # sur la frontière, il n'y a que le côté T
    if kind == "frontiere":
        boundary_aliases = {
            "uTnx": "nxu",
            "uTny": "nyu",
            "vTnx": "nxv",
            "vTny": "nyv",
        }
        op = boundary_aliases.get(op, op)

    return op


def add_hyperbo(m,n,A,Mat,Nglob):
    """
    Assemble le block sparse A dans la matrice globale Mat
    aux positions (m,n) du système hyperbolique
    
    Parameters:
    -----------
    m, n : int
        Indices de block (composantes du système hyperbolique)
    A : COOMatrix
        Matrice block sparse à assembler
    Mat : COOMatrix
        Matrice globale où assembler
    Nglob : int
        Dimension d'un block
    """
    for i in range(A.l):
        Mat.ajout(m*Nglob + A.rows[i], n*Nglob + A.cols[i], A.data[i])


def build_exemple_1_F0():
    f = lambda x, y: 1 + x**2 + y**4
    g = lambda x, y: 1 + x**4 + y**2
    h = lambda x, y: 1 + x**2 + y**2

    return np.array([
        [f,                ZERO, ZERO],
        [ZERO,    g,             ZERO],
        [ZERO,    ZERO, h]
    ], dtype=object)
#########################################
### Construction d'un exmple de Fx et Fy#
#########################################


def build_exemple_1_F1():
    # Fx : dxd
    f11x = lambda x, y: 1 + x**2
    f22x = lambda x, y: 2 + y**2
    f33x = lambda x, y: 3 + x*y

    # Fy : dxd
    f11y = lambda x, y: 1 + y**2
    f22y = lambda x, y: 2 + x**2
    f33y = lambda x, y: 3 + x + y

    Fx = np.array([
        [f11x,             ZERO, ZERO],
        [ZERO,    f22x,          ZERO],
        [ZERO,    ZERO, f33x]
    ], dtype=object)

    Fy = np.array([
        [f11y,             ZERO, ZERO],
        [ZERO,    f22y,          ZERO],
        [ZERO,    ZERO, f33y]
    ], dtype=object)

    return (Fx, Fy)




def give_F_format(F):
    """
    Reconnaît:
      - F0 : ndarray (d,d) dtype=object contenant des callables
      - F1 : tuple (Fx,Fy) avec Fx,Fy ndarray (d,d) dtype=object contenant des callables

    Retourne:
      (Ftype, d) avec Ftype in {"F0type","F1type"}
    """

    # -----------------------
    # **cas F0 : matrice (d,d)**
    # -----------------------
    if isinstance(F, np.ndarray):
        if F.dtype != object:
            raise TypeError("F0 doit être un numpy array dtype=object.")
        if F.ndim != 2 or F.shape[0] != F.shape[1]:
            raise TypeError("F0 doit être une matrice carrée (d,d).")

        d = F.shape[0]
        for i in range(d):
            for j in range(d):
                if not callable(F[i, j]):
                    raise TypeError(f"F0[{i},{j}] n'est pas une fonction callable.")
        return ("F0type", d)  # **correction**

    # -----------------------
    # **cas F1 : tuple (Fx,Fy)**
    # -----------------------
    if isinstance(F, tuple) and len(F) == 2:
        Fx, Fy = F

        if not (isinstance(Fx, np.ndarray) and isinstance(Fy, np.ndarray)):
            raise TypeError("F1 doit être un tuple (Fx,Fy) de numpy arrays.")

        if Fx.dtype != object or Fy.dtype != object:
            raise TypeError("Fx et Fy doivent être dtype=object.")

        if Fx.ndim != 2 or Fy.ndim != 2:
            raise TypeError("Fx et Fy doivent être des matrices 2D (d,d).")

        if Fx.shape[0] != Fx.shape[1]:
            raise TypeError("Fx doit être une matrice carrée (d,d).")
        if Fy.shape[0] != Fy.shape[1]:
            raise TypeError("Fy doit être une matrice carrée (d,d).")
        if Fx.shape != Fy.shape:
            raise TypeError("Fx et Fy doivent avoir la même shape (d,d).")

        d = Fx.shape[0]
        # vérif callables dans Fx et Fy
        for A, name in [(Fx, "Fx"), (Fy, "Fy")]:
            for i in range(d):
                for j in range(d):
                    if not callable(A[i, j]):
                        raise TypeError(f"{name}[{i},{j}] n'est pas une fonction callable.")

        return ("F1type", d)  # **correction**

    raise TypeError("Format de F non reconnu (attendu F0: (d,d) ou F1: tuple(Fx,Fy)).")


F0ex =build_exemple_1_F0()

def assemble_hyperbo(
        mesh,
        ordre:int,
        opu:str,
        opv:str,
        F=F0ex,
        kind:str="volume",
        methode:str="DG",
        domaine:str="all") -> COOMatrix:
    """ 
    Assemble la matrice de squelette pour un système hyperbolique donné par les opérateurs opu et opv.
    
    Parameters:
    -----------
    mesh : Mesh
        Le maillage sur lequel assembler
    opu, opv : str
        Les opérateurs à assembler (ex: "uT", "vV", "(M.n)uT", etc.)
    kind : str
        Le type de système hyperbolique (ex: "advection", "maxwell", etc.)
    methode : str, optional
        La méthode d'assemblage ("DG" par défaut)
    domaine : str, optional
        Le domaine d'assemblage ("all" par défaut)
    dtype : data-type, optional
        Le type de données pour la matrice assemblée (np.complex128 par défaut)
    
    Returns:
    --------
    COOMatrix
        La matrice de squelette assemblée pour le système hyperbolique spécifié.
    """

    opu = normalize_hyperbo_operator(opu, kind)
    opv = normalize_hyperbo_operator(opv, kind)

    # Calcul de
    # Nglob : nombre de degrés de liberté globaux par composante du système
    # NT : nombre de triangles du maillage
    # Nloc : nombre de degrés de liberté locaux par triangle
    LoctoGlob = loc_to_glob_general(mesh, ordre, methode)
    Nglob = int(np.max(LoctoGlob)) + 1
    Nloc = (ordre + 1) * (ordre + 2) // 2
    NT = mesh.cells_dict["triangle"].shape[0]

    # Récupération du format de F
    Ftype,d = give_F_format(F)
    if kind == "volume":
        if Ftype == "F0type":
            Mat = COOMatrix(d*Nglob, d*Nglob, d**2 * NT * Nloc**2)  # Estimation du nombre d'entrées non nulles
            for m in range(d):
                for n in range(d):
                    coef_mn = F[m, n]
                    A_mn = assemble_volume(mesh, ordre, coef_mn, opu, opv, methode=methode)
                    add_hyperbo(m, n, A_mn, Mat, Nglob)
            return Mat
        elif Ftype == "F1type":
            raise NotImplementedError(
                "Pour F1=(Fx,Fy), assembler séparément : "
                "A_x = assemble_hyperbo(..., F=Fx, opu='dxu', opv='v') et "
                "A_y = assemble_hyperbo(..., F=Fy, opu='dyu', opv='v'), puis sommer."
            )
        else:
            raise NotImplementedError("Format de F non reconnu pour l'assemblage volume.")
    elif kind == "frontiere" or methode == "surface":
        if Ftype == "F0type":
            Mat = COOMatrix(d*Nglob, d*Nglob, d**2 * NT * Nloc**2)  # Estimation du nombre d'entrées non nulles
            for m in range(d):
                for n in range(d):
                    coef_mn = F[m, n]
                    A_mn = assemble_surface(mesh, ordre, coef_mn, opu, opv, methode=methode,domaine=domaine)
                    add_hyperbo(m, n, A_mn, Mat, Nglob)
            return Mat
            
        elif Ftype == "F1type":
            raise NotImplementedError(
                "Pour F1=(Fx,Fy), assembler séparément : "
                "A_x = assemble_surface(..., F=Fx, opu='u', opv='v') et "
                "A_y = assemble_surface(..., F=Fy, opu='v', opv='v'), puis sommer."
            )
        else:
            raise NotImplementedError("Format de F non reconnu pour l'assemblage frontiere.")
    elif kind == "squelette_element" or methode == "squelette":
        if Ftype == "F0type":
            # Nombre de faces internes (comptées une fois)
            triangles = np.asarray(mesh.cells_dict["triangle"])
            N_faces_int = count_interior_faces(mesh)
            nnz_est = max(1, 4 * N_faces_int * Nloc * Nloc)
            Mat = COOMatrix(d*Nglob, d*Nglob, d**2 * nnz_est)  # Estimation du nombre d'entrées non nulles
            for m in range(d):
                for n in range(d):
                    coef_mn = F[m, n]
                    A_mn = assemble_skeleton_par_element(mesh, ordre, coef_mn, opu, opv, methode)
                    add_hyperbo(m, n, A_mn, Mat, Nglob)
            return Mat
        elif Ftype == "F1type":
            raise NotImplementedError(
                "Pour F1=(Fx,Fy), assembler séparément : "
                "A_x = assemble_skeleton_par_element(..., F=Fx, opu='u', opv='v') et "
                "A_y = assemble_skeleton_par_element(..., F=Fy, opu='v', opv='v'), puis sommer."
            )
        else: 
            raise NotImplementedError("Format de F non reconnu pour l'assemblage squelette.")
    else:
        raise NotImplementedError("type d'assemblage non pris en charge pour l'instant.")
    
def exemple_fonction_vectorielle(theta, kappa):
    kx = kappa * np.cos(theta)
    ky = kappa * np.sin(theta)

    def phase(x, y):
        return np.exp(1j * (kx * x + ky * y))

    func0 = lambda x, y, f=phase: f(x, y)
    func1 = lambda x, y, f=phase: (kx / kappa) * f(x, y)
    func2 = lambda x, y, f=phase: (ky / kappa) * f(x, y)

    return func0, func1, func2


def build_vecteur_nodal_hyperbolique(mesh, ordre, func_vec, methode="DG"):
    """
    Construit le vecteur nodal global associé à une fonction vectorielle func_vec
    pour un système hyperbolique.

    Parameters
    ----------
    mesh : mesh
    ordre : int
        Ordre des éléments finis
    func_vec : tuple/list de fonctions
        func_vec[m](x,y) = composante m du champ
    methode : "DG" ou "CG"

    Returns
    -------
    U : ndarray (d*Nglob,)
        vecteur nodal global concaténé par composantes
    """

    # nombre de composantes du système
    d = len(func_vec)

    # connectivité locale → globale (déjà utilisée dans tes matrices)
    LoctoGlob = loc_to_glob_general(mesh, ordre, methode)
    Nglob = int(np.max(LoctoGlob)) + 1

    # coordonnées des DOFs
    if methode == "DG":
        dof_coords = build_dof_coordinates_DG(mesh, ordre)
    else:
        raise NotImplementedError("CG non encore branché ici.")

    # vecteur global hyperbolique
    U = np.zeros(d * Nglob, dtype=np.complex128)

    # remplissage composante par composante
    for m in range(d):
        f = func_vec[m]

        for i in range(Nglob):
            x, y = dof_coords[i]
            U[m * Nglob + i] = f(x, y)

    return U

def plot_nodal_vector_hyperbolique(mesh, U, d, ordre, methode="DG"):
    """
    Affiche le champ nodal vectoriel U pour un système hyperbolique.

    Parameters
    ----------
    mesh : mesh
    U : ndarray (d*Nglob,)
        vecteur nodal global concaténé par composantes
    d : int
        nombre de composantes du système
    ordre : int
        ordre des éléments finis
    methode : "DG" ou "CG"
    secondes : float
        durée d'affichage en secondes (0 pour rester ouvert)
    """

    # connectivité locale → globale
    LoctoGlob = loc_to_glob_general(mesh, ordre, methode)
    Nglob = int(np.max(LoctoGlob)) + 1

    # coordonnées des DOFs
    if methode == "DG":
        dof_coords = build_dof_coordinates_DG(mesh, ordre)
    else:
        raise NotImplementedError("CG non encore branché ici.")

    # extraction des composantes et affichage
    for m in range(d):
        comp_m = U[m * Nglob:(m + 1) * Nglob]
        plot_nodal_vector_DG(U=comp_m,mesh=mesh, ordre=ordre, title=f"Composante {m}")

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

def assemble_saut_acoustique(mesh, ordre, kappa):
    _, Fx, Fy = build_acoustic_matrices(kappa)
    func_F_p, func_F_vx, func_F_vy, func_F_cross = build_penalty_matrices()

    A_p_TT = assemble_hyperbo(
        mesh, ordre, opu="uT", opv="vT",
        F=func_F_p, kind="squelette_element", methode="DG", domaine="all")
    A_p_TV = assemble_hyperbo(
        mesh, ordre, opu="uT", opv="vV",
        F=func_F_p, kind="squelette_element", methode="DG", domaine="all")
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

    blocs = [
        (+1, A_p_TT),
        (-1, A_p_TV),
        (+1, A_xx_TT),
        (+1, A_xx_TV),
        (+1, A_yy_TT),
        (+1, A_yy_TV),
        (+1, A_yx_TT),
        (+1, A_yx_TV),
        (+1, A_px_TT),
        (+1, A_px_TV),
        (+1, A_py_TT),
        (+1, A_py_TV),
    ]
    MATRICE_INTERIEURE = assemble_signed_blocks(blocs)
    return MATRICE_INTERIEURE

def assemble_frontiere_coercif(mesh, ordre):
    func_F_p, func_F_vx, func_F_vy, func_F_cross = build_penalty_matrices()
    A_p_TT = assemble_hyperbo(
        mesh, ordre, opu="u", opv="v",
        F=func_F_p, kind="frontiere", methode="DG", domaine="all")
    A_xx_TT = assemble_hyperbo(
        mesh, ordre, opu="nxu", opv="nxv",
        F=func_F_vx, kind="frontiere", methode="DG", domaine="all")
    A_yy_TT = assemble_hyperbo(
        mesh, ordre, opu="nyu", opv="nyv",
        F=func_F_vy, kind="frontiere", methode="DG", domaine="all")
    A_yx_TT = assemble_hyperbo(
        mesh, ordre, opu="nyu", opv="nxv",
        F=func_F_cross, kind="frontiere", methode="DG", domaine="all")

    blocs = [
        (+1, A_p_TT),
        (+1, A_xx_TT),
        (+1, A_yy_TT),
        (+1, A_yx_TT),
    ]
    Mat_bord = assemble_signed_blocks(blocs)
    return Mat_bord

def assemble_penalisation_interieure(mesh, ordre,kappa):
    _, Fx, Fy = build_acoustic_matrices(kappa)
    func_F_p, func_F_vx, func_F_vy, func_F_cross = build_penalty_matrices()
    A_p_TT = assemble_hyperbo(
        mesh, ordre, opu="uT", opv="vT",
        F=func_F_p, kind="squelette_element", methode="DG", domaine="all")
    A_p_TV = assemble_hyperbo(
        mesh, ordre, opu="uT", opv="vV",
        F=func_F_p, kind="squelette_element", methode="DG", domaine="all")
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

    blocs = [
        (+1, A_p_TT),
        (-1, A_p_TV),
        (+1, A_xx_TT),
        (+1, A_xx_TV),
        (+1, A_yy_TT),
        (+1, A_yy_TV),
        (+1, A_yx_TT),
        (+1, A_yx_TV),]
    Mat_penalisation = assemble_signed_blocks(blocs)
    return Mat_penalisation


def assemble_interieur_coercif(mesh, ordre, kappa):
    _, Fx, Fy = build_acoustic_matrices(kappa)
    func_F_p, func_F_vx, func_F_vy, func_F_cross = build_penalty_matrices()
    A_p_TT = assemble_hyperbo(
        mesh, ordre, opu="uT", opv="vT",
        F=func_F_p, kind="squelette_element", methode="DG", domaine="all")
    A_p_TV = assemble_hyperbo(
        mesh, ordre, opu="uT", opv="vV",
        F=func_F_p, kind="squelette_element", methode="DG", domaine="all")
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
    A_px_TV = assemble_hyperbo(
        mesh, ordre, opu="nxuV", opv="vT",F=Fx, kind="squelette_element", methode="DG", domaine="all")
    A_py_TV = assemble_hyperbo(
        mesh, ordre, opu="nyuV", opv="vT",F=Fy, kind="squelette_element", methode="DG", domaine="all")

    blocs = [
        (+1, A_p_TT),
        (-1, A_p_TV),
        (+1, A_xx_TT),
        (+1, A_xx_TV),
        (+1, A_yy_TT),
        (+1, A_yy_TV),
        (+1, A_yx_TT),
        (+1, A_yx_TV),
        (+1, A_px_TV),
        (+1, A_py_TV),
    ]
    Mat_coercive = assemble_signed_blocks(blocs)
    return Mat_coercive

def assemble_volume_acoustique(mesh, ordre, kappa): 
    F0, Fx, Fy = build_acoustic_matrices(kappa)
    blocs=[]
    # Volume
    M0 = assemble_hyperbo(mesh, ordre, opu="u",   opv="v", F=F0, kind="volume", methode="DG", domaine="all")
    Mx = assemble_hyperbo(mesh, ordre, opu="dxu", opv="v", F=Fx, kind="volume", methode="DG", domaine="all")
    My = assemble_hyperbo(mesh, ordre, opu="dyu", opv="v", F=Fy, kind="volume", methode="DG", domaine="all")
    blocs=[
        (+1, M0),
        (+1, Mx),
        (+1, My)
    ]
    MATRICE_VOLUME = assemble_signed_blocks(blocs)
    return MATRICE_VOLUME

def assemble_volume_acoustique_consistant(mesh, ordre, kappa):
    F0, Fx, Fy = build_acoustic_matrices(kappa)
    # Volume
    M0 = assemble_hyperbo(mesh, ordre, opu="u",   opv="v", F=F0, kind="volume", methode="DG", domaine="all")
    Mx = assemble_hyperbo(mesh, ordre, opu="dxu", opv="v", F=Fx, kind="volume", methode="DG", domaine="all")
    My = assemble_hyperbo(mesh, ordre, opu="dyu", opv="v", F=Fy, kind="volume", methode="DG", domaine="all")
    blocs=[
        (+1, M0),
        (+1, Mx),
        (+1, My)
    ]
    MATRICE_VOLUME_CONSISTANT = assemble_signed_blocks(blocs)
    return MATRICE_VOLUME_CONSISTANT

def assemble_volume_acoustique_positif(mesh, ordre, kappa): 
    F0, Fx, Fy = build_acoustic_matrices(kappa)
    blocs=[]
    # Volume
    M0 = assemble_hyperbo(mesh, ordre, opu="u",   opv="v", F=F0, kind="volume", methode="DG", domaine="all")
    Mx = assemble_hyperbo(mesh, ordre, opu="dxu", opv="v", F=Fx, kind="volume", methode="DG", domaine="all")
    MxT = assemble_hyperbo(mesh, ordre, opu="u", opv="dxv", F=Fx, kind="volume", methode="DG", domaine="all")
    My = assemble_hyperbo(mesh, ordre, opu="dyu", opv="v", F=Fy, kind="volume", methode="DG", domaine="all")
    MyT = assemble_hyperbo(mesh, ordre, opu="u", opv="dyv", F=Fy, kind="volume", methode="DG", domaine="all")
    blocs=[
        (+2, M0),
        (-1, Mx),
        (-1, My),
        (1, MxT),
        (1, MyT)
    ]
    MATRICE_VOLUME_POSITIF = assemble_signed_blocks(blocs)
    return MATRICE_VOLUME_POSITIF

def assemble_saut_reciprocite(mesh, ordre, kappa):
    _,func_F_1, func_F_2 = build_acoustic_matrices(kappa)
    A_px_TT = assemble_hyperbo(mesh, ordre, opu="nxuT", opv="vT",
        F=func_F_1, kind="squelette_element", methode="DG", domaine="all")
    A_px_TV = assemble_hyperbo(mesh, ordre, opu="nxuV", opv="vT",
        F=func_F_1, kind="squelette_element", methode="DG", domaine="all")
    A_py_TT = assemble_hyperbo(mesh, ordre, opu="nyuT", opv="vT",
        F=func_F_2, kind="squelette_element", methode="DG", domaine="all")
    A_py_TV = assemble_hyperbo(mesh, ordre, opu="nyuV", opv="vT",
        F=func_F_2, kind="squelette_element", methode="DG", domaine="all")
    blocs = [
        (+1, A_px_TT),
        (+1, A_px_TV),
        (+1, A_py_TT),
        (+1, A_py_TV),
    ]
    MATRICE_SAUT_RECIPROCITE = assemble_signed_blocks(blocs)
    return MATRICE_SAUT_RECIPROCITE

def assemble_matrice_totale_sans_frontiere(mesh, ordre, kappa):
    M_volume = assemble_volume_acoustique(mesh, ordre, kappa)
    M_penalisation = assemble_saut_acoustique(mesh, ordre, kappa)
    blocs = [
        (+1, M_volume),
        (+1, M_penalisation)
    ]
    MATRICE_TOTALE_SANS_FRONTIERE = assemble_signed_blocks(blocs)
    return MATRICE_TOTALE_SANS_FRONTIERE


def assemble_frontiere_consistant(mesh, ordre, kappa,Z):
    if np.isclose(Z, 0):
       raise ValueError("Z ne doit pas être nul.")
    else:
        _, Fx, Fy = build_acoustic_matrices(kappa)
        func_F_p, func_F_vx, func_F_vy, func_F_cross = build_penalty_matrices()
        A_p_TT = assemble_hyperbo(
            mesh, ordre, opu="u", opv="v",
            F=func_F_p, kind="frontiere", methode="DG", domaine="all")
        A_xx_TT = assemble_hyperbo(
            mesh, ordre, opu="nxu", opv="nxv",
            F=func_F_vx, kind="frontiere", methode="DG", domaine="all")
        A_yy_TT = assemble_hyperbo(
            mesh, ordre, opu="nyu", opv="nyv",
            F=func_F_vy, kind="frontiere", methode="DG", domaine="all")
        A_yx_TT = assemble_hyperbo(
            mesh, ordre, opu="nyu", opv="nxv",
            F=func_F_cross, kind="frontiere", methode="DG", domaine="all")
        A_px_TT = assemble_hyperbo(
            mesh, ordre, opu="nxu", opv="v",F=Fx, kind="frontiere", methode="DG", domaine="all")
        A_py_TT = assemble_hyperbo(mesh, ordre, opu="nyu", opv="v",F=Fy, 
            kind="frontiere", methode="DG", domaine="all")

        blocs = [
            (Z, A_p_TT),
            (1/Z, A_xx_TT),
            (1/Z, A_yy_TT),
            (1/Z, A_yx_TT),
            (+1, A_px_TT),
            (+1, A_py_TT),
        ]
        Mat_bord = assemble_signed_blocks(blocs)

        return Mat_bord
    
def assemble_rhs_acoustique(mesh,ordre, func_vec=None, Z=1.0, domaine="all"):
    """
    Assemble le second membre de frontière consistant associé à

        (1/Z) (v_inc.n + Z p_inc) (v'.n + Z p')

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
        func_vec = exemple_fonction_vectorielle(np.pi/3, 2*np.pi/1.3)

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

    # Il y a quatre blocs à assembler pour le second membre de frontière consistant :

    # Bloc 1 : ∫ Z p_inc p' ds
    rhs_p = Z*assemble_rhs_surface(
            mesh, ordre,f=p_inc,op_f="f",
            op_v="v",methode="DG",domaine=domaine
        )
    # Bloc 2 : ∫ v_inc.n p' ds
    rhs_p+= assemble_rhs_surface(
            mesh, ordre,f=v_inc,op_f="f.n",
            op_v="v",methode="DG",domaine=domaine)

    # Bloc 3 : ∫ p_inc v'.n ds
    # Le bloc 3 se divise en deux parties :
    # ∫ p_inc nx v'_x ds
    # ∫ p_inc ny v'_y ds
    rhs_vx = assemble_rhs_surface(
            mesh, ordre,f=p_inc,op_f="f",
            op_v="nxv",methode="DG",domaine=domaine)
    rhs_vy = assemble_rhs_surface(
            mesh, ordre,f=p_inc,op_f="f",
            op_v="nyv",methode="DG",domaine=domaine)
    # Bloc 4 : ∫ (1/Z) v_inc.n v'.n ds
    # Le bloc 4 se divise en deux parties :
    # ∫ (1/Z) v_inc.n nx v'_x ds
    # ∫ (1/Z) v_inc.n ny v'_y ds
    rhs_vx += (1.0 / Z) * assemble_rhs_surface(
            mesh, ordre,f=v_inc,
            op_f="f.n",op_v="nxv",methode="DG",domaine=domaine)
    rhs_vy += (1.0 / Z) * assemble_rhs_surface(
            mesh, ordre,f=v_inc,op_f="f.n",
            op_v="nyv",methode="DG",domaine=domaine)

    Nglob = rhs_p.shape[0]
    rhs = np.zeros(3 * Nglob, dtype=np.complex128)
    rhs[0:Nglob] = rhs_p
    rhs[Nglob:2*Nglob] = rhs_vx
    rhs[2*Nglob:3*Nglob] = rhs_vy

    return rhs