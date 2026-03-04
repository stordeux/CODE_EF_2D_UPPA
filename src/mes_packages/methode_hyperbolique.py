import numpy as np
from mes_packages.assemblage_general import *

ZERO = lambda x,y: 0.0

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
                    A_mn = assemble_volume(mesh, ordre, coef_mn, opu, opv, methode)
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
                    A_mn = assemble_surface(mesh, ordre, coef_mn, opu, opv, methode)
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
            neighbors, neighbor_faces, edges_to_triangles = build_neighborhood_structure(triangles)
            n_faces_int = 0
            for T in range(NT):
                for Face in range(3):
                    V = neighbors[T, Face]
                    if V >= 0 and T < V:  # on compte chaque face interne une fois (T < V)
                        n_faces_int += 1
            nnz_est = max(1, 4 * n_faces_int * Nloc * Nloc)
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
