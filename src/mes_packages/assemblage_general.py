import numpy as np
from mes_packages import *
from .quadrature import quadrature_triangle_ref_2D


def loc_to_glob_general(mesh, ordre, methode):
    """
    Retourne la table locale -> globale adaptée à la méthode choisie (CG ou DG).

    Parameters
    ----------
    mesh : meshio.Mesh
        Maillage triangulaire.
    ordre : int
        Ordre polynomial.
    methode : {"CG", "DG"}
        Type de discrétisation :
        - "CG" : éléments finis continus (identification des DDL entre éléments)
        - "DG" : éléments discontinus (DDL indépendants par élément)

    Returns
    -------
    LoctoGlob : ndarray (NT, Nloc)
        Table locale -> globale plate telle que
            LoctoGlob[T, k] = indice global du k-ième DDL de l’élément T
        avec
            Nloc = (ordre+1)*(ordre+2)//2.
    """
    if methode == "DG":
        LoctoGlob,_=build_loctoglob_DG(mesh, ordre)
    elif methode == "CG":
        LoctoGlob,_,_=build_loctoglob_CG(mesh, ordre)
    else:
        raise ValueError(f"Méthode inconnue : {methode}")
    return LoctoGlob 

def insert(Mat, T:int,Mloc,ordre:int,LoctoGlob):
    """
    Insère une matrice élémentaire dans la matrice globale via la table
    locale -> globale fournie.

    Cette routine est indépendante de la méthode (CG ou DG) :
    toute l'information topologique est contenue dans `LoctoGlob`.

    Parameters
    ----------
    Mat : COOMatrix
        Matrice globale en cours d'assemblage.
    T : int
        Indice de l’élément.
    Mloc : ndarray (Nloc, Nloc)
        Matrice locale associée à l’élément T.
    ordre : int
        Ordre polynomial (sert uniquement à la cohérence dimensionnelle).
    LoctoGlob : ndarray (NT, Nloc)
        Table locale -> globale telle que retournée par `loc_to_glob_general`.

        - En DG : indices globaux distincts pour chaque élément
        - En CG : indices partagés entre éléments voisins

    Notes
    -----
    Cette fonction réalise l’assemblage générique :

        A[I,J] += Mloc[i,j]
        avec I = LoctoGlob[T,i], J = LoctoGlob[T,j]

    Le comportement CG/DG est entièrement déterminé par `LoctoGlob`.
    """
    for i,iglob in enumerate(LoctoGlob[T]):
            for j, jglob in enumerate(LoctoGlob[T]):
                Mat.ajout(iglob, jglob, Mloc[i, j])


# ============================================================
# Gradient des bases sur le triangle de référence (vectorisé)
# ============================================================

def grad_base_ref(xq, yq, m: int, n: int, ordre: int):
    """
    Gradient exact de la fonction de base (m,n) sur le triangle de référence.

    On calcule ici les dérivées par rapport aux coordonnées de référence
    (x̂, ŷ) et NON les dérivées physiques.

    Parameters
    ----------
    xq, yq : array_like
        Points de quadrature dans le triangle de référence.
    m, n : int
        Indices barycentriques de la fonction de base.
    ordre : int
        Ordre polynomial.

    Returns
    -------
    dphidxhat, dphidyhat : ndarray
        Dérivées de φ par rapport aux coordonnées de référence :
            dphidxhat = ∂φ/∂x̂
            dphidyhat = ∂φ/∂ŷ
    """

    # vectorisation de ta dérivée analytique
    dphidxhat = np.vectorize(
        lambda x, y: derivative_base(x, y, m, n, ordre, var='x')
    )(xq, yq)

    dphidyhat = np.vectorize(
        lambda x, y: derivative_base(x, y, m, n, ordre, var='y')
    )(xq, yq)

    return dphidxhat, dphidyhat


def precompute_ref(ordre: int, ordreq: int | None = None):
    if ordreq is None:
        ordreq = 2 * ordre + 2

    wq, xq, yq = quadrature_triangle_ref_2D(ordreq)
    xq = np.asarray(xq); yq = np.asarray(yq); wq = np.asarray(wq)

    Nloc = (ordre + 1) * (ordre + 2) // 2
    Nq = len(wq)

    Phi_val   = np.empty((Nloc, Nq), dtype=np.complex128)
    Phi_dxhat = np.empty((Nloc, Nq), dtype=np.complex128)
    Phi_dyhat = np.empty((Nloc, Nq), dtype=np.complex128)

    # vectorisation une fois
    dbase_dx = np.vectorize(lambda xx, yy, m, n: derivative_base(xx, yy, m, n, ordre, var="x"))
    dbase_dy = np.vectorize(lambda xx, yy, m, n: derivative_base(xx, yy, m, n, ordre, var="y"))

    for m in range(ordre + 1):
        for n in range(ordre + 1 - m):
            k = loc2D_to_loc1D(m, n)
            Phi_val[k, :]   = base(xq, yq, m, n, ordre)
            Phi_dxhat[k, :] = dbase_dx(xq, yq, m, n)
            Phi_dyhat[k, :] = dbase_dy(xq, yq, m, n)

    return wq, xq, yq, Phi_val, Phi_dxhat, Phi_dyhat

def assemble_volume(mesh, ordre, func, operatoru, operatorv, methode="CG"):
    # Pré-calcul ref (UNE fois)
    wq, xq, yq, Phi_val, Phi_dxhat, Phi_dyhat = precompute_ref(ordre)
    wq = np.asarray(wq)
    xq = np.asarray(xq)
    yq = np.asarray(yq)

    Nloc = (ordre + 1) * (ordre + 2) // 2

    # Connectivité CG/DG
    LoctoGlob = loc_to_glob_general(mesh, ordre, methode)
    Nglob = int(np.max(LoctoGlob)) + 1
    NT = mesh.cells_dict["triangle"].shape[0]
    A  = COOMatrix(Nglob, Nglob, NT * Nloc * Nloc)

    points = mesh.points[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])

    # besoin de dx/dy ?
    need_dx = (operatoru == "dxu") or (operatorv == "dxv")
    need_dy = (operatoru == "dyu") or (operatorv == "dyv")

    for T, (i0, i1, i2) in enumerate(triangles):

        A0 = points[i0]
        A1 = points[i1]
        A2 = points[i2]

        # Jacobien affine
        J = np.column_stack((A1 - A0, A2 - A0))
        detJ = abs(np.linalg.det(J))
        Jinv = np.linalg.inv(J)

        # Points physiques
        x_phys = A0[0] + xq*(A1[0]-A0[0]) + yq*(A2[0]-A0[0])
        y_phys = A0[1] + xq*(A1[1]-A0[1]) + yq*(A2[1]-A0[1])

        f_q = np.asarray(func(x_phys, y_phys), dtype=np.complex128)


        Phi_dx = np.zeros_like(Phi_dxhat)
        Phi_dy = np.zeros_like(Phi_dyhat)
        
        # Dérivées physiques (si nécessaires)
        if need_dx:
            Phi_dx[:, :] = Jinv[0, 0] * Phi_dxhat + Jinv[1, 0] * Phi_dyhat  # ∂/∂x
        if need_dy:
            Phi_dy[:, :] = Jinv[0, 1] * Phi_dxhat + Jinv[1, 1] * Phi_dyhat  # ∂/∂y

        # Sélection opérateurs
        if operatoru == "u":
            Phi_u = Phi_val
        elif operatoru == "dxu":
            Phi_u = Phi_dx
        elif operatoru == "dyu":
            Phi_u = Phi_dy
        else:
            raise ValueError(f"Opérateur inconnu pour u : {operatoru}")

        if operatorv == "v":
            Phi_v = Phi_val
        elif operatorv == "dxv":
            Phi_v = Phi_dx
        elif operatorv == "dyv":
            Phi_v = Phi_dy
        else:
            raise ValueError(f"Opérateur inconnu pour v : {operatorv}")

        # Matrice locale (version “vraie brique” vectorisée)
        weight = wq * f_q                      # (Nq,)
        Mloc = detJ * (Phi_v * weight[None, :]) @ Phi_u.T  # (Nloc,Nloc)

        # Insertion CG/DG
        insert(A, T, Mloc, ordre, LoctoGlob)

    return A
