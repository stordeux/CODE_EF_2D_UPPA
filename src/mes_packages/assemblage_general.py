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

#####################################################################################
####### Assemblage symbolique des frontières ######################################## #####################################################################################

import numpy as np

def precompute_face_ref(ordre: int, ordreq: int | None = None, dtype=np.complex128):
    """
    Pré-calcul sur le triangle de référence des quantités nécessaires aux intégrales de frontière.

    La fonction construit, pour chacune des 3 faces :
        - les points et poids de quadrature 1D sur [0,1],
        - les valeurs des fonctions de base locales restreintes à la face,
        - leurs dérivées dans les coordonnées de référence.

    Ces tableaux sont indépendants du maillage et peuvent être réutilisés pour tous
    les éléments lors de l’assemblage (CG ou DG), ce qui évite de recalculer les bases
    à chaque boucle élément.

    Paramètres
    ----------
    ordre : int
        Ordre polynomial de l'élément fini.
    ordreq : int, optionnel
        Ordre de quadrature (par défaut : 2*ordre + 2).
    dtype : type numpy
        Type de stockage des tableaux.

    Retour
    ------
    w : (Nq,)
        Poids de quadrature paramétriques (à multiplier par la longueur physique).
    xhat, yhat : (3, Nq)
        Coordonnées des points de quadrature sur chaque face du triangle de référence.
    phi : (3, Nloc, Nq)
        Valeurs des fonctions de base sur les faces.
    dphidxh, dphidyh : (3, Nloc, Nq)
        Dérivées des fonctions de base dans le repère de référence.
    """
    if ordreq is None:
        ordreq = 2 * ordre + 2

    # Quadrature Gauss-Legendre sur t ∈ [0,1]
    xi, wi = np.polynomial.legendre.leggauss(ordreq)   # sur [-1,1]
    t = 0.5 * (xi + 1.0)                               # -> [0,1]
    w = 0.5 * wi                                       # poids pour dt
    Nloc = (ordre + 1) * (ordre + 2) // 2
    Nq = t.size

    # Sommets du triangle de référence
    A = np.array([[0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]])

    # Faces locales : face f = segment entre A[i0] et A[i1]
    # (ici : face 0 = (1,2), face 1 = (0,2), face 2 = (0,1))
    faces = np.array([[1, 2],
                      [0, 2],
                      [1, 0]], dtype=int)

    # Coordonnées (xhat,yhat) des points de quadrature sur chaque face
    xhat = np.empty((3, Nq), dtype=float)
    yhat = np.empty((3, Nq), dtype=float)

    for f, (i0, i1) in enumerate(faces):
        P0, P1 = A[i0], A[i1]
        # Paramétrisation cohérente avec ton code : Xhat(t)=P0*t + P1*(1-t)
        xhat[f, :] = P0[0] * t + P1[0] * (1.0 - t)
        yhat[f, :] = P0[1] * t + P1[1] * (1.0 - t)

    # Tenseurs : (face, ddl_local, point_quadrature)
    phi      = np.empty((3, Nloc, Nq), dtype=dtype)
    dphidxh  = np.empty((3, Nloc, Nq), dtype=dtype)
    dphidyh  = np.empty((3, Nloc, Nq), dtype=dtype)

    for m in range(ordre + 1):
        for n in range(ordre + 1 - m):
            k = loc2D_to_loc1D(m, n)
            for f in range(3):
                xx = xhat[f, :]
                yy = yhat[f, :]
                phi[f, k, :]     = base(xx, yy, m, n, ordre)
                dphidxh[f, k, :] = derivative_base(xx, yy, m, n, ordre, var="x")
                dphidyh[f, k, :] = derivative_base(xx, yy, m, n, ordre, var="y")

    return w, xhat, yhat, phi, dphidxh, dphidyh





def assemble_surface(mesh, ordre, func, operatoru, operatorv, methode="CG", domaine = "all"):
    # Pré-calcul ref face (UNE fois)
    wq, xq, yq, Phi_val, Phi_dxhat, Phi_dyhat = precompute_face_ref(ordre)
    Nloc = (ordre + 1) * (ordre + 2) // 2

    # Connectivité CG/DG
    LoctoGlob = loc_to_glob_general(mesh, ordre, methode)
    Nglob = int(np.max(LoctoGlob)) + 1
    NT = mesh.cells_dict["triangle"].shape[0]
    # Type de frontière :
    neighbors, neighbor_faces, edges_to_triangles, reference_BC, bc_name= build_neighborhood_structure_with_bc(mesh)

    A  = COOMatrix(Nglob, Nglob, NT * Nloc * Nloc)

    points = mesh.points[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])

    # besoin de dx/dy ?
    need_dx = (operatoru in ("dxu","dyu","dnu","dtu") or operatorv in ("dxv","dyv","dnv","dtv"))
    need_dn = (operatoru == "dnu") or (operatorv == "dnv")
    need_dt = (operatoru == "dtu") or (operatorv == "dtv")
    need_n = ("nx" in operatoru or "ny" in operatoru or "(M.n)" in operatoru or 
          "nx" in operatorv or "ny" in operatorv or "(M.n)" in operatorv)

    x_phys = np.empty_like(xq[0, :])
    y_phys = np.empty_like(yq[0, :])
    Phi_dx = np.empty_like(Phi_dxhat[0,:,:])
    Phi_dy = np.empty_like(Phi_dyhat[0,:,:])
    Phi_dn = np.empty_like(Phi_dxhat[0,:,:])
    Phi_dt = np.empty_like(Phi_dxhat[0,:,:])    
    Phi_u = np.empty_like(Phi_dxhat[0,:,:])    
    Phi_v = np.empty_like(Phi_dxhat[0,:,:])

    nx,ny=0,0

    for T, (i0, i1, i2) in enumerate(triangles):
        for F in range(3):
            V = neighbors[T, F]
            if domaine == "all":
                TEST = V <= -1
            else:
                refer = reference_BC(domaine)
                TEST = V == refer
            if TEST:   
                # Vérification du type de frontière à traiter
                A0 = points[i0]
                A1 = points[i1]
                A2 = points[i2]
                
                face_nodes = ((i1,i2),(i2,i0),(i0,i1))
                edge = points[face_nodes[F][1]] - points[face_nodes[F][0]]
                
                # Calcul de la longueur de la faces pour la pondération quadrature
                edge_length = np.linalg.norm(edge)
                
                # Jacobien affine
                J = np.column_stack((A1 - A0, A2 - A0))
                Jinv = np.linalg.inv(J)


                # Points physiques
                x_phys[:] = A0[0] + xq[F][:]*(A1[0]-A0[0]) + yq[F][:]*(A2[0]-A0[0])
                y_phys[:] = A0[1] + xq[F][:]*(A1[1]-A0[1]) + yq[F][:]*(A2[1]-A0[1])

 
                if need_dn or need_dt or need_n:
                    normale = calcul_normale(A0, A1, A2, F) 
                    nx, ny = normale[0], normale[1]

                fval = func(x_phys, y_phys)


                if isinstance(fval, tuple):
                    fx, fy = fval
                    fx = np.asarray(fx, dtype=np.complex128)
                    fy = np.asarray(fy, dtype=np.complex128)
                    Mn  = fx*nx + fy*ny
                    Mn  = np.asarray(Mn, dtype=np.complex128)
                    f_q = np.ones_like(Mn, dtype=np.complex128)
                else:
                    f_q = np.asarray(fval, dtype=np.complex128)
                    Mn  = np.ones_like(f_q, dtype=np.complex128)
                    
                # Dérivées physiques (si nécessaires)
                if need_dx:
                    Phi_dx[:, :] = Jinv[0, 0] * Phi_dxhat[F, :, :] + Jinv[1, 0] * Phi_dyhat[F, :, :]  # ∂/∂x
                    Phi_dy[:, :] = Jinv[0, 1] * Phi_dxhat[F, :, :] + Jinv[1, 1] * Phi_dyhat[F, :, :]  # ∂/∂y
                if need_dn:
                    Phi_dn[:,:] = nx*Phi_dx + ny*Phi_dy 
                if need_dt:
                    Phi_dt[:,:] = -ny*Phi_dx + nx*Phi_dy


                # Sélection opérateurs
                if operatoru == "u":
                    Phi_u[:,:] = Phi_val[F,:,:]
                elif operatoru == "dxu":
                    Phi_u[:,:] = Phi_dx[:,:]
                elif operatoru == "dyu":
                    Phi_u[:,:]   = Phi_dy[:,:]
                elif operatoru == "dnu":
                    Phi_u[:,:] = Phi_dn[:,:]
                elif operatoru == "dtu":
                    Phi_u[:,:] = Phi_dt[:,:]
                elif operatoru == "nxu":                            # *********************
                    Phi_u[:,:] = nx * Phi_val[F,:,:]               # *********************
                elif operatoru == "nyu":                            # *********************
                    Phi_u[:,:] = ny * Phi_val[F,:,:]
                elif (operatoru == "(M.n)u"):
                    Phi_u[:,:] =  Phi_val[F,:,:]
                else:
                    raise ValueError(f"Opérateur inconnu pour u : {operatoru}")

                if operatorv == "v":
                    Phi_v[:,:] = Phi_val[F,:,:]
                elif operatorv == "dxv":
                    Phi_v[:,:] = Phi_dx[:,:]
                elif operatorv == "dyv":
                    Phi_v[:,:] = Phi_dy[:,:]
                elif operatorv == "dnv":
                    Phi_v[:,:] = Phi_dn[:,:]
                elif operatorv == "dtv":
                    Phi_v[:,:] = Phi_dt[:,:]
                elif operatorv == "nxv":                            # *********************
                    Phi_v[:,:] = nx * Phi_val[F,:,:]               # *********************      
                elif operatorv == "nyv":                            # *********************
                    Phi_v[:,:] = ny * Phi_val[F,:,:]               # *********************
                elif (operatorv == "(M.n)v"):
                    Phi_v[:,:] =  Phi_val[F,:,:]

                else:
                    raise ValueError(f"Opérateur inconnu pour v : {operatorv}")

                # -------- poids quadrature correct --------
                weight = wq * f_q
                if operatoru == "(M.n)u" and operatorv == "(M.n)v":
                    weight = weight * (Mn * Mn)
                elif operatoru == "(M.n)u" or operatorv == "(M.n)v":
                    weight = weight * Mn

                # Matrice locale (version “vraie brique” vectorisée)

                Mloc = edge_length * (Phi_v * weight[None, :]) @ Phi_u.T  # (Nloc,Nloc)

                # Insertion CG/DG
                insert(A, T, Mloc, ordre, LoctoGlob)

    return A
#############################################################################
###   Terme sous \int_{\Omega} f Op(v) dx    ################################
#############################################################################

def assemble_rhs_volume(mesh, ordre, func, operatorv="v", methode="CG"):
    """
    Assemble un second membre volumique générique :

        F_i = ∫_Ω f(x,y) * O(v_i) dx

    avec O(v) ∈ { v, ∂x v, ∂y v }.

    operatorv accepté :
        - "v", "dxv", "dyv"
        - "fv", "fdxv", "fdyv"  (synonymes)

    Retour
    ------
    F : ndarray (Nglob,) complex
    """

    # Synonymes "fv" -> "v", etc.
    op_map = {"fv": "v", "fdxv": "dxv", "fdyv": "dyv"}
    operatorv = op_map.get(operatorv, operatorv)

    # Pré-calcul ref (UNE fois) : exactement comme assemble_volume
    wq, xq, yq, Phi_val, Phi_dxhat, Phi_dyhat = precompute_ref(ordre)
    wq = np.asarray(wq)
    xq = np.asarray(xq)
    yq = np.asarray(yq)

    # Connectivité CG/DG
    LoctoGlob = loc_to_glob_general(mesh, ordre, methode)
    Nglob = int(np.max(LoctoGlob)) + 1
    F = np.zeros(Nglob, dtype=np.complex128)

    points = mesh.points[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])

    need_dx = (operatorv == "dxv")
    need_dy = (operatorv == "dyv")

    # Buffers (optionnel mais évite des reallocs)
    Phi_dx = np.zeros_like(Phi_dxhat)
    Phi_dy = np.zeros_like(Phi_dyhat)

    for T, (i0, i1, i2) in enumerate(triangles):

        A0 = points[i0]
        A1 = points[i1]
        A2 = points[i2]

        # Jacobien affine (même convention que toi)
        J = np.column_stack((A1 - A0, A2 - A0))
        detJ = abs(np.linalg.det(J))
        Jinv = np.linalg.inv(J)

        # Points physiques
        x_phys = A0[0] + xq*(A1[0]-A0[0]) + yq*(A2[0]-A0[0])
        y_phys = A0[1] + xq*(A1[1]-A0[1]) + yq*(A2[1]-A0[1])

        f_q = np.asarray(func(x_phys, y_phys), dtype=np.complex128)

        # Dérivées physiques si nécessaire
        if need_dx:
            Phi_dx[:, :] = Jinv[0, 0] * Phi_dxhat + Jinv[1, 0] * Phi_dyhat
        if need_dy:
            Phi_dy[:, :] = Jinv[0, 1] * Phi_dxhat + Jinv[1, 1] * Phi_dyhat

        # Sélection opérateur test
        if operatorv == "v":
            Phi_test = Phi_val
        elif operatorv == "dxv":
            Phi_test = Phi_dx
        elif operatorv == "dyv":
            Phi_test = Phi_dy
        else:
            raise ValueError(f"Opérateur inconnu pour v : {operatorv}")

        # Intégration locale vectorisée :
        # Floc[i] = detJ * Σ_q wq[q] * f_q[q] * Phi_test[i,q]
        weight = wq * f_q                      # (Nq,)
        Floc = detJ * (Phi_test * weight[None, :]).sum(axis=1)  # (Nloc,)

        # Insertion CG/DG via LoctoGlob
        for iloc, iglob in enumerate(LoctoGlob[T]):
            F[iglob] += Floc[iloc]

    return F



#############################################################################
###   Terme sous \int_{\Omega} f Op(v) dx    ################################
#############################################################################


def assemble_rhs_surface(mesh, ordre, f, op_f="f", op_v="v", methode="CG", domaine="all", dtype=np.complex128):
    """
    Assemble un second membre frontière :

        F_i = ∫_{∂Ω_domaine}  OP_F(f) * OP_V(v_i) ds

    avec

        op_f ∈ {"f","f.n","f.t","f.ex","f.ey"}
        op_v ∈ {"v","dnv","dtv","dxv","dyv"}

    - f peut être scalaire (si op_f="f")
    - sinon f doit être vectoriel : f(x,y) -> (fx,fy)

    Paramètres
    ----------
    mesh : meshio.Mesh
    ordre : int
    f : callable
    op_f : str   opérateur appliqué à f
    op_v : str   opérateur appliqué à v
    methode : "CG" ou "DG"
    domaine : "all" ou nom BC
    """

    # --- Pré-calcul référence (exactement comme assemble_surface)
    wq, xq, yq, Phi_val, Phi_dxhat, Phi_dyhat = precompute_face_ref(ordre, dtype=dtype)

    Nloc = (ordre + 1) * (ordre + 2) // 2

    # --- Connectivité locale/globale
    LoctoGlob = loc_to_glob_general(mesh, ordre, methode)
    Nglob = int(np.max(LoctoGlob)) + 1

    # --- Voisinage + BC
    neighbors, neighbor_faces, edges_to_triangles, reference_BC, bc_name = \
        build_neighborhood_structure_with_bc(mesh)

    points = mesh.points[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])

    Fglob = np.zeros(Nglob, dtype=dtype)

    # --- Buffers (évite realloc)
    x_phys = np.empty_like(xq[0])
    y_phys = np.empty_like(yq[0])

    Phi_dx = np.empty((Nloc, len(wq)), dtype=dtype)
    Phi_dy = np.empty((Nloc, len(wq)), dtype=dtype)
    Phi_dn = np.empty((Nloc, len(wq)), dtype=dtype)
    Phi_dt = np.empty((Nloc, len(wq)), dtype=dtype)

    for T, (i0, i1, i2) in enumerate(triangles):

        A0 = points[i0]
        A1 = points[i1]
        A2 = points[i2]

        # Jacobien affine
        J = np.column_stack((A1 - A0, A2 - A0))
        Jinv = np.linalg.inv(J)

        for F in range(3):

            code = neighbors[T, F]

            if domaine == "all":
                is_boundary = (code <= -1)
            else:
                is_boundary = (code == reference_BC(domaine))

            if not is_boundary:
                continue

            # --- longueur de face
            face_nodes = ((i1, i2), (i2, i0), (i0, i1))
            edge_vec = points[face_nodes[F][1]] - points[face_nodes[F][0]]
            edge_length = np.linalg.norm(edge_vec)

            # --- normale extérieure (ta routine validée)
            n = calcul_normale(A0, A1, A2, F)
            tx, ty = -n[1], n[0]   # tangente orientée

            # --- points physiques de quadrature
            x_phys[:] = A0[0] + xq[F] * (A1[0] - A0[0]) + yq[F] * (A2[0] - A0[0])
            y_phys[:] = A0[1] + xq[F] * (A1[1] - A0[1]) + yq[F] * (A2[1] - A0[1])

            fq = f(x_phys, y_phys)

            # --------------------------------------------------
            #   OPÉRATEUR SUR f
            # --------------------------------------------------
            if op_f == "f":
                fq_eff = np.asarray(fq, dtype=dtype)

            else:
                fx, fy = fq  # f doit être vectoriel

                if op_f == "f.n":
                    fq_eff = fx * n[0] + fy * n[1]

                elif op_f == "f.t":
                    fq_eff = fx * tx + fy * ty

                elif op_f == "f.ex":
                    fq_eff = fx

                elif op_f == "f.ey":
                    fq_eff = fy

                else:
                    raise ValueError(f"Unknown op_f '{op_f}'")

                fq_eff = np.asarray(fq_eff, dtype=dtype)

            # --------------------------------------------------
            #   DÉRIVÉES PHYSIQUES DES φ
            # --------------------------------------------------
            need_grad = op_v in ("dxv", "dyv", "dnv", "dtv")

            if need_grad:
                Phi_dx[:, :] = Jinv[0, 0] * Phi_dxhat[F] + Jinv[1, 0] * Phi_dyhat[F]
                Phi_dy[:, :] = Jinv[0, 1] * Phi_dxhat[F] + Jinv[1, 1] * Phi_dyhat[F]

            if op_v == "v":
                PhiV = Phi_val[F]

            elif op_v == "dxv":
                PhiV = Phi_dx

            elif op_v == "dyv":
                PhiV = Phi_dy

            elif op_v == "dnv":
                Phi_dn[:, :] = n[0] * Phi_dx + n[1] * Phi_dy
                PhiV = Phi_dn

            elif op_v == "dtv":
                Phi_dt[:, :] = tx * Phi_dx + ty * Phi_dy
                PhiV = Phi_dt

            else:
                raise ValueError(f"Unknown op_v '{op_v}'")

            # --------------------------------------------------
            #   ASSEMBLAGE
            # --------------------------------------------------
            W = wq * fq_eff * edge_length

            for i_loc in range(Nloc):
                ig = int(LoctoGlob[T, i_loc])
                if ig >= 0:
                    Fglob[ig] += np.sum(W * PhiV[i_loc])

    return Fglob

def make_vector_field(fx, fy):
    def fvec(x, y):
        return fx(x, y), fy(x, y)
    return fvec



def assemble_skeleton_par_element(mesh, ordre, coef,
                      operatoru="sautu", operatorv="sautv",
                      methode="DG", dtype=np.complex128):
    """
    Assemble une matrice de squelette (faces internes uniquement) :

        A_ij += ∫_{F_int} coef(x,y) * OPV(v_i) * OPU(u_j) ds

    Convention :
      - T = élément courant
      - V = élément voisin
      - nT = normale sortante de T sur la face
      - nV = normale sortante de V sur la face commune (opposée à nT)
      - t = tangente orientée suivant T : t = (-nT_y, nT_x)
        et on utilise CE MÊME t pour les opérateurs côté V quand nécessaire.

    Opérateurs supportés (pour u et v, avec u↔v) :
      - Traces : uT, uV ; vT, vV
      - Sauts : sautu = uT - uV ; sautv = vT - vV
      - Dérivées normales : dnuT, dnuV ; dnvT, dnvV
      - Saut normal : sautdnu = dnuT + dnuV ; sautdnv = dnvT + dnvV
      - Dérivées cartésiennes : dxuT, dxuV, dyuT, dyuV (et idem v)
      - Dérivées tangentielles (t orientée suivant T) :
            dtuT, dtuV ; dtvT, dtvV
            sautdtu = dtuT - dtuV ; sautdtv = dtvT - dtvV
      - Flux vectoriel scalaire*normale :
            uTnT = uT * nT (vecteur 2D), uVnV = uV * nV
            sautDGu = uT*nT + uV*nV (vecteur 2D)
        (et idem : vTnT, vVnV, sautDGv)
        Si OPV et OPU sont vectoriels, on contracte par produit scalaire (dot).

    Notes :
      - On traite chaque face interne UNE fois (T < V).
      - L’alignement des points de quadrature entre T et V est géré automatiquement
        (on inverse l’ordre des points côté V si nécessaire).
    """
    # --- Alias pratique "sautDG"
    if operatoru == "sautDG":
        operatoru = "sautDGu"
    if operatorv == "sautDG":
        operatorv = "sautDGv"
    # ------------------------------------------------------------
    # Helpers locaux
    # ------------------------------------------------------------
    def _insert_block(Mat, rows_glob, cols_glob, Mblock):
        """Insertion rapide d'un bloc (Nloc x Nloc) dans Mat."""
        rr = np.repeat(rows_glob, cols_glob.size)
        cc = np.tile(cols_glob, rows_glob.size)
        vv = Mblock.reshape(-1)
        Mat.ajout_rapide(rr, cc, vv.size, vv)

    def _op_kind(op: str) -> str:
        """Normalise les alias u/v -> 'kind' indépendant de la variable."""
        # Synonymes pratiques : on interprète sans côté comme côté T
        alias = {
            "u": "uT", "v": "vT",
            "dxu": "dxuT", "dxv": "dxvT",
            "dyu": "dyuT", "dyv": "dyvT",
            "dnu": "dnuT", "dnv": "dnvT",
            "dtu": "dtuT", "dtv": "dtvT",
        }
        op = alias.get(op, op)

        mapping = {
            # traces
            "uT": "valT", "vT": "valT",
            "uV": "valV", "vV": "valV",

            # sauts trace
            "sautu": "jump", "sautv": "jump",

            # dérivées normales
            "dnuT": "dnT", "dnvT": "dnT",
            "dnuV": "dnV", "dnvV": "dnV",
            "sautdnu": "jump_dn", "sautdnv": "jump_dn",

            # dérivées cartésiennes
            "dxuT": "dxT", "dxvT": "dxT",
            "dxuV": "dxV", "dxvV": "dxV",
            "dyuT": "dyT", "dyvT": "dyT",
            "dyuV": "dyV", "dyvV": "dyV",

            # dérivées tangentielles (t orientée suivant T)
            "dtuT": "dtT", "dtvT": "dtT",
            "dtuV": "dtV", "dtvV": "dtV",
            "sautdtu": "jump_dt", "sautdtv": "jump_dt",

            # flux vectoriel scalaire*normale (vecteur 2D)
            "uTnT": "fluxT", "vTnT": "fluxT",
            "uVnV": "fluxV", "vVnV": "fluxV",
            "sautDGu": "jump_flux", "sautDGv": "jump_flux",
        }
        if op not in mapping:
            raise ValueError(
                f"Opérateur inconnu : {op}\n"
                f"Opérateurs supportés : {sorted(mapping.keys())}"
            )
        return mapping[op]

    def _zeros_like(ref):
        return np.zeros_like(ref, dtype=dtype)

    def _build_op(op_str,
                  phiT, dxT, dyT, dnT, dtT, fluxT,
                  phiV, dxV, dyV, dnV, dtV, fluxV):
        """
        Retourne (OP_T, OP_V) où :
          OP_T multiplie les ddl de T
          OP_V multiplie les ddl de V
        Chaque OP est soit (Nloc,Nq) (scalaire) soit (2,Nloc,Nq) (vectoriel).
        """
        kind = _op_kind(op_str)

        # Références de zéros (scalaire / vectoriel)
        zS = _zeros_like(phiT)
        zV = _zeros_like(fluxT)

        if kind == "valT":       return phiT, zS
        if kind == "valV":       return zS,  phiV
        if kind == "jump":       return phiT, -phiV

        if kind == "dxT":        return dxT, zS
        if kind == "dxV":        return zS,  dxV
        if kind == "dyT":        return dyT, zS
        if kind == "dyV":        return zS,  dyV

        if kind == "dnT":        return dnT, zS
        if kind == "dnV":        return zS,  dnV
        if kind == "jump_dn":    return dnT, dnV     # + côté V (normale sortante V)

        if kind == "dtT":        return dtT, zS
        if kind == "dtV":        return zS,  dtV
        if kind == "jump_dt":    return dtT, -dtV    # t orientée suivant T

        if kind == "fluxT":      return fluxT, zV
        if kind == "fluxV":      return zV,    fluxV
        if kind == "jump_flux":  return fluxT, fluxV # uT*nT + uV*nV

        raise RuntimeError("Cas non géré (ne devrait pas arriver).")

    def _block_from_ops(opV, opU, weight, edge_length):
        """
        Construit le bloc (Nloc x Nloc) :
            edge_length * ∫ opV_i(q) * weight(q) * opU_j(q) dq
        Si opV/opU sont vectoriels (2,...), on fait un dot produit.
        """
        if opV.ndim != opU.ndim:
            raise ValueError("Incompatibilité scalaire/vectoriel entre operatorv et operatoru.")

        if opV.ndim == 2:
            # scalaire : (Nloc,Nq) @ (Nq,Nloc)
            return edge_length * (opV * weight[None, :]) @ opU.T

        if opV.ndim == 3 and opV.shape[0] == 2:
            # vectoriel 2D : somme des composantes
            M = np.zeros((opV.shape[1], opU.shape[1]), dtype=dtype)
            for k in range(2):
                M += (opV[k] * weight[None, :]) @ opU[k].T
            return edge_length * M

        raise ValueError("Format d'opérateur non supporté (attendu scalaire ou vectoriel 2D).")

    # ------------------------------------------------------------
    # Données maillage / pré-calculs référence
    # ------------------------------------------------------------
    wq, xq, yq, Phi_val, Phi_dxhat, Phi_dyhat = precompute_face_ref(ordre, dtype=dtype)
    wq = np.asarray(wq)

    triangles = np.asarray(mesh.cells_dict["triangle"])
    points = mesh.points[:, :2]
    NT = triangles.shape[0]
    Nloc = (ordre + 1) * (ordre + 2) // 2

    # Connectivité globale (CG ou DG)
    LoctoGlob = loc_to_glob_general(mesh, ordre, methode)
    Nglob = int(np.max(LoctoGlob)) + 1

    # Voisinage (faces internes)
    neighbors, neighbor_faces, edges_to_triangles = build_neighborhood_structure(triangles)

    # Nombre de faces internes (comptées une fois)
    n_faces_int = 0
    for T in range(NT):
        for F in range(3):
            V = neighbors[T, F]
            if V >= 0:
                n_faces_int += 1

    nnz_est = max(1, 4 * n_faces_int * Nloc * Nloc)
    A = COOMatrix(Nglob, Nglob, nnz_est)

    # Buffers
    Nq = wq.size
    xT = np.empty(Nq, dtype=float)
    yT = np.empty(Nq, dtype=float)
    xV = np.empty(Nq, dtype=float)
    yV = np.empty(Nq, dtype=float)

    dxT = np.empty((Nloc, Nq), dtype=dtype)
    dyT = np.empty((Nloc, Nq), dtype=dtype)
    dxV = np.empty((Nloc, Nq), dtype=dtype)
    dyV = np.empty((Nloc, Nq), dtype=dtype)

    # ------------------------------------------------------------
    # Boucle faces internes
    # ------------------------------------------------------------
    face_nodes = ((1, 2), (2, 0), (0, 1))  # en indices locaux de sommets

    for T in range(NT):
        i0, i1, i2 = triangles[T]
        A0 = points[i0]
        A1 = points[i1]
        A2 = points[i2]

        JT = np.column_stack((A1 - A0, A2 - A0))
        JinvT = np.linalg.inv(JT)

        for F in range(3):
            V = neighbors[T, F]
            if V < 0:
                continue

            FV = neighbor_faces[T, F]
            j0, j1, j2 = triangles[V]
            B0 = points[j0]
            B1 = points[j1]
            B2 = points[j2]

            JV = np.column_stack((B1 - B0, B2 - B0))
            JinvV = np.linalg.inv(JV)

            # Longueur physique de la face (prise côté T)
            a_loc, b_loc = face_nodes[F]
            nodes_T = (triangles[T][a_loc], triangles[T][b_loc])
            edge_vec = points[nodes_T[1]] - points[nodes_T[0]]
            edge_length = np.linalg.norm(edge_vec)

            # Normales sortantes
            nT = calcul_normale(A0, A1, A2, F)
            nV_geom = calcul_normale(B0, B1, B2, FV)
            # on force nV opposée à nT (robuste)
            nV = -nV_geom if (nV_geom[0]*nT[0] + nV_geom[1]*nT[1]) > 0 else nV_geom

            # Tangente orientée suivant T
            t = np.array([-nT[1], nT[0]], dtype=float)

            # Points physiques côté T (référence face F)
            xT[:] = A0[0] + xq[F, :] * (A1[0] - A0[0]) + yq[F, :] * (A2[0] - A0[0])
            yT[:] = A0[1] + xq[F, :] * (A1[1] - A0[1]) + yq[F, :] * (A2[1] - A0[1])

            # Points physiques côté V (référence face FV) pour détecter orientation
            xV[:] = B0[0] + xq[FV, :] * (B1[0] - B0[0]) + yq[FV, :] * (B2[0] - B0[0])
            yV[:] = B0[1] + xq[FV, :] * (B1[1] - B0[1]) + yq[FV, :] * (B2[1] - B0[1])

            err_same = np.max(np.abs(xT - xV)) + np.max(np.abs(yT - yV))
            err_rev  = np.max(np.abs(xT - xV[::-1])) + np.max(np.abs(yT - yV[::-1]))
            reverse_V = (err_rev < err_same)

            # Bases sur la face (référence)
            phiT = Phi_val[F, :, :]             # (Nloc,Nq)
            dphxT = Phi_dxhat[F, :, :]
            dphyT = Phi_dyhat[F, :, :]

            phiV = Phi_val[FV, :, :]
            dphxV = Phi_dxhat[FV, :, :]
            dphyV = Phi_dyhat[FV, :, :]

            if reverse_V:
                phiV = phiV[:, ::-1]
                dphxV = dphxV[:, ::-1]
                dphyV = dphyV[:, ::-1]

            # Gradients physiques
            dxT[:, :] = JinvT[0, 0] * dphxT + JinvT[1, 0] * dphyT
            dyT[:, :] = JinvT[0, 1] * dphxT + JinvT[1, 1] * dphyT

            dxV[:, :] = JinvV[0, 0] * dphxV + JinvV[1, 0] * dphyV
            dyV[:, :] = JinvV[0, 1] * dphxV + JinvV[1, 1] * dphyV

            # Dérivées normales/tangentielles
            dnT = nT[0] * dxT + nT[1] * dyT
            dnV = nV[0] * dxV + nV[1] * dyV

            dtT = t[0] * dxT + t[1] * dyT
            dtV = t[0] * dxV + t[1] * dyV  # t orientée suivant T (ta convention)

            # Flux vectoriel scalaire*normale
            fluxT = np.stack((nT[0] * phiT, nT[1] * phiT), axis=0)  # (2,Nloc,Nq)
            fluxV = np.stack((nV[0] * phiV, nV[1] * phiV), axis=0)

            # Coefficient sur la face (évalué aux points xT,yT)
            if callable(coef):
                cq = np.asarray(coef(xT, yT), dtype=dtype)
            else:
                cq = np.asarray(coef, dtype=dtype)

            if cq.ndim == 0:
                cq = cq * np.ones_like(wq, dtype=dtype)

            weight = (wq.astype(dtype) * cq)  # (Nq,)

            # Opérateurs : (OP_T, OP_V)
            opu_T, opu_V = _build_op(operatoru,
                                     phiT, dxT, dyT, dnT, dtT, fluxT,
                                     phiV, dxV, dyV, dnV, dtV, fluxV)
            opv_T, opv_V = _build_op(operatorv,
                                     phiT, dxT, dyT, dnT, dtT, fluxT,
                                     phiV, dxV, dyV, dnV, dtV, fluxV)

            # 4 blocs
            M_TT = _block_from_ops(opv_T, opu_T, weight, edge_length)
            M_TV = _block_from_ops(opv_T, opu_V, weight, edge_length)
            M_VT = _block_from_ops(opv_V, opu_T, weight, edge_length)
            M_VV = _block_from_ops(opv_V, opu_V, weight, edge_length)

            # Insertion globale
            rows_T = LoctoGlob[T]
            rows_V = LoctoGlob[V]
            cols_T = LoctoGlob[T]
            cols_V = LoctoGlob[V]

            _insert_block(A, rows_T, cols_T, M_TT)
            _insert_block(A, rows_T, cols_V, M_TV)
            _insert_block(A, rows_V, cols_T, M_VT)
            _insert_block(A, rows_V, cols_V, M_VV)

    return A



def assemble_skeleton_par_face(mesh, ordre, coef,
                      operatoru="sautu", operatorv="sautv",
                      methode="DG", dtype=np.complex128):
    """
    Assemble une matrice de squelette (faces internes uniquement) :

        A_ij += ∫_{F_int} coef(x,y) * OPV(v_i) * OPU(u_j) ds

    Convention :
      - T = élément courant
      - V = élément voisin
      - nT = normale sortante de T sur la face
      - nV = normale sortante de V sur la face commune (opposée à nT)
      - t = tangente orientée suivant T : t = (-nT_y, nT_x)
        et on utilise CE MÊME t pour les opérateurs côté V quand nécessaire.

    Opérateurs supportés (pour u et v, avec u↔v) :
      - Traces : uT, uV ; vT, vV
      - Sauts : sautu = uT - uV ; sautv = vT - vV
      - Dérivées normales : dnuT, dnuV ; dnvT, dnvV
      - moyenne trace : moyu = (uT + uV)/2 ; moyv = (vT + vV)/2
      - moyenne gradient : moynablau = (∇u_T + ∇u_V)/2, moynablav = (∇v_T + ∇v_V)/2
      - Saut normal : sautdnu = dnuT + dnuV ; sautdnv = dnvT + dnvV
      - Dérivées cartésiennes : dxuT, dxuV, dyuT, dyuV (et idem v)
      - Dérivées tangentielles (t orientée suivant T) :
            dtuT, dtuV ; dtvT, dtvV
            sautdtu = dtuT - dtuV ; sautdtv = dtvT - dtvV
      - Flux vectoriel scalaire*normale :
            uTnT = uT * nT (vecteur 2D), uVnV = uV * nV
            sautDGu = uT*nT + uV*nV (vecteur 2D)
        (et idem : vTnT, vVnV, sautDGv)
        Si OPV et OPU sont vectoriels, on contracte par produit scalaire (dot).

    Notes :
      - On traite chaque face interne UNE fois (T < V).
      - L’alignement des points de quadrature entre T et V est géré automatiquement
        (on inverse l’ordre des points côté V si nécessaire).
    """
    # --- Alias pratique "sautDG"
    if operatoru == "sautDG":
        operatoru = "sautDGu"
    if operatorv == "sautDG":
        operatorv = "sautDGv"
    # ------------------------------------------------------------
    # Helpers locaux
    # ------------------------------------------------------------
    def _as_vector(dx, dy):
        out = np.empty((2, dx.shape[0], dx.shape[1]), dtype=dtype)
        out[0] = dx
        out[1] = dy
        return out
    def _insert_block(Mat, rows_glob, cols_glob, Mblock):
        """Insertion rapide d'un bloc (Nloc x Nloc) dans Mat."""
        rr = np.repeat(rows_glob, cols_glob.size)
        cc = np.tile(cols_glob, rows_glob.size)
        vv = Mblock.reshape(-1)
        Mat.ajout_rapide(rr, cc, vv.size, vv)

    def _op_kind(op: str) -> str:
        """Normalise les alias u/v -> 'kind' indépendant de la variable."""
        # Synonymes pratiques : on interprète sans côté comme côté T
        alias = {
            "u": "uT", "v": "vT",
            "dxu": "dxuT", "dxv": "dxvT",
            "dyu": "dyuT", "dyv": "dyvT",
            "dnu": "dnuT", "dnv": "dnvT",
            "dtu": "dtuT", "dtv": "dtvT",
        }
        op = alias.get(op, op)

        mapping = {
            # traces
            "uT": "valT", "vT": "valT",
            "uV": "valV", "vV": "valV",

            "moyu": "avg",
            "moyv": "avg",

            "moydxu": "avg_dx",
            "moydxv": "avg_dx",

            "moydyu": "avg_dy",
            "moydyv": "avg_dy",

            "moynablau": "avg_grad",
            "moynablav": "avg_grad",

            # sauts trace
            "sautu": "jump", "sautv": "jump",

            # dérivées normales
            "dnuT": "dnT", "dnvT": "dnT",
            "dnuV": "dnV", "dnvV": "dnV",
            "sautdnu": "jump_dn", "sautdnv": "jump_dn",

            # dérivées cartésiennes
            "dxuT": "dxT", "dxvT": "dxT",
            "dxuV": "dxV", "dxvV": "dxV",
            "dyuT": "dyT", "dyvT": "dyT",
            "dyuV": "dyV", "dyvV": "dyV",

            # dérivées tangentielles (t orientée suivant T)
            "dtuT": "dtT", "dtvT": "dtT",
            "dtuV": "dtV", "dtvV": "dtV",
            "sautdtu": "jump_dt", "sautdtv": "jump_dt",

            # flux vectoriel scalaire*normale (vecteur 2D)
            "uTnT": "fluxT", "vTnT": "fluxT",
            "uVnV": "fluxV", "vVnV": "fluxV",
            "sautDGu": "jump_flux", "sautDGv": "jump_flux",
        }
        if op not in mapping:
            raise ValueError(
                f"Opérateur inconnu : {op}\n"
                f"Opérateurs supportés : {sorted(mapping.keys())}"
            )
        return mapping[op]

    def _zeros_like(ref):
        return np.zeros_like(ref, dtype=dtype)

    def _build_op(op_str,
                  phiT, dxT, dyT, dnT, dtT, fluxT,
                  phiV, dxV, dyV, dnV, dtV, fluxV):
        """
        Retourne (OP_T, OP_V) où :
          OP_T multiplie les ddl de T
          OP_V multiplie les ddl de V
        Chaque OP est soit (Nloc,Nq) (scalaire) soit (2,Nloc,Nq) (vectoriel).
        """
        kind = _op_kind(op_str)

        # Références de zéros (scalaire / vectoriel)
        zS = _zeros_like(phiT)
        zV = _zeros_like(fluxT)

        if kind == "valT":       return phiT, zS
        if kind == "valV":       return zS,  phiV
        if kind == "jump":       return phiT, -phiV

        if kind == "dxT":        return dxT, zS
        if kind == "dxV":        return zS,  dxV
        if kind == "dyT":        return dyT, zS
        if kind == "dyV":        return zS,  dyV

        if kind == "dnT":        return dnT, zS
        if kind == "dnV":        return zS,  dnV
        if kind == "jump_dn":    return dnT, dnV     # + côté V (normale sortante V)

        if kind == "dtT":        return dtT, zS
        if kind == "dtV":        return zS,  dtV
        if kind == "jump_dt":    return dtT, -dtV    # t orientée suivant T

        if kind == "fluxT":      return fluxT, zV
        if kind == "fluxV":      return zV,    fluxV
        if kind == "jump_flux":  return fluxT, fluxV # uT*nT + uV*nV
        if kind == "avg":        return 0.5 * phiT, 0.5 * phiV
        if kind == "avg_dx":     return 0.5 * dxT, 0.5 * dxV
        if kind == "avg_dy":     return 0.5 * dyT, 0.5 * dyV
        if kind == "avg_grad":   return 0.5 * _as_vector(dxT, dyT), 0.5 * _as_vector(dxV, dyV)

        raise RuntimeError("Cas non géré (ne devrait pas arriver).")

    def _block_from_ops(opV, opU, weight, edge_length):
        """
        Construit le bloc (Nloc x Nloc) :
            edge_length * ∫ opV_i(q) * weight(q) * opU_j(q) dq
        Si opV/opU sont vectoriels (2,...), on fait un dot produit.
        """
        if opV.ndim != opU.ndim:
            raise ValueError("Incompatibilité scalaire/vectoriel entre operatorv et operatoru.")

        if opV.ndim == 2:
            # scalaire : (Nloc,Nq) @ (Nq,Nloc)
            return edge_length * (opV * weight[None, :]) @ opU.T

        if opV.ndim == 3 and opV.shape[0] == 2:
            # vectoriel 2D : somme des composantes
            M = np.zeros((opV.shape[1], opU.shape[1]), dtype=dtype)
            for k in range(2):
                M += (opV[k] * weight[None, :]) @ opU[k].T
            return edge_length * M

        raise ValueError("Format d'opérateur non supporté (attendu scalaire ou vectoriel 2D).")

    # ------------------------------------------------------------
    # Données maillage / pré-calculs référence
    # ------------------------------------------------------------
    wq, xq, yq, Phi_val, Phi_dxhat, Phi_dyhat = precompute_face_ref(ordre, dtype=dtype)
    wq = np.asarray(wq)

    triangles = np.asarray(mesh.cells_dict["triangle"])
    points = mesh.points[:, :2]
    NT = triangles.shape[0]
    Nloc = (ordre + 1) * (ordre + 2) // 2

    # Connectivité globale (CG ou DG)
    LoctoGlob = loc_to_glob_general(mesh, ordre, methode)
    Nglob = int(np.max(LoctoGlob)) + 1

    # Voisinage (faces internes)
    neighbors, neighbor_faces, edges_to_triangles = build_neighborhood_structure(triangles)

    # Nombre de faces internes (comptées une fois)
    n_faces_int = 0
    for T in range(NT):
        for F in range(3):
            V = neighbors[T, F]
            if V >= 0 and T < V:  # on compte chaque face interne une fois (T < V)
                n_faces_int += 1

    nnz_est = max(1, 4 * n_faces_int * Nloc * Nloc)
    A = COOMatrix(Nglob, Nglob, nnz_est)

    # Buffers
    Nq = wq.size
    xT = np.empty(Nq, dtype=float)
    yT = np.empty(Nq, dtype=float)
    xV = np.empty(Nq, dtype=float)
    yV = np.empty(Nq, dtype=float)

    dxT = np.empty((Nloc, Nq), dtype=dtype)
    dyT = np.empty((Nloc, Nq), dtype=dtype)
    dxV = np.empty((Nloc, Nq), dtype=dtype)
    dyV = np.empty((Nloc, Nq), dtype=dtype)

    # ------------------------------------------------------------
    # Boucle faces internes
    # ------------------------------------------------------------
    face_nodes = ((1, 2), (2, 0), (0, 1))  # en indices locaux de sommets

    for T in range(NT):
        i0, i1, i2 = triangles[T]
        A0 = points[i0]
        A1 = points[i1]
        A2 = points[i2]

        JT = np.column_stack((A1 - A0, A2 - A0))
        JinvT = np.linalg.inv(JT)

        for F in range(3):
            V = neighbors[T, F]
            if V < 0:
                continue
            if T > V:
                continue  # on traite chaque face interne une fois (T < V)

            FV = neighbor_faces[T, F]
            j0, j1, j2 = triangles[V]
            B0 = points[j0]
            B1 = points[j1]
            B2 = points[j2]

            JV = np.column_stack((B1 - B0, B2 - B0))
            JinvV = np.linalg.inv(JV)

            # Longueur physique de la face (prise côté T)
            a_loc, b_loc = face_nodes[F]
            nodes_T = (triangles[T][a_loc], triangles[T][b_loc])
            edge_vec = points[nodes_T[1]] - points[nodes_T[0]]
            edge_length = np.linalg.norm(edge_vec)

            # Normales sortantes
            nT = calcul_normale(A0, A1, A2, F)
            nV_geom = calcul_normale(B0, B1, B2, FV)
            # on force nV opposée à nT (robuste)
            nV = -nV_geom if (nV_geom[0]*nT[0] + nV_geom[1]*nT[1]) > 0 else nV_geom

            # Tangente orientée suivant T
            t = np.array([-nT[1], nT[0]], dtype=float)

            # Points physiques côté T (référence face F)
            xT[:] = A0[0] + xq[F, :] * (A1[0] - A0[0]) + yq[F, :] * (A2[0] - A0[0])
            yT[:] = A0[1] + xq[F, :] * (A1[1] - A0[1]) + yq[F, :] * (A2[1] - A0[1])

            # Points physiques côté V (référence face FV) pour détecter orientation
            xV[:] = B0[0] + xq[FV, :] * (B1[0] - B0[0]) + yq[FV, :] * (B2[0] - B0[0])
            yV[:] = B0[1] + xq[FV, :] * (B1[1] - B0[1]) + yq[FV, :] * (B2[1] - B0[1])

            err_same = np.max(np.abs(xT - xV)) + np.max(np.abs(yT - yV))
            err_rev  = np.max(np.abs(xT - xV[::-1])) + np.max(np.abs(yT - yV[::-1]))
            reverse_V = (err_rev < err_same)

            # Bases sur la face (référence)
            phiT = Phi_val[F, :, :]             # (Nloc,Nq)
            dphxT = Phi_dxhat[F, :, :]
            dphyT = Phi_dyhat[F, :, :]

            phiV = Phi_val[FV, :, :]
            dphxV = Phi_dxhat[FV, :, :]
            dphyV = Phi_dyhat[FV, :, :]

            if reverse_V:
                phiV = phiV[:, ::-1]
                dphxV = dphxV[:, ::-1]
                dphyV = dphyV[:, ::-1]

            # Gradients physiques
            dxT[:, :] = JinvT[0, 0] * dphxT + JinvT[1, 0] * dphyT
            dyT[:, :] = JinvT[0, 1] * dphxT + JinvT[1, 1] * dphyT

            dxV[:, :] = JinvV[0, 0] * dphxV + JinvV[1, 0] * dphyV
            dyV[:, :] = JinvV[0, 1] * dphxV + JinvV[1, 1] * dphyV

            # Dérivées normales/tangentielles
            dnT = nT[0] * dxT + nT[1] * dyT
            dnV = nV[0] * dxV + nV[1] * dyV

            dtT = t[0] * dxT + t[1] * dyT
            dtV = t[0] * dxV + t[1] * dyV  # t orientée suivant T (ta convention)

            # Flux vectoriel scalaire*normale
            fluxT = np.stack((nT[0] * phiT, nT[1] * phiT), axis=0)  # (2,Nloc,Nq)
            fluxV = np.stack((nV[0] * phiV, nV[1] * phiV), axis=0)

            # Coefficient sur la face (évalué aux points xT,yT)
            if callable(coef):
                cq = np.asarray(coef(xT, yT), dtype=dtype)
            else:
                cq = np.asarray(coef, dtype=dtype)

            if cq.ndim == 0:
                cq = cq * np.ones_like(wq, dtype=dtype)

            weight = (wq.astype(dtype) * cq)  # (Nq,)

            # Opérateurs : (OP_T, OP_V)
            opu_T, opu_V = _build_op(operatoru,
                                     phiT, dxT, dyT, dnT, dtT, fluxT,
                                     phiV, dxV, dyV, dnV, dtV, fluxV)
            opv_T, opv_V = _build_op(operatorv,
                                     phiT, dxT, dyT, dnT, dtT, fluxT,
                                     phiV, dxV, dyV, dnV, dtV, fluxV)

            # 4 blocs
            M_TT = _block_from_ops(opv_T, opu_T, weight, edge_length)
            M_TV = _block_from_ops(opv_T, opu_V, weight, edge_length)
            M_VT = _block_from_ops(opv_V, opu_T, weight, edge_length)
            M_VV = _block_from_ops(opv_V, opu_V, weight, edge_length)

            # Insertion globale
            rows_T = LoctoGlob[T]
            rows_V = LoctoGlob[V]
            cols_T = LoctoGlob[T]
            cols_V = LoctoGlob[V]

            _insert_block(A, rows_T, cols_T, M_TT)
            _insert_block(A, rows_T, cols_V, M_TV)
            _insert_block(A, rows_V, cols_T, M_VT)
            _insert_block(A, rows_V, cols_V, M_VV)

    return A


#################################################################
### COPIE COLLE POUR FUTURE DEV #################################
#################################################################

def assemble_skeleton_par_element_2(mesh, ordre, coef,
                      operatoru="sautu", operatorv="sautv",
                      methode="DG", dtype=np.complex128):
    """
    Assemble une matrice de squelette (faces internes uniquement) :

        A_ij += ∫_{F_int} coef(x,y) * OPV(v_i) * OPU(u_j) ds

    Convention :
      - T = élément courant
      - V = élément voisin
      - nT = normale sortante de T sur la face
      - nV = normale sortante de V sur la face commune (opposée à nT)
      - t  = tangente orientée suivant T : t = (-nT_y, nT_x)
        et on utilise CE MÊME t pour les opérateurs côté V quand nécessaire.

    Opérateurs supportés (pour u et v, avec u↔v) :
      - Traces : uT, uV ; vT, vV
      - Sauts : sautu = uT - uV ; sautv = vT - vV
      - Dérivées normales : dnuT, dnuV ; dnvT, dnvV
      - Saut normal : sautdnu = dnuT + dnuV ; sautdnv = dnvT + dnvV
      - Dérivées cartésiennes : dxuT, dxuV, dyuT, dyuV (et idem v)
      - Dérivées tangentielles (t orientée suivant T) :
            dtuT, dtuV ; dtvT, dtvV
            sautdtu = dtuT - dtuV ; sautdtv = dtvT - dtvV
      - Flux vectoriel scalaire*normale :
            uTnT = uT * nT (vecteur 2D), uVnV = uV * nV
            sautDGu = uT*nT + uV*nV (vecteur 2D)
        (et idem : vTnT, vVnV, sautDGv)
      - Composantes normales (scalaires) :
            uTnx = uT * nT_x,  uTny = uT * nT_y
            uVnx = uV * nV_x,  uVny = uV * nV_y
        (et idem vTnx, vTny, vVnx, vVny)
        Alias acceptés : u_Tnx, u_Vnx, u_Tny, u_Vny (idem pour v)
      - Flux matriciel (vecteur 2D) :
            (M.n)uT = (M nT) * uT   ;   (M.n)uV = (M nV) * uV
        Alias acceptés : (M.n)u_T, (M.n)u_V (idem pour v)

    IMPORTANT :
      - Si operatoru/operatorv utilise (M.n)…, alors `coef` est interprété comme
        un champ matriciel 2x2 (constant ou évalué aux points de quadrature).
        Dans ce cas, on n’applique PAS un coefficient scalaire supplémentaire.
        (Si tu veux un scalaire *et* M, multiplie-le directement dans M.)
      - Si OPV et OPU sont vectoriels, on contracte par produit scalaire (dot).

    Notes :
      - L’alignement des points de quadrature entre T et V est géré automatiquement
        (on inverse l’ordre des points côté V si nécessaire).
    """
    # --- Alias pratique "sautDG"
    if operatoru == "sautDG":
        operatoru = "sautDGu"
    if operatorv == "sautDG":
        operatorv = "sautDGv"

    # ------------------------------------------------------------
    # Helpers locaux
    # ------------------------------------------------------------
    def _insert_block(Mat, rows_glob, cols_glob, Mblock):
        """Insertion rapide d'un bloc (Nloc x Nloc) dans Mat."""
        rr = np.repeat(rows_glob, cols_glob.size)
        cc = np.tile(cols_glob, rows_glob.size)
        vv = Mblock.reshape(-1)
        Mat.ajout_rapide(rr, cc, vv.size, vv)

    def _zeros_like(ref):
        return np.zeros_like(ref, dtype=dtype)

    def _normalize_op(op: str) -> str:
        """Normalise quelques alias utilisateur (underscore, espaces)."""
        op = op.strip().replace(" ", "")
        # u_T... -> uT..., u_V... -> uV... (idem v)
        op = (op.replace("u_T", "uT").replace("u_V", "uV")
                .replace("v_T", "vT").replace("v_V", "vV"))
        return op

    def _op_kind(op: str) -> str:
        """Normalise les alias u/v -> 'kind' indépendant de la variable."""
        op = _normalize_op(op)

        # Synonymes pratiques : on interprète sans côté comme côté T
        alias = {
            "u": "uT", "v": "vT",
            "dxu": "dxuT", "dxv": "dxvT",
            "dyu": "dyuT", "dyv": "dyvT",
            "dnu": "dnuT", "dnv": "dnvT",
            "dtu": "dtuT", "dtv": "dtvT",
        }
        op = alias.get(op, op)

        mapping = {
            # traces
            "uT": "valT", "vT": "valT",
            "uV": "valV", "vV": "valV",

            # sauts trace
            "sautu": "jump", "sautv": "jump",

            # dérivées normales
            "dnuT": "dnT", "dnvT": "dnT",
            "dnuV": "dnV", "dnvV": "dnV",
            "sautdnu": "jump_dn", "sautdnv": "jump_dn",

            # dérivées cartésiennes
            "dxuT": "dxT", "dxvT": "dxT",
            "dxuV": "dxV", "dxvV": "dxV",
            "dyuT": "dyT", "dyvT": "dyT",
            "dyuV": "dyV", "dyvV": "dyV",

            # dérivées tangentielles (t orientée suivant T)
            "dtuT": "dtT", "dtvT": "dtT",
            "dtuV": "dtV", "dtvV": "dtV",
            "sautdtu": "jump_dt", "sautdtv": "jump_dt",

            # flux vectoriel scalaire*normale (vecteur 2D)
            "uTnT": "fluxT", "vTnT": "fluxT",
            "uVnV": "fluxV", "vVnV": "fluxV",
            "sautDGu": "jump_flux", "sautDGv": "jump_flux",

            # composantes normales (scalaires)
            "uTnx": "flux_xT", "vTnx": "flux_xT",
            "uTny": "flux_yT", "vTny": "flux_yT",
            "uVnx": "flux_xV", "vVnx": "flux_xV",
            "uVny": "flux_yV", "vVny": "flux_yV",

            # flux matriciel (vecteur 2D)
            "(M.n)uT": "Mn_fluxT", "(M.n)vT": "Mn_fluxT",
            "(M.n)uV": "Mn_fluxV", "(M.n)vV": "Mn_fluxV",
            # alias plus "compact" si besoin
            "MnuT": "Mn_fluxT", "MnvT": "Mn_fluxT",
            "MnuV": "Mn_fluxV", "MnvV": "Mn_fluxV",
        }

        if op not in mapping:
            raise ValueError(
                f"Opérateur inconnu : {op}\n"
                f"Opérateurs supportés : {sorted(mapping.keys())}"
            )
        return mapping[op]

    def _as_matrix_field(val, Nq: int):
        """
        Convertit `val` en champ (2,2,Nq).
        Formats acceptés :
          - (2,2) constant
          - (2,2,Nq)
          - (Nq,2,2)
        """
        M = np.asarray(val, dtype=dtype)
        if M.shape == (2, 2):
            M = np.repeat(M[:, :, None], Nq, axis=2)
        elif M.shape == (2, 2, Nq):
            pass
        elif M.shape == (Nq, 2, 2):
            M = np.moveaxis(M, 0, -1)  # -> (2,2,Nq)
        else:
            raise ValueError(
                f"Pour un opérateur (M.n)…, `coef` doit être un champ 2x2. "
                f"Formes acceptées: (2,2), (2,2,Nq), (Nq,2,2). Reçu: {M.shape}"
            )
        return M

    def _matvec_2x2(M_2x2xNq, n_vec):
        """(2,2,Nq) @ (2,) -> (2,Nq)."""
        nx, ny = float(n_vec[0]), float(n_vec[1])
        out0 = M_2x2xNq[0, 0, :] * nx + M_2x2xNq[0, 1, :] * ny
        out1 = M_2x2xNq[1, 0, :] * nx + M_2x2xNq[1, 1, :] * ny
        return np.stack((out0, out1), axis=0)

    def _build_op(op_str,
                  phiT, dxT, dyT, dnT, dtT, fluxT,
                  phiV, dxV, dyV, dnV, dtV, fluxV,
                  MnfluxT=None, MnfluxV=None):
        """
        Retourne (OP_T, OP_V) où :
          OP_T multiplie les ddl de T
          OP_V multiplie les ddl de V
        Chaque OP est soit (Nloc,Nq) (scalaire) soit (2,Nloc,Nq) (vectoriel).
        """
        kind = _op_kind(op_str)

        # Références de zéros (scalaire / vectoriel)
        zS = _zeros_like(phiT)
        zV = _zeros_like(fluxT)

        # -------------------------
        # Traces / sauts
        # -------------------------
        if kind == "valT":   return phiT, zS
        if kind == "valV":   return zS,  phiV
        if kind == "jump":   return phiT, -phiV

        # -------------------------
        # Dérivées cartésiennes
        # -------------------------
        if kind == "dxT":    return dxT, zS
        if kind == "dxV":    return zS,  dxV
        if kind == "dyT":    return dyT, zS
        if kind == "dyV":    return zS,  dyV

        # -------------------------
        # Dérivées normales/tangentielles
        # -------------------------
        if kind == "dnT":        return dnT, zS
        if kind == "dnV":        return zS,  dnV
        if kind == "jump_dn":    return dnT, dnV     # + côté V (normale sortante V)

        if kind == "dtT":        return dtT, zS
        if kind == "dtV":        return zS,  dtV
        if kind == "jump_dt":    return dtT, -dtV    # t orientée suivant T

        # -------------------------
        # Flux n*u (vectoriel) + composantes (scalaires)
        # -------------------------
        if kind == "fluxT":      return fluxT, zV
        if kind == "fluxV":      return zV,    fluxV
        if kind == "jump_flux":  return fluxT, fluxV  # uT*nT + uV*nV

        if kind == "flux_xT":    return fluxT[0], zS
        if kind == "flux_yT":    return fluxT[1], zS
        if kind == "flux_xV":    return zS, fluxV[0]
        if kind == "flux_yV":    return zS, fluxV[1]

        # -------------------------
        # Flux (M n) * u (vectoriel)
        # -------------------------
        if kind == "Mn_fluxT":
            if MnfluxT is None:
                raise ValueError("Opérateur (M.n)… demandé mais MnfluxT n'est pas fourni.")
            return MnfluxT, _zeros_like(MnfluxT)
        if kind == "Mn_fluxV":
            if MnfluxV is None:
                raise ValueError("Opérateur (M.n)… demandé mais MnfluxV n'est pas fourni.")
            return _zeros_like(MnfluxV), MnfluxV

        raise RuntimeError("Cas non géré (ne devrait pas arriver).")

    def _block_from_ops(opV, opU, weight, edge_length):
        """
        Construit le bloc (Nloc x Nloc) :
            edge_length * ∫ opV_i(q) * weight(q) * opU_j(q) dq
        Si opV/opU sont vectoriels (2,...), on fait un dot produit.
        """
        if opV.ndim != opU.ndim:
            raise ValueError("Incompatibilité scalaire/vectoriel entre operatorv et operatoru.")

        if opV.ndim == 2:
            # scalaire : (Nloc,Nq) @ (Nq,Nloc)
            return edge_length * (opV * weight[None, :]) @ opU.T

        if opV.ndim == 3 and opV.shape[0] == 2:
            # vectoriel 2D : somme des composantes
            M = np.zeros((opV.shape[1], opU.shape[1]), dtype=dtype)
            for k in range(2):
                M += (opV[k] * weight[None, :]) @ opU[k].T
            return edge_length * M

        raise ValueError("Format d'opérateur non supporté (attendu scalaire ou vectoriel 2D).")

    # ------------------------------------------------------------
    # Détection globale : besoin du champ matriciel M ?
    # ------------------------------------------------------------
    kind_u_global = _op_kind(operatoru)
    kind_v_global = _op_kind(operatorv)
    need_Mn = (kind_u_global.startswith("Mn_") or kind_v_global.startswith("Mn_"))

    # ------------------------------------------------------------
    # Données maillage / pré-calculs référence
    # ------------------------------------------------------------
    wq, xq, yq, Phi_val, Phi_dxhat, Phi_dyhat = precompute_face_ref(ordre, dtype=dtype)
    wq = np.asarray(wq)

    triangles = np.asarray(mesh.cells_dict["triangle"])
    points = mesh.points[:, :2]
    NT = triangles.shape[0]
    Nloc = (ordre + 1) * (ordre + 2) // 2

    # Connectivité globale (CG ou DG)
    LoctoGlob = loc_to_glob_general(mesh, ordre, methode)
    Nglob = int(np.max(LoctoGlob)) + 1

    # Voisinage (faces internes)
    neighbors, neighbor_faces, edges_to_triangles = build_neighborhood_structure(triangles)

    # Estimation nnz (comme avant : large, mais sûr)
    n_faces_int = 0
    for T in range(NT):
        for F in range(3):
            V = neighbors[T, F]
            if V >= 0:
                n_faces_int += 1

    nnz_est = max(1, 4 * n_faces_int * Nloc * Nloc)
    A = COOMatrix(Nglob, Nglob, nnz_est)

    # Buffers
    Nq = wq.size
    xT = np.empty(Nq, dtype=float)
    yT = np.empty(Nq, dtype=float)
    xV = np.empty(Nq, dtype=float)
    yV = np.empty(Nq, dtype=float)

    dxT = np.empty((Nloc, Nq), dtype=dtype)
    dyT = np.empty((Nloc, Nq), dtype=dtype)
    dxV = np.empty((Nloc, Nq), dtype=dtype)
    dyV = np.empty((Nloc, Nq), dtype=dtype)

    # ------------------------------------------------------------
    # Boucle faces internes
    # ------------------------------------------------------------
    face_nodes = ((1, 2), (2, 0), (0, 1))  # indices locaux des sommets de face

    for T in range(NT):
        i0, i1, i2 = triangles[T]
        A0 = points[i0]
        A1 = points[i1]
        A2 = points[i2]

        JT = np.column_stack((A1 - A0, A2 - A0))
        JinvT = np.linalg.inv(JT)

        for F in range(3):
            V = neighbors[T, F]
            if V < 0:
                continue

            FV = neighbor_faces[T, F]
            j0, j1, j2 = triangles[V]
            B0 = points[j0]
            B1 = points[j1]
            B2 = points[j2]

            JV = np.column_stack((B1 - B0, B2 - B0))
            JinvV = np.linalg.inv(JV)

            # Longueur physique de la face (prise côté T)
            a_loc, b_loc = face_nodes[F]
            nodes_T = (triangles[T][a_loc], triangles[T][b_loc])
            edge_vec = points[nodes_T[1]] - points[nodes_T[0]]
            edge_length = np.linalg.norm(edge_vec)

            # Normales sortantes
            nT = calcul_normale(A0, A1, A2, F)
            nV_geom = calcul_normale(B0, B1, B2, FV)
            # on force nV opposée à nT (robuste)
            nV = -nV_geom if (nV_geom[0]*nT[0] + nV_geom[1]*nT[1]) > 0 else nV_geom

            # Tangente orientée suivant T
            t = np.array([-nT[1], nT[0]], dtype=float)

            # Points physiques côté T (référence face F)
            xT[:] = A0[0] + xq[F, :] * (A1[0] - A0[0]) + yq[F, :] * (A2[0] - A0[0])
            yT[:] = A0[1] + xq[F, :] * (A1[1] - A0[1]) + yq[F, :] * (A2[1] - A0[1])

            # Points physiques côté V (référence face FV) pour détecter orientation
            xV[:] = B0[0] + xq[FV, :] * (B1[0] - B0[0]) + yq[FV, :] * (B2[0] - B0[0])
            yV[:] = B0[1] + xq[FV, :] * (B1[1] - B0[1]) + yq[FV, :] * (B2[1] - B0[1])

            err_same = np.max(np.abs(xT - xV)) + np.max(np.abs(yT - yV))
            err_rev  = np.max(np.abs(xT - xV[::-1])) + np.max(np.abs(yT - yV[::-1]))
            reverse_V = (err_rev < err_same)

            # Bases sur la face (référence)
            phiT = Phi_val[F, :, :]             # (Nloc,Nq)
            dphxT = Phi_dxhat[F, :, :]
            dphyT = Phi_dyhat[F, :, :]

            phiV = Phi_val[FV, :, :]
            dphxV = Phi_dxhat[FV, :, :]
            dphyV = Phi_dyhat[FV, :, :]

            if reverse_V:
                phiV  = phiV[:, ::-1]
                dphxV = dphxV[:, ::-1]
                dphyV = dphyV[:, ::-1]

            # Gradients physiques
            dxT[:, :] = JinvT[0, 0] * dphxT + JinvT[1, 0] * dphyT
            dyT[:, :] = JinvT[0, 1] * dphxT + JinvT[1, 1] * dphyT

            dxV[:, :] = JinvV[0, 0] * dphxV + JinvV[1, 0] * dphyV
            dyV[:, :] = JinvV[0, 1] * dphxV + JinvV[1, 1] * dphyV

            # Dérivées normales/tangentielles
            dnT = nT[0] * dxT + nT[1] * dyT
            dnV = nV[0] * dxV + nV[1] * dyV

            dtT = t[0] * dxT + t[1] * dyT
            dtV = t[0] * dxV + t[1] * dyV  # t orientée suivant T

            # Flux vectoriel scalaire*normale (n*u)
            fluxT = np.stack((nT[0] * phiT, nT[1] * phiT), axis=0)  # (2,Nloc,Nq)
            fluxV = np.stack((nV[0] * phiV, nV[1] * phiV), axis=0)

            # -----------------------------
            # Coefficient / poids
            # -----------------------------
            if need_Mn:
                # Ici `coef` est M(x,y) (2x2) : on l'injecte dans Mnflux, et le poids reste wq
                if callable(coef):
                    Mraw = coef(xT, yT)
                else:
                    Mraw = coef
                Mq = _as_matrix_field(Mraw, Nq)           # (2,2,Nq)
                MnT = _matvec_2x2(Mq, nT)                 # (2,Nq)
                MnV = _matvec_2x2(Mq, nV)                 # (2,Nq)
                MnfluxT = np.stack((MnT[0] * phiT, MnT[1] * phiT), axis=0)  # (2,Nloc,Nq)
                MnfluxV = np.stack((MnV[0] * phiV, MnV[1] * phiV), axis=0)
                weight = wq.astype(dtype)                 # (Nq,)
            else:
                # Cas standard : coef scalaire sur la face
                if callable(coef):
                    cq = np.asarray(coef(xT, yT), dtype=dtype)
                else:
                    cq = np.asarray(coef, dtype=dtype)

                if cq.ndim == 0:
                    cq = cq * np.ones_like(wq, dtype=dtype)

                weight = (wq.astype(dtype) * cq)          # (Nq,)
                MnfluxT = None
                MnfluxV = None

            # -----------------------------
            # Opérateurs : (OP_T, OP_V)
            # -----------------------------
            opu_T, opu_V = _build_op(operatoru,
                                     phiT, dxT, dyT, dnT, dtT, fluxT,
                                     phiV, dxV, dyV, dnV, dtV, fluxV,
                                     MnfluxT=MnfluxT, MnfluxV=MnfluxV)
            opv_T, opv_V = _build_op(operatorv,
                                     phiT, dxT, dyT, dnT, dtT, fluxT,
                                     phiV, dxV, dyV, dnV, dtV, fluxV,
                                     MnfluxT=MnfluxT, MnfluxV=MnfluxV)

            # 4 blocs
            M_TT = _block_from_ops(opv_T, opu_T, weight, edge_length)
            M_TV = _block_from_ops(opv_T, opu_V, weight, edge_length)
            M_VT = _block_from_ops(opv_V, opu_T, weight, edge_length)
            M_VV = _block_from_ops(opv_V, opu_V, weight, edge_length)

            # Insertion globale
            rows_T = LoctoGlob[T]
            rows_V = LoctoGlob[V]
            cols_T = LoctoGlob[T]
            cols_V = LoctoGlob[V]

            _insert_block(A, rows_T, cols_T, M_TT)
            _insert_block(A, rows_T, cols_V, M_TV)
            _insert_block(A, rows_V, cols_T, M_VT)
            _insert_block(A, rows_V, cols_V, M_VV)

    return A