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

    x_phys = np.empty_like(xq[0, :])
    y_phys = np.empty_like(yq[0, :])
    Phi_dx = np.empty_like(Phi_dxhat[0,:,:])
    Phi_dy = np.empty_like(Phi_dyhat[0,:,:])
    Phi_dn = np.empty_like(Phi_dxhat[0,:,:])
    Phi_dt = np.empty_like(Phi_dxhat[0,:,:])    
    Phi_u = np.empty_like(Phi_dxhat[0,:,:])    
    Phi_v = np.empty_like(Phi_dxhat[0,:,:])    
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

                f_q = np.asarray(func(x_phys, y_phys), dtype=np.complex128)


                
                # Dérivées physiques (si nécessaires)
                if need_dx:
                    Phi_dx[:, :] = Jinv[0, 0] * Phi_dxhat[F, :, :] + Jinv[1, 0] * Phi_dyhat[F, :, :]  # ∂/∂x
                    Phi_dy[:, :] = Jinv[0, 1] * Phi_dxhat[F, :, :] + Jinv[1, 1] * Phi_dyhat[F, :, :]  # ∂/∂y
                if need_dn:
                    normale =calcul_normale(A0, A1, A2, F)
                    Phi_dn[:,:] = normale[0]*Phi_dx + normale[1]*Phi_dy 
                if need_dt:
                    normale =calcul_normale(A0, A1, A2, F)
                    Phi_dt[:,:] = -normale[1]*Phi_dx + normale[0]*Phi_dy


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
                else:
                    raise ValueError(f"Opérateur inconnu pour v : {operatorv}")

                # Matrice locale (version “vraie brique” vectorisée)
                weight = wq * f_q                      # (Nq,)
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