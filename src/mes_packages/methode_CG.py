import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from mes_packages import (
    base,
    loc2D_to_loc1D, 
    Mref,
    build_masse_locale, 
    build_mixte_locale,
    build_rigidite_locale,
    build_neighborhood_structure,
    build_masse_ref_1D,
    base_1D,
    build_dof_coordinates_DG,
    plot_nodal_vector_DG,
    build_loctoglob_DG,
    scatter_nodal_vector_DG,
    calcul_normale,
    Kref
    )
from mes_packages.sparse import COOMatrix
from mes_packages.quadrature import (
    quadrature_triangle_ref_2D
    )


def build_loctoglob_CG(mesh, ordre:int, tol=1e-10):
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]
    """
    Construit la table de correspondance locale -> globale pour les éléments finis C0

    Parameters
    ----------
    triangles : array (NT, 3)
        Indices des sommets de chaque triangle
    points : array (N_points, 2)
        Coordonnées (x, y) des sommets du maillage
    ordre : int
        Ordre des éléments finis (>= 1)
    tol : float
        Tolérance de fusion/tri (quantification)

    Returns
    -------
    loc_to_glob : array (NT, ordre+1, ordre+1)
        Table locale -> globale : loc_to_glob[iT, iloc1, iloc2] = iglob
        Les indices invalides (hors du triangle) valent **-1**
    glob_to_xy : array (Nglob_C0, 2)
        Coordonnées des DDL globaux (arrondies à tol)
    Nglob_C0 : int
        Nombre total de DDL C0
    """
    if ordre < 1:
        raise ValueError("**L'ordre doit être au moins 1** pour les éléments finis continus.")

    NT = len(triangles)
    Nloc = (ordre + 1) * (ordre + 2) // 2

    LOC_TO_GLOB_INTERxy = np.zeros((NT * Nloc, 2), dtype=float)
    LOC_TO_GLOB_INTER   = np.zeros((NT * Nloc, 4), dtype=int)

    iglob_DG = 0
    for iT in range(NT):
        i1, i2, i3 = triangles[iT]
        A1, A2, A3 = points[i1], points[i2], points[i3]

        for iloc1 in range(ordre + 1):
            for iloc2 in range(ordre + 1 - iloc1):
                xhat = iloc1 / ordre
                yhat = iloc2 / ordre

                x = A1[0] + xhat * (A2[0] - A1[0]) + yhat * (A3[0] - A1[0])
                y = A1[1] + xhat * (A2[1] - A1[1]) + yhat * (A3[1] - A1[1])

                LOC_TO_GLOB_INTERxy[iglob_DG, 0] = x
                LOC_TO_GLOB_INTERxy[iglob_DG, 1] = y

                LOC_TO_GLOB_INTER[iglob_DG, 0] = iT
                LOC_TO_GLOB_INTER[iglob_DG, 1] = iloc1
                LOC_TO_GLOB_INTER[iglob_DG, 2] = iloc2
                iglob_DG += 1

    # Arrondi/quantification cohérente à tol (vectorisé)
    LOC_TO_GLOB_INTERxyrounded = np.rint(LOC_TO_GLOB_INTERxy / tol) * tol

    # Tri lexicographique sur les coordonnées arrondies
    IND = np.lexsort((LOC_TO_GLOB_INTERxyrounded[:, 1], LOC_TO_GLOB_INTERxyrounded[:, 0]))

    # Construction des indices globaux : même (x_rounded,y_rounded) => même iglob
    iglob = 0
    LOC_TO_GLOB_INTER[IND[0], 3] = iglob  # important

    for k in range(1, NT * Nloc):
        i  = IND[k]
        ip = IND[k - 1]

        x,  y  = LOC_TO_GLOB_INTERxyrounded[i]
        xo, yo = LOC_TO_GLOB_INTERxyrounded[ip]

        if (x != xo) or (y != yo):
            iglob += 1

        LOC_TO_GLOB_INTER[i, 3] = iglob

    Nglob_C0 = iglob + 1

    # Table locale->globale : mettre -1 pour les indices invalides
    loc_to_glob = -np.ones((NT, ordre + 1, ordre + 1), dtype=int)

    # Coordonnées des DDL globaux : on prend les coordonnées arrondies (cohérentes avec la fusion)
    glob_to_xy = np.zeros((Nglob_C0, 2), dtype=float)

    for k in range(NT * Nloc):
        iT    = LOC_TO_GLOB_INTER[k, 0]
        iloc1 = LOC_TO_GLOB_INTER[k, 1]
        iloc2 = LOC_TO_GLOB_INTER[k, 2]
        g     = LOC_TO_GLOB_INTER[k, 3]

        loc_to_glob[iT, iloc1, iloc2] = g
        glob_to_xy[g, 0] = LOC_TO_GLOB_INTERxyrounded[k, 0]
        glob_to_xy[g, 1] = LOC_TO_GLOB_INTERxyrounded[k, 1]

    return loc_to_glob, glob_to_xy, Nglob_C0


# Création d'un vecteur nodal DG à partir d'un vecteur nodal C0
def nodal_CG_to_DG(U_CG:np.ndarray, mesh,ordre:int) -> np.ndarray:
    """
    Convertit un vecteur nodal C0 en vecteur nodal DG
    
    Parameters:
    -----------
    U_CG : array (Nglob_CG,)
        Vecteur nodal C0
    mesh : Mesh
        
    Returns:
    --------
    U_DG : array (Nglob_DG,)
        Vecteur nodal DG
    """
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]   
    loctoglob_CG, glob_to_xy_CG, Nglob_CG=build_loctoglob_CG(mesh, ordre)
    loctoglob_DG, Nglob_DG = build_loctoglob_DG(triangles, ordre)

    NT = loctoglob_CG.shape[0]
    Nloc = loctoglob_DG.shape[1]
    Nglob_DG = np.max(loctoglob_DG) + 1
    
    U_DG = np.zeros(Nglob_DG,dtype=np.complex128)
    
    for iT in range(NT):
        for iloc1 in range(ordre + 1):
            for iloc2 in range(ordre + 1 - iloc1):
                iglob_CG = loctoglob_CG[iT, iloc1, iloc2]
                iloc = loc2D_to_loc1D(iloc1, iloc2)
                iglob_DG = loctoglob_DG[iT,iloc]
                U_DG[iglob_DG] = U_CG[iglob_CG]
    
    return U_DG

def scatter_nodal_vector_CG(U_CG, mesh, ordre:int, title):
    """
    Affiche un scatter plot du vecteur nodal C0
    
    Parameters:
    -----------
    U_CG : array (Nglob_CG,)
        Vecteur nodal C0
    glob_to_xy_CG : array (Nglob_C0, 2)
        Coordonnées (x, y) des DDL C0
    title : str
        Titre du graphique
    """
    loc_to_glob_CG, glob_to_xy_CG, Nglob_CG=build_loctoglob_CG(mesh, ordre)
    # Recuperation de la géométrie du maillage
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(glob_to_xy_CG[:, 0], glob_to_xy_CG[:, 1], 
                          c=U_CG, cmap='viridis', s=30)
    plt.triplot(points[:, 0], points[:, 1], triangles, 'k-', 
                alpha=0.5, linewidth=1)
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.show()


def build_masse_CG_lent(mesh, ordre:int, verbose=True):
    """
    Construit la matrice de masse globale en méthode C0 (continue)
    
    Parameters:
    -----------
    mesh : Mesh
        Maillage
    ordre : int
        Ordre des éléments finis
    verbose : bool, optional
        Afficher les statistiques
        
    Returns:
    --------
    M_CG : COOMatrix
        Matrice de masse globale CG
    """

    # Recuperation de la géométrie du maillage
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]    
    loc_to_glob_CG, glob_to_xy_CG, Nglob_CG=build_loctoglob_CG(mesh, ordre)
    n_triangles = len(triangles)
    Nloc = (ordre + 1) * (ordre + 2) // 2
    Nglob_CG = np.max(loc_to_glob_CG) + 1
    
    # Estimation du nombre d'éléments non nuls
    nnz_estime = n_triangles * Nloc * Nloc
    M_CG = COOMatrix(Nglob_CG, Nglob_CG, nnz_estime)
    
    for ielt in range(n_triangles):
        A1 = points[triangles[ielt, 0]]
        A2 = points[triangles[ielt, 1]]
        A3 = points[triangles[ielt, 2]]    
        Mloc = build_masse_locale(ordre, A1, A2, A3)
        for iloc1 in range(ordre + 1):
            for jloc1 in range(ordre + 1 - iloc1):
                i_local = loc2D_to_loc1D(iloc1, jloc1)
                iglob = loc_to_glob_CG[ielt, iloc1, jloc1]
                
                for iloc2 in range(ordre + 1):
                    for jloc2 in range(ordre + 1 - iloc2):
                        j_local = loc2D_to_loc1D(iloc2, jloc2)
                        jglob = loc_to_glob_CG[ielt, iloc2, jloc2]
                        
                        M_CG.ajout(iglob, jglob, Mloc[i_local, j_local])
    
    if verbose:
        print(f"Matrice de masse CG assemblée : {M_CG.nb_lig} x {M_CG.nb_col}")
        print(f"Éléments non nuls : {M_CG.l}")
    
    return M_CG

def build_masse_CG(mesh, ordre:int, verbose=True):
    """
    Construit la matrice de masse globale en méthode C0 (continue)
    
    Parameters:
    -----------
    mesh : Mesh
        Maillage
    ordre : int
        Ordre des éléments finis
    verbose : bool, optional
        Afficher les statistiques
        
    Returns:
    --------
    M_CG : COOMatrix
        Matrice de masse globale CG
    """

    # Recuperation de la géométrie du maillage
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]    
    loc_to_glob_CG, glob_to_xy_CG, Nglob_CG=build_loctoglob_CG(mesh, ordre)
    n_triangles = len(triangles)
    Nloc = (ordre + 1) * (ordre + 2) // 2
    Nglob_CG = np.max(loc_to_glob_CG) + 1
    
    # Estimation du nombre d'éléments non nuls
    nnz_estime = n_triangles * Nloc * Nloc
    M_CG = COOMatrix(Nglob_CG, Nglob_CG, nnz_estime)
    
    Mhat = Mref(ordre)

    for ielt in range(n_triangles):
        A1 = points[triangles[ielt, 0]]
        A2 = points[triangles[ielt, 1]]
        A3 = points[triangles[ielt, 2]]    
        # Calcul de l'aire du triangle physique
        Jac = abs((A2[0] - A1[0]) * (A3[1] - A1[1]) - (A3[0] - A1[0]) * (A2[1] - A1[1]))
        Mloc = Jac * Mhat
        # Mloc = build_masse_locale(ordre, A1, A2, A3)
        for iloc1 in range(ordre + 1):
            for jloc1 in range(ordre + 1 - iloc1):
                i_local = loc2D_to_loc1D(iloc1, jloc1)
                iglob = loc_to_glob_CG[ielt, iloc1, jloc1]
                
                for iloc2 in range(ordre + 1):
                    for jloc2 in range(ordre + 1 - iloc2):
                        j_local = loc2D_to_loc1D(iloc2, jloc2)
                        jglob = loc_to_glob_CG[ielt, iloc2, jloc2]
                        
                        M_CG.ajout(iglob, jglob, Mloc[i_local, j_local])
    
    if verbose:
        print(f"Matrice de masse CG assemblée : {M_CG.nb_lig} x {M_CG.nb_col}")
        print(f"Éléments non nuls : {M_CG.l}")
    
    return M_CG


def terme_source_CG(func, mesh, ordre:int):

    points = mesh.points[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])
    loc_to_glob_CG, glob_to_xy_CG, Nglob_CG = build_loctoglob_CG(mesh, ordre)

    F_CG = np.zeros(Nglob_CG, dtype=np.complex128)

    # Quadrature (une seule fois)
    ordreq = 2*ordre 
    wq,xq, yq = quadrature_triangle_ref_2D(ordreq+1)

    Nloc = (ordre+1)*(ordre+2)//2
    Nq = len(wq)

    # Pré-évaluation des bases
    Phi = np.zeros((Nloc, Nq))
    k = 0
    for i in range(ordre+1):
        for j in range(ordre+1-i):
            k = loc2D_to_loc1D(i, j)
            Phi[k,:] = base(xq, yq, i, j, ordre)

    for T, (pt0, pt1, pt2) in enumerate(triangles):

        A0 = points[pt0]
        A1 = points[pt1]
        A2 = points[pt2]

        # Transformation affine des points de quad
        x_phys = A0[0] + xq*(A1[0]-A0[0]) + yq*(A2[0]-A0[0])
        y_phys = A0[1] + xq*(A1[1]-A0[1]) + yq*(A2[1]-A0[1])

        f_q = func(x_phys, y_phys)

        # Jacobien (une fois)
        Jac = abs((A1[0]-A0[0])*(A2[1]-A0[1]) - (A2[0]-A0[0])*(A1[1]-A0[1]))

        # Intégrale locale vectorisée
        F_loc = Jac * (Phi @ (wq * f_q))

        # Assemblage
        for iloc in range(ordre + 1):
            for jloc in range(ordre + 1 - iloc):
                k=loc2D_to_loc1D(iloc, jloc)
                iglob = loc_to_glob_CG[T,iloc, jloc]
                F_CG[iglob] += F_loc[k]

    return F_CG




def build_nodal_vector_CG(f, mesh,ordre:int):
    """
    Construit le vecteur nodal associé à une fonction f(x,y)
    en évaluant f aux points d'interpolation (DDL) en C0
    
    Parameters:
    -----------
    f : function
        Fonction f(x, y) à évaluer
    mesh : meshio.Mesh
        Maillage
    ordre : int
        Ordre polynomial        
    Returns:
    --------
    U : array (Nglob_C0,)
        Vecteur nodal : U[iglob] = f(x_iglob, y_iglob)
    """
    _, glob_to_xy_CG,_= build_loctoglob_CG(mesh, ordre)


    n_dof_CG = len(glob_to_xy_CG)
    U = np.zeros(n_dof_CG,dtype=np.complex128)
    
    for iglob in range(n_dof_CG):
        x = glob_to_xy_CG[iglob, 0]
        y = glob_to_xy_CG[iglob, 1]
        U[iglob] = f(x, y)
    
    return U


def build_rigidite_CG_lent(mesh, ordre:int, verbose=True):
    """
    Construit la matrice de rigidité globale en méthode CG (continue)
    
    Parameters:
    -----------
    mesh : Mesh
        Maillage contenant les triangles et les points
    ordre : int
        Ordre des éléments finis
    verbose : bool, optional
        Afficher les statistiques
        
    Returns:
    --------
    K_CG : COOMatrix
        Matrice de rigidité globale CG (Laplacien)
    """
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]    
    loc_to_glob_CG, glob_to_xy_CG, Nglob_CG=build_loctoglob_CG(mesh, ordre)
    n_triangles = len(triangles)
    Nloc = (ordre + 1) * (ordre + 2) // 2
    Nglob_CG = np.max(loc_to_glob_CG) + 1

    # Estimation du nombre d'éléments non nuls
    nnz_estime = 20 * Nglob_CG * Nloc
    K_CG = COOMatrix(Nglob_CG, Nglob_CG, nnz_estime)
    # Matrices de référence
    for ielt in range(n_triangles):
        A1 = points[triangles[ielt, 0]]
        A2 = points[triangles[ielt, 1]]
        A3 = points[triangles[ielt, 2]]
        Kloc = build_rigidite_locale(ordre, A1, A2, A3, equation='laplace')
        
        for iloc1 in range(ordre + 1):
            for jloc1 in range(ordre + 1 - iloc1):
                i_local = loc2D_to_loc1D(iloc1, jloc1)
                iglob = loc_to_glob_CG[ielt, iloc1, jloc1]
                
                for iloc2 in range(ordre + 1):
                    for jloc2 in range(ordre + 1 - iloc2):
                        j_local = loc2D_to_loc1D(iloc2, jloc2)
                        jglob = loc_to_glob_CG[ielt, iloc2, jloc2]
                        
                        K_CG.ajout(iglob, jglob, Kloc[i_local, j_local])

    if verbose:
        print(f"Matrice de rigidité CG assemblée : {K_CG.nb_lig} x {K_CG.nb_col}")
        print(f"Éléments non nuls : {K_CG.l}")

    return K_CG


def build_rigidite_CG(mesh, ordre:int, verbose=True):
    """
    Construit la matrice de rigidité globale en méthode CG (continue)
    
    Parameters:
    -----------
    mesh : Mesh
        Maillage contenant les triangles et les points
    ordre : int
        Ordre des éléments finis
    verbose : bool, optional
        Afficher les statistiques
        
    Returns:
    --------
    K_CG : COOMatrix
        Matrice de rigidité globale CG (Laplacien)
    """
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]    
    loc_to_glob_CG, glob_to_xy_CG, Nglob_CG=build_loctoglob_CG(mesh, ordre)
    n_triangles = len(triangles)
    Nloc = (ordre + 1) * (ordre + 2) // 2
    Nglob_CG = np.max(loc_to_glob_CG) + 1

    # Estimation du nombre d'éléments non nuls
    nnz_estime = 20 * Nglob_CG * Nloc
    K_CG = COOMatrix(Nglob_CG, Nglob_CG, nnz_estime)
    # Matrices de référence
    Krefxx, Krefxy, Krefyx, Krefyy = Kref(ordre)
    for ielt in range(n_triangles):
        A1 = points[triangles[ielt, 0]]
        A2 = points[triangles[ielt, 1]]
        A3 = points[triangles[ielt, 2]]
        J=np.column_stack((A2 - A1, A3 - A1))
        detJ = abs(np.linalg.det(J))
        J_inv = np.linalg.inv(J)
        Klocxx = detJ * (J_inv[0, 0]**2 * Krefxx + J_inv[0, 0]*J_inv[1, 0]*(Krefxy + Krefyx) + J_inv[1, 0]**2 * Krefyy)
        Klocyy = detJ * (J_inv[0, 1]**2 * Krefxx + J_inv[0, 1]*J_inv[1, 1]*(Krefxy + Krefyx) + J_inv[1, 1]**2 * Krefyy)
        Kloc = Klocxx + Klocyy
        # Kloc = build_rigidite_locale(ordre, A1, A2, A3, equation='laplace')
        
        for iloc1 in range(ordre + 1):
            for jloc1 in range(ordre + 1 - iloc1):
                i_local = loc2D_to_loc1D(iloc1, jloc1)
                iglob = loc_to_glob_CG[ielt, iloc1, jloc1]
                
                for iloc2 in range(ordre + 1):
                    for jloc2 in range(ordre + 1 - iloc2):
                        j_local = loc2D_to_loc1D(iloc2, jloc2)
                        jglob = loc_to_glob_CG[ielt, iloc2, jloc2]
                        
                        K_CG.ajout(iglob, jglob, Kloc[i_local, j_local])

    if verbose:
        print(f"Matrice de rigidité CG assemblée : {K_CG.nb_lig} x {K_CG.nb_col}")
        print(f"Éléments non nuls : {K_CG.l}")

    return K_CG




def iface_iglob_CG(ielt, iface, iloc_face, ordre:int, loctoglob_CG):
    """
    Détermine l'indice global du iloc_face-ième DDL de la face iface du triangle ielt.
    
    Paramètres:
    -----------
    ielt : int
        Numéro du triangle
    iface : int
        Numéro de la face (0, 1, ou 2)
        - face 0 : arête opposée au sommet 0 (entre sommets 1 et 2)
        - face 1 : arête opposée au sommet 1 (entre sommets 2 et 0)
        - face 2 : arête opposée au sommet 2 (entre sommets 0 et 1)
    iloc_face : int
        Indice du point sur la face (de 0 à ordre)
    ordre : int
        Ordre polynomial
    loctoglob : ndarray
        Table de correspondance locale vers globale (Nloc x num_elements)
        
    Retourne:
    ---------
    iglob : int
        Indice global du DDL
    """
    
    if iface == 0:
        # Face 0 opposée au sommet 0: arête entre sommets 1=(ordre,0) et 2=(0,ordre)
        # Paramètrée par m+n=ordre, avec iloc_face variant de 0 à ordre
        m = ordre - iloc_face
        n = iloc_face
    elif iface == 1:
        # Face 1 opposée au sommet 1: arête entre sommets 2=(0,ordre) et 0=(0,0)
        # Paramètrée par m=0, avec n allant de ordre à 0
        m = 0
        n = ordre - iloc_face
    elif iface == 2:
        # Face 2 opposée au sommet 2: arête entre sommets 0=(0,0) et 1=(ordre,0)
        # Paramètrée par n=0, avec m allant de 0 à ordre
        m = iloc_face
        n = 0
    else:
        raise ValueError(f"iface doit être 0, 1 ou 2, pas {iface}")
    
    # Conversion indice local -> indice global
    iglob_CG = loctoglob_CG[ ielt,m,n]
    
    return iglob_CG


# def build_frontiere_CG(ordre, triangles, points, neighbors, neighbor_faces, mesh, loctoglob_CG, M_ref_1D):

def nombre_dof_CG(mesh,ordre:int) -> int:
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]    
    loc_to_glob_CG, glob_to_xy_CG, Nglob_CG=build_loctoglob_CG(mesh, ordre)
    return Nglob_CG

def build_masse_frontiere_CG(mesh,ordre:int):
    
    M_ref_1D = build_masse_ref_1D(ordre)
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]    
    loctoglob_CG, glob_to_xy_CG, Nglob_CG=build_loctoglob_CG(mesh, ordre)
    neighbors, neighbor_faces, edges_to_triangles = build_neighborhood_structure(triangles)

    # Calcul des dimensions
    n_dof_face = ordre + 1
    
     # Nombre de faces de bord
    n_faces_bord = np.sum(neighbors < 0)
    nnz = n_faces_bord * (ordre + 1)**2    

    # Création de la matrice globale de saut
    MAT_bord_CG  = COOMatrix(Nglob_CG, Nglob_CG, nnz)
    
    # Boucle sur tous les triangles
    for T in range(len(triangles)):
        # Boucle sur les 3 faces de chaque triangle
        for F in range(3):
            V = neighbors[T, F]
            
            if V == -1: 
                # Récupération des sommets du triangle physique
                pt0, pt1, pt2 = triangles[T]
                A0 = points[pt0]
                A1 = points[pt1]
                A2 = points[pt2]
                
                # Détermination des extrémités de la face physique
                if F == 0:
                    # Face opposée au sommet 0: arête A1→A2
                    P_debut = A1
                    P_fin = A2
                elif F == 1:
                    # Face opposée au sommet 1: arête A2→A0
                    P_debut = A2
                    P_fin = A0
                elif F == 2:
                    # Face opposée au sommet 2: arête A0→A1
                    P_debut = A0
                    P_fin = A1
                else:
                    raise ValueError(f"F doit être 0, 1 ou 2, pas {F}")
                
                # Calcul de la longueur de l'arête physique
                longueur_arete = np.linalg.norm(P_fin - P_debut)               
                # Transformation: ds = longueur * dξ pour ξ ∈ [0,1]
                # La matrice de masse physique est M_face = longueur * M_ref_1D
                M_face_locale = longueur_arete * M_ref_1D                    
                # Assemblage dans la matrice globale
                for i_loc in range(n_dof_face):
                    # Indice global du i-ème DDL sur la face
                    i_glob_CG = iface_iglob_CG(T, F, i_loc, ordre, loctoglob_CG)
                    for j_loc in range(n_dof_face):
                        # Indice global du j-ème DDL sur la face
                        j_glob_CG = iface_iglob_CG(T, F, j_loc, ordre, loctoglob_CG)
                        # Ajout de la contribution à la matrice globale
                        MAT_bord_CG.ajout(i_glob_CG, j_glob_CG, M_face_locale[i_loc, j_loc])
                
    
    return MAT_bord_CG



def termes_source_frontiere_CG(func,mesh, ordre:int):
    """
    Construit le terme **source frontière** en méthode C0 (continue)

    Parameters:
    -----------
    func : callable
        Fonction définissant la condition aux limites sur la frontière: func(x, y)
    ordre : int
        Ordre des éléments finis
    triangles : array (NT, 3)
        Triangles du maillage (indices des sommets)
    points : array (N_points, 2)
        Coordonnées des sommets
    neighbors : array (NT, 3)
        Voisinage des triangles (-1 si face de bord)
    loctoglob_CG : array (NT, ordre+1, ordre+1)
        Table locale -> globale C0

    Returns:
    --------
    F_boundary_CG : array (Nglob_CG,)
        Terme source frontière C0
    """

    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]    
    loctoglob_CG, glob_to_xy_CG, Nglob_CG=build_loctoglob_CG(mesh, ordre)
    neighbors, neighbor_faces, edges_to_triangles = build_neighborhood_structure(triangles)

    # Taille globale
    # **Sécurise si jamais il y a des -1 dans loctoglob_CG**
    valid = loctoglob_CG[loctoglob_CG >= 0]
    Nglob_CG = int(np.max(valid)) + 1
    F_boundary_CG = np.zeros(Nglob_CG,dtype=np.complex128)

    n_dof_face = ordre + 1

    # Construction d'une **formule** de quadrature à 2*ordre + 2 points
    quad_points, quad_weights = leggauss(2 * ordre + 2)
    quad_points = 0.5 * (quad_points + 1)   # [-1,1] -> [0,1]
    quad_weights = 0.5 * quad_weights       # ajuste les poids

    NT = len(triangles)

    for T in range(NT):
        # Sommets du triangle physique
        pt0, pt1, pt2 = triangles[T]
        A0 = points[pt0]
        A1 = points[pt1]
        A2 = points[pt2]

        for F in range(3):
            V = neighbors[T, F]
            if V != -1:
                continue  # pas une face de bord

            # Extrémités de la face physique
            if F == 0:
                P_debut, P_fin = A1, A2
            elif F == 1:
                P_debut, P_fin = A2, A0
            elif F == 2:
                P_debut, P_fin = A0, A1
            else:
                raise ValueError(f"F doit être 0, 1 ou 2, pas {F}")

            longueur_arete = np.linalg.norm(P_fin - P_debut)

            for xi, wq in zip(quad_points, quad_weights):
                P_q = P_debut + xi * (P_fin - P_debut)
                x_q, y_q = P_q
                f_q = np.complex128(func(x_q, y_q))

                for i_loc_face in range(n_dof_face):
                    phi_i = base_1D(xi,i_loc_face, ordre)
                    i_glob_CG = iface_iglob_CG(T, F, i_loc_face, ordre, loctoglob_CG)
                    F_boundary_CG[i_glob_CG] += f_q * phi_i * wq * longueur_arete

    return F_boundary_CG

def plot_nodal_vector_CG(U_CG, mesh, ordre:int, title:str, flag_maillage=True):
    """
    Visualise un vecteur nodal C0 en le convertissant en DG, puis en appelant
    la routine de tracé DG.
    """
    U_CG = U_CG.flatten()
    # DDL locaux et géométrie du maillage
    # Conversion C0 -> DG, puis tracé DG
    U_DG = nodal_CG_to_DG(U_CG, mesh, ordre)
    plot_nodal_vector_DG(U_DG, mesh, ordre, title, flag_maillage)
    
def plot_support_terme_source(F_CG, mesh, ordre:int):
    # DDL locaux et géométrie du maillage
    points = mesh.points[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])
    # Tables de correspondance C0 et DG
    loctoglob_DG, n_glob_DG = build_loctoglob_DG(triangles, ordre)
    loc_to_glob_CG, glob_to_xy, Nglob_CG = build_loctoglob_CG(mesh, ordre)
    # Coordonnées DG des DDL pour l'affichage
    dof_coords = build_dof_coordinates_DG(mesh, ordre)
    # Support du terme source : 1 là où F_CG est non nul, 0 ailleurs    
    F_binaire = (F_CG != 0).astype(int)
    F_DG=nodal_CG_to_DG(F_binaire, mesh, ordre)
    scatter_nodal_vector_DG(F_DG, mesh, ordre, title="", figsize=(10, 8), cmap='viridis', s=20)


def termes_source_frontiere_gradn_CG(fx, fy, mesh,ordre:int):
    """
    Assemble F_i = ∫_{∂Ω} (∇f · n_ext) φ_i ds   en C0
    en utilisant la routine du notebook: calcul_normale(A0,A1,A2,i).

    fx, fy : callables
        fx(x,y)=∂f/∂x, fy(x,y)=∂f/∂y
    """
    points = mesh.points[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])
    # Tables de correspondance C0 et DG
    loctoglob_CG, glob_to_xy, Nglob_CG = build_loctoglob_CG(mesh, ordre)
    neighbors, neighbor_faces, edges_to_triangles = build_neighborhood_structure(triangles)



    # Taille globale
    valid = loctoglob_CG[loctoglob_CG >= 0]
    Nglob_CG = int(np.max(valid)) + 1
    F_boundary_CG = np.zeros(Nglob_CG,dtype=np.complex128)

    n_dof_face = ordre + 1

    # Quadrature 1D sur [0,1]
    quad_points, quad_weights = leggauss(2 * ordre + 2)
    quad_points  = 0.5 * (quad_points + 1.0)   # [-1,1] -> [0,1]
    quad_weights = 0.5 * quad_weights

    NT = len(triangles)

    for T in range(NT):
        pt0, pt1, pt2 = triangles[T]
        A0 = points[pt0]
        A1 = points[pt1]
        A2 = points[pt2]

        for F in range(3):
            if neighbors[T, F] != -1:
                continue  # pas une face de bord

            # Extrémités de la face physique (comme dans ton code)
            if F == 0:
                P_debut, P_fin = A1, A2
            elif F == 1:
                P_debut, P_fin = A2, A0
            elif F == 2:
                P_debut, P_fin = A0, A1
            else:
                raise ValueError(f"F doit être 0, 1 ou 2, pas {F}")

            # **Normale extérieure unitaire via TA routine**
            n = calcul_normale(A0, A1, A2, F)

            # longueur de l'arête
            longueur_arete = np.linalg.norm(P_fin - P_debut)

            for xi, wq in zip(quad_points, quad_weights):
                P_q = P_debut + xi * (P_fin - P_debut)
                x_q, y_q = P_q

                # ∇f · n
                gn_q = np.complex128(fx(x_q, y_q) * n[0] + fy(x_q, y_q) * n[1])

                for i_loc_face in range(n_dof_face):
                    phi_i = base_1D(xi, i_loc_face, ordre)
                    i_glob_CG = iface_iglob_CG(T, F, i_loc_face, ordre, loctoglob_CG)
                    F_boundary_CG[i_glob_CG] += gn_q * phi_i * wq * longueur_arete

    return F_boundary_CG

# Affichage des faces de bord du domaine
def plot_faces_bord_CG(mesh, ordre:int):
    points = mesh.points[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])
    neighbors, neighbor_faces, edges_to_triangles = build_neighborhood_structure(triangles)
    loc_to_glob_CG, glob_to_xy_CG, Nglob_CG = build_loctoglob_CG(mesh, ordre)
    face_coords = []
    face_colors = []
    from matplotlib import cm
    from matplotlib.cm import get_cmap

    tab10 = get_cmap('tab10')

    for T in range(len(triangles)):
        for F in range(3):
            V = neighbors[T, F]
                
            if V == -1:
                color = tab10(F % 10)
                # if neighbors[T, F] < 0:  # Face de bord
                for iloc_face in range(ordre + 1):
                        iglob = iface_iglob_CG(T, F, iloc_face, ordre, loc_to_glob_CG)
                        x, y = glob_to_xy_CG[iglob]
                        face_coords.append([x, y])
                        face_colors.append(color)
    face_coords = np.array(face_coords)
    face_colors = np.array(face_colors)
    plt.triplot(points[:, 0],points[:, 1],triangles,'k-',alpha=0.6,linewidth=2)
    plt.scatter(face_coords[:, 0], face_coords[:, 1], color='r', s=20)
    plt.title("DDL de bord (C0) - couleur par face")
    plt.axis('equal')
    plt.show()




def plot_faces_interieur_CG(mesh, ordre:int):
    from matplotlib import cm
    from matplotlib.cm import get_cmap
    tab10 = get_cmap('tab10')
    points = mesh.points[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])
    neighbors, neighbor_faces, edges_to_triangles = build_neighborhood_structure(triangles)
    loc_to_glob_CG, glob_to_xy_CG, Nglob_CG = build_loctoglob_CG(mesh, ordre)
    face_coords = []
    face_colors = []
    for T in range(len(triangles)):
        for F in range(3):
            V = neighbors[T, F]
                
            if V >=0:
                color = tab10(F % 10)
                # if neighbors[T, F] < 0:  # Face de bord
                for iloc_face in range(ordre + 1):
                        iglob = iface_iglob_CG(T, F, iloc_face, ordre, loc_to_glob_CG)
                        x, y = glob_to_xy_CG[iglob]
                        face_coords.append([x, y])
                        face_colors.append(color)
    face_coords = np.array(face_coords)
    face_colors = np.array(face_colors)
    plt.triplot(points[:, 0],points[:, 1],triangles,'k-',alpha=0.6,linewidth=2)
    plt.scatter(face_coords[:, 0], face_coords[:, 1], color='g', s=20)
    plt.title("DDL de bord (C0) - couleur par face")
    plt.axis('equal')
    plt.show()


def plot_nodal_vector_moins_fonction_CG(U_CG,func, mesh, ordre, title):
    Nloc = (ordre + 1) * (ordre + 2) // 2

    # Recuperation de la géométrie du maillage
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]    
    loctoglob_DG, n_glob_DG = build_loctoglob_DG(triangles, ordre)
    dof_coords = build_dof_coordinates_DG(mesh, ordre)
    U_func_CG = build_nodal_vector_CG(func, mesh,ordre)
    U_diff_CG = U_CG - U_func_CG
    plot_nodal_vector_CG(U_diff_CG, mesh, ordre,  title)


def nombre_DDL_CG_par_DDL_DG(mesh,  ordre):
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]    
    loc_to_glob_CG, _, Nglob_CG = build_loctoglob_CG(mesh, ordre)  

    n_triangles = len(triangles)
    Nglob_CG = np.max(loc_to_glob_CG) + 1

    compteur =np.zeros(Nglob_CG, dtype=int)

    for ielt in range(n_triangles):    
        for iloc1 in range(ordre + 1):
            for jloc1 in range(ordre + 1 - iloc1):
                iglob = loc_to_glob_CG[ielt, iloc1, jloc1]
                compteur[iglob] += 1

    compteur_DG = nodal_CG_to_DG(compteur, mesh,ordre)
    scatter_nodal_vector_DG(compteur_DG, mesh, ordre, title="", figsize=(10, 8), cmap='Pastel1', s=20)
    return compteur

