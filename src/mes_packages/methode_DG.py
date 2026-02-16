import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mes_packages import (
    loc2D_to_loc1D, 
    build_masse_locale, 
    build_mixte_locale,
    build_rigidite_locale,
    build_masse_ref_1D,
    build_neighborhood_structure,
    base,derivative_base,
    build_neighborhood_structure_with_bc,
    base_1D,
    COOMatrix)
from mes_packages.matrice_reference import Mref
from mes_packages.quadrature import quadrature_triangle_ref_2D

def build_loctoglob_DG(triangles, ordre):
    """
    Construit la table de correspondance locale -> globale
    pour la méthode Galerkin Discontinue (DG)
    
    En DG, chaque élément a ses propres DDL sans partage entre éléments
    
    Parameters:
    -----------
    triangles : array (n_triangles, 3)
        Indices des sommets de chaque triangle
    ordre : int
        Ordre des éléments finis (1 pour P1, 2 pour P2, etc.)
        
    Returns:
    --------
    loctoglob : array (n_triangles, Nloc)
        loctoglob[ielt, iloc] = iglob avec iglob = Nloc * ielt + iloc
    n_dof : int
        Nombre total de degrés de liberté = Nloc * n_triangles
    """
    n_triangles = len(triangles)
    Nloc = (ordre + 1) * (ordre + 2) // 2
    n_dof = Nloc * n_triangles
    
    # Construction de la table loctoglob
    loctoglob_DG = np.zeros((n_triangles, Nloc), dtype=int)
    
    for ielt in range(n_triangles):
        for iloc in range(Nloc):
            iglob_DG = Nloc * ielt + iloc
            loctoglob_DG[ielt, iloc] = iglob_DG
    
    return loctoglob_DG, n_dof

def print_loctoglob_DG(loctoglob_DG,triangles, ordre,n_glob_DG):
    print(f"=== Méthode Galerkin Discontinue (DG) ===")
    print(f"Ordre : P{ordre}")
    print(f"Nloc (DDL par élément) : {(ordre+1)*(ordre+2)//2}")
    print(f"Nombre d'éléments : {len(triangles)}")
    print(f"Nombre total de DDL : {n_glob_DG}")
    print(f"\nTable loctoglob (premiers triangles) :")
    for i in range(min(5, len(triangles))):
        print(f"Triangle {i} : {loctoglob_DG[i]}")
    print(75*":")
    print(75*":")
    for i in range(len(triangles)-5,len(triangles)):
        print(f"Triangle {i} : {loctoglob_DG[i]}")


def build_dof_coordinates_DG(mesh, ordre):
    """
    Construit les coordonnées (x, y) de chaque degré de liberté en DG
    
    Pour chaque triangle, calcule les coordonnées des points d'interpolation
    sur le triangle de référence puis transforme vers le triangle réel
    
    Parameters:
    -----------
    mesh : Mesh object
        Maillage contenant les triangles et les points
    ordre : int
        Ordre des éléments finis
        
    Returns:
    --------
    dof_coords : array (n_dof, 2)
        Coordonnées (x, y) de chaque DDL
        dof_coords[iglob] = [x, y]
    """
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    n_triangles = len(triangles)
    Nloc = (ordre + 1) * (ordre + 2) // 2
    n_dof = Nloc * n_triangles
    
    dof_coords = np.zeros((n_dof, 2))
    
    # Points d'interpolation sur le triangle de référence
    z = np.linspace(0, 1, ordre + 1)
    
    for ielt in range(n_triangles):
        # Sommets du triangle réel
        A1 = points[triangles[ielt, 0]]
        A2 = points[triangles[ielt, 1]]
        A3 = points[triangles[ielt, 2]]
        
        # Pour chaque fonction de base (m, n)
        for m in range(ordre + 1):
            for n in range(ordre + 1 - m):
                # Indice local
                iloc = loc2D_to_loc1D(m, n)
                
                # Coordonnées sur le triangle de référence
                # Point d'interpolation associé à (m, n)
                x_ref = z[m]
                y_ref = z[n]
                
                # Transformation vers le triangle réel
                # P = A1 + x_ref * (A2 - A1) + y_ref * (A3 - A1)
                x_real = A1[0] + x_ref * (A2[0] - A1[0]) + y_ref * (A3[0] - A1[0])
                y_real = A1[1] + x_ref * (A2[1] - A1[1]) + y_ref * (A3[1] - A1[1])
                
                # Indice global
                iglob = Nloc * ielt + iloc
                
                dof_coords[iglob, 0] = x_real
                dof_coords[iglob, 1] = y_real
    
    return dof_coords

def print_premier_DDL_DG(dof_coords):
    print(f"Nombre de DDL : {len(dof_coords)}")
    print(f"\nCoordonnées des premiers DDL :")
    for i in range(min(10, len(dof_coords))):
        print(f"DDL {i} : x = {dof_coords[i, 0]:.6f}, y = {dof_coords[i, 1]:.6f}")

def scatter_dof_coords_DG(mesh, ordre):
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    dof_coords = build_dof_coordinates_DG(mesh, ordre)
        # Visualisation des DDL
    plt.figure(figsize=(10, 10))
    plt.triplot(points[:, 0], points[:, 1], triangles, 'k-', alpha=0.3, linewidth=2)
    plt.scatter(dof_coords[:, 0], dof_coords[:, 1], c='red', s=10, alpha=0.6)
    plt.gca().set_aspect('equal')
    plt.title(f"Position des DDL - Méthode DG P{ordre}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_un_triangle(ielt,mesh,ordre):
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    dof_coords = build_dof_coordinates_DG(mesh, ordre)
    # Zoom sur un triangle pour mieux voir
    
    A1 = points[triangles[ielt, 0]]
    A2 = points[triangles[ielt, 1]]
    A3 = points[triangles[ielt, 2]]

    # DDL du triangle
    Nloc = (ordre + 1) * (ordre + 2) // 2
    dof_indices = [Nloc * ielt + iloc for iloc in range(Nloc)]
    dof_tri = dof_coords[dof_indices]

    plt.figure(figsize=(8, 8))
    plt.plot([A1[0], A2[0]], [A1[1], A2[1]], 'k-', linewidth=2)
    plt.plot([A2[0], A3[0]], [A2[1], A3[1]], 'k-', linewidth=2)
    plt.plot([A3[0], A1[0]], [A3[1], A1[1]], 'k-', linewidth=2)
    plt.scatter(dof_tri[:, 0], dof_tri[:, 1], c='red', s=100, zorder=5)

    # Numérotation locale
    for iloc in range(Nloc):
        plt.text(dof_tri[iloc, 0], dof_tri[iloc, 1], str(iloc), 
                fontsize=10, ha='center', va='bottom')

    plt.gca().set_aspect('equal')
    plt.title(f"DDL du triangle {ielt} (P{ordre})")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.show()


def build_nodal_vector_DG(f, mesh, ordre):
    """
    Construit le vecteur nodal associé à une fonction f(x,y)
    en évaluant f aux points d'interpolation (DDL)
    
    Parameters:
    -----------
    f : function
        Fonction f(x, y) à évaluer
    mesh : meshio.Mesh
        Maillage contenant les points et les triangles
    ordre : int
        Ordre des éléments finis
        
    Returns:
    --------
    U : array (n_dof,)
        Vecteur nodal : U[iglob] = f(x_iglob, y_iglob)
    """
    points = mesh.points[:, :2]
    triangles = mesh.cells_dict["triangle"]
    dof_coords = build_dof_coordinates_DG(mesh, ordre)
    n_dof = len(dof_coords)
    U = np.zeros(n_dof,dtype=np.complex128)
    
    for iglob in range(n_dof):
        x = dof_coords[iglob, 0]
        y = dof_coords[iglob, 1]
        U[iglob] = f(x, y)
    
    return U


def scatter_nodal_vector_DG(U, mesh,ordre,
                            title="", figsize=(10, 8),
                            cmap='viridis', s=20):
    """
    Visualise un vecteur nodal DG sur le maillage
    """
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]   
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    dof_coords=build_dof_coordinates_DG(mesh, ordre)
    U = np.asarray(U)
    imag_max = np.max(np.abs(np.imag(U)))

    def _scatter(fig, ax, values, subtitle):
        scatter = ax.scatter(
            dof_coords[:, 0],
            dof_coords[:, 1],
            c=values,
            cmap=cmap,
            s=s
        )
        ax.triplot(
            points[:, 0],
            points[:, 1],
            triangles,
            'k-',
            alpha=0.6,
            linewidth=2
        )
        ax.set_aspect('equal')
        ax.set_title(subtitle)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(scatter, ax=ax)
        ax.grid(True, alpha=0.3)

    # --- Cas réel -----------------------------------------------------------
    if imag_max < 1e-10:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        _scatter(fig, ax, np.real(U), title or "Re(U)")
        plt.show()
        return fig, ax

    # --- Cas complexe -------------------------------------------------------
    fig, axes = plt.subplots(
        3, 1,
        figsize=(figsize[0], 3 * figsize[1]),
        constrained_layout=True
    )

    _scatter(fig, axes[0], np.real(U),
             (title + " - Re(U)") if title else "Re(U)")
    _scatter(fig, axes[1], np.imag(U),
             (title + " - Im(U)") if title else "Im(U)")
    _scatter(fig, axes[2], np.abs(U),
             (title + " - |U|") if title else "|U|")

    plt.show()
    return fig, axes

# Assemblage de la matrice de masse globale
def build_masse_DG(mesh, ordre):
    """
    Construit la matrice de masse globale en méthode DG
    
    Parameters:
    -----------
    mesh : meshio.Mesh
        Maillage contenant les points et les triangles
    ordre : int
        Ordre des éléments finis
        
    Returns:
    --------
    Mglob : array (n_dof, n_dof)
        Matrice de masse globale
    """
    points = mesh.points[:, :2]
    triangles = mesh.cells_dict["triangle"]
    
    n_triangles = len(triangles)
    Nloc = (ordre + 1) * (ordre + 2) // 2
    n_dof = Nloc * n_triangles
    
    Mat = COOMatrix(n_dof, n_dof,n_dof*Nloc)  # Estimation du nombre de non-zéros (peut être ajustée)
    
    Mhat = Mref(ordre)
    for ielt in range(n_triangles):
        A1 = points[triangles[ielt, 0]]
        A2 = points[triangles[ielt, 1]]
        A3 = points[triangles[ielt, 2]]
        Jac = abs((A2[0] - A1[0]) * (A3[1] - A1[1]) - (A3[0] - A1[0]) * (A2[1] - A1[1]))        
        Mloc = Jac * Mhat
        for iloc1 in range(Nloc):
            iglob1 = Nloc * ielt + iloc1
            for iloc2 in range(Nloc):
                iglob2 = Nloc * ielt + iloc2
                Mat.ajout(iglob1, iglob2, Mloc[iloc1, iloc2])

    return Mat


def build_mixte_DG(mesh, ordre):
    """
    Construit les matrices mixtes globales Kx et Ky en méthode DG
    
    Parameters:
    -----------
    mesh : meshio.Mesh
        Maillage contenant les points et les triangles
    ordre : int
        Ordre des éléments finis
    Matx : COOMatrix
        Matrice COO pour stocker Kx
    Maty : COOMatrix
        Matrice COO pour stocker Ky
    """
    points = mesh.points[:, :2]
    triangles = mesh.cells_dict["triangle"]
    
    
    n_triangles = len(triangles)
    Nloc = (ordre + 1) * (ordre + 2) // 2
    Nglob = Nloc * n_triangles
    
    Matx = COOMatrix(Nglob, Nglob, Nloc*Nglob)
    Maty = COOMatrix(Nglob, Nglob, Nloc*Nglob)
    
    for ielt in range(n_triangles):
        A1 = points[triangles[ielt, 0]]
        A2 = points[triangles[ielt, 1]]
        A3 = points[triangles[ielt, 2]]
        
        # Calcul des matrices mixtes locales
        Klocx, Klocy = build_mixte_locale(ordre, A1, A2, A3)
        
        # Assemblage dans les matrices globales
        for iloc1 in range(Nloc):
            iglob1 = Nloc * ielt + iloc1
            for iloc2 in range(Nloc):
                iglob2 = Nloc * ielt + iloc2
                Matx.ajout(iglob1, iglob2, Klocx[iloc1, iloc2])
                Maty.ajout(iglob1, iglob2, Klocy[iloc1, iloc2])
    return Matx, Maty

def plot_on_mesh_function(func, mesh, ordre, flag_maillage=True):
# def plot_on_mesh_DG_cellwise(func, triangles, points, ordre, loctoglob_DG, dof_coords):
    """
    Affiche une fonction DG en créant des sous-triangles avec les DDL.
    Représentation P1 continue sur chaque sous-triangle.
    Génère deux figures : une avec maillage, une sans.
    """
    from matplotlib.tri import Triangulation
    
    Nloc = (ordre + 1) * (ordre + 2) // 2

    # Recuperation de la géométrie du maillage
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]    
    
    loctoglob_DG, n_glob_DG = build_loctoglob_DG(triangles, ordre)
    dof_coords = build_dof_coordinates_DG(mesh, ordre)

    # Listes pour stocker tous les points et sous-triangles
    all_X = []
    all_Y = []
    all_Z = []
    all_triangles = []
    offset = 0
    
    for iT in range(len(triangles)):
        # Récupérer les coordonnées et valeurs aux DDL du triangle
        X_dof = np.zeros(Nloc)
        Y_dof = np.zeros(Nloc)
        Z_dof = np.zeros(Nloc)
        
        for iloc in range(Nloc):
            iglob = loctoglob_DG[iT, iloc]
            X_dof[iloc], Y_dof[iloc] = dof_coords[iglob]
            Z_dof[iloc] = func(X_dof[iloc], Y_dof[iloc])
        
        # Ajouter les points DDL
        all_X.extend(X_dof)
        all_Y.extend(Y_dof)
        all_Z.extend(Z_dof)
        
        # Créer des sous-triangles reliant les DDL
        for m in range(ordre):
            for n in range(ordre - m):
                i0 = loc2D_to_loc1D(m, n)
                i1 = loc2D_to_loc1D(m+1, n)
                i2 = loc2D_to_loc1D(m, n+1)
                
                all_triangles.append([offset + i0, offset + i1, offset + i2])
                
                # Triangle "inversé" si nécessaire
                if m + n + 1 < ordre:
                    i3 = loc2D_to_loc1D(m+1, n+1)
                    all_triangles.append([offset + i1, offset + i3, offset + i2])
        
        offset += Nloc
    
    # Créer la triangulation
    X_array = np.array(all_X)
    Y_array = np.array(all_Y)
    Z_array = np.array(all_Z)
    tri = Triangulation(X_array, Y_array, all_triangles)
    
    # Figure 1 : Sans maillage
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    tcf1 = ax1.tripcolor(tri, Z_array, cmap='viridis', shading='gouraud', edgecolors='k', linewidths=0.5)
    if flag_maillage:
        ax1.triplot(points[:, 0], points[:, 1], triangles, 'k-', linewidth=1.5, alpha=0.7)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("DG : représentation P1 (sans maillage)")
    plt.colorbar(tcf1, ax=ax1, label="u")
    plt.show()
    
def plot_nodal_vector_DG(U, mesh, ordre, title,flag_maillage=True):
    """
    Affiche un vecteur nodal DG complexe en créant des sous-triangles avec les DDL.
    Représentation P1 continue sur chaque sous-triangle (tripcolor gouraud).
    Génère deux figures : une avec maillage, une sans, chacune en 3 frames:
    - partie réelle
    - partie imaginaire
    - module
    """
    from matplotlib.tri import Triangulation
    import numpy as np
    import matplotlib.pyplot as plt

    Nloc = (ordre + 1) * (ordre + 2) // 2
    # Recuperation de la géométrie du maillage
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]        
    loctoglob_DG, n_glob_DG = build_loctoglob_DG(triangles, ordre)
    dof_coords = build_dof_coordinates_DG(mesh, ordre)


    U = np.asarray(U)

    Nloc = (ordre + 1) * (ordre + 2) // 2

    # Listes pour stocker tous les points et sous-triangles
    all_X = []
    all_Y = []
    all_Z = []
    all_triangles = []
    offset = 0

    for iT in range(len(triangles)):
        # Récupérer les coordonnées et valeurs aux DDL du triangle
        X_dof = np.zeros(Nloc)
        Y_dof = np.zeros(Nloc)
        Z_dof = np.zeros(Nloc, dtype=np.complex128)  # **CHANGÉ**

        for iloc in range(Nloc):
            iglob = loctoglob_DG[iT, iloc]
            X_dof[iloc], Y_dof[iloc] = dof_coords[iglob]
            Z_dof[iloc] = U[iglob]

        all_X.extend(X_dof)
        all_Y.extend(Y_dof)
        all_Z.extend(Z_dof)

        # Créer des sous-triangles reliant les DDL
        for m in range(ordre):
            for n in range(ordre - m):
                i0 = loc2D_to_loc1D(m, n)
                i1 = loc2D_to_loc1D(m + 1, n)
                i2 = loc2D_to_loc1D(m, n + 1)

                all_triangles.append([offset + i0, offset + i1, offset + i2])

                # Triangle "inversé" si nécessaire
                if m + n + 1 < ordre:
                    i3 = loc2D_to_loc1D(m + 1, n + 1)
                    all_triangles.append([offset + i1, offset + i3, offset + i2])

        offset += Nloc

    # Créer la triangulation
    X_array = np.array(all_X)
    Y_array = np.array(all_Y)
    Z_array = np.array(all_Z, dtype=np.complex128)  # **CHANGÉ**
    tri = Triangulation(X_array, Y_array, all_triangles)

    # Préparer les 3 champs réels à afficher
    champs = [
        (np.real(Z_array), "partie réelle", "Re(u)"),
        (np.imag(Z_array), "partie imaginaire", "Im(u)"),
        (np.abs(Z_array),  "module", "|u|")
    ]


    # ---------- Figure 2 : Avec maillage (3 frames) ----------
    fig2, ax2 = plt.subplots(1, 3, figsize=(18, 6))

    for k, (Zk, label_long, cblabel) in enumerate(champs):
        tcf2 = ax2[k].tripcolor(
            tri, Zk,
            cmap='viridis',
            shading='gouraud',
            edgecolors='k',
            linewidths=0.5
        )
        if flag_maillage:
            ax2[k].triplot(points[:, 0], points[:, 1], triangles, 'k-',
                       linewidth=1, alpha=0.4)
        ax2[k].set_aspect("equal")
        ax2[k].set_xlabel("x")
        ax2[k].set_ylabel("y")
        ax2[k].set_title(f"DG : {title} ({label_long}) — avec maillage" if title
                         else f"DG : représentation P1 ({label_long}) — avec maillage")
        plt.colorbar(tcf2, ax=ax2[k], label=cblabel)

    plt.tight_layout()
    plt.show()

    return fig2, ax2

def iface_iglob(ielt, iface, iloc_face, ordre, loctoglob_DG):
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
    
    # Conversion (m,n) -> indice local
    iloc = loc2D_to_loc1D(m, n)
    
    # Conversion indice local -> indice global
    iglob = loctoglob_DG[ ielt,iloc]
    
    return iglob

def plot_face_to_glob(ielt_test,iface_test,mesh,ordre):

    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]        
    loctoglob_DG, n_glob_DG = build_loctoglob_DG(triangles, ordre)
    dof_coords = build_dof_coordinates_DG(mesh, ordre)

    # Récupération des sommets du triangle physique
    pt1, pt2, pt3 = triangles[ielt_test]
    A0_test = points[pt1]  # Sommet 0
    A1_test = points[pt2]  # Sommet 1
    A2_test = points[pt3]  # Sommet 2

    fig, ax = plt.subplots(figsize=(10, 8))

    # Tracé du triangle
    triangle = Polygon([A0_test, A1_test, A2_test], fill=False, edgecolor='black', linewidth=2) # type: ignore
    ax.add_patch(triangle)
    # Annotation des sommets
    ax.plot(A0_test[0], A0_test[1], 'ro', markersize=10, label='Sommet 0')
    ax.plot(A1_test[0], A1_test[1], 'go', markersize=10, label='Sommet 1')
    ax.plot(A2_test[0], A2_test[1], 'bo', markersize=10, label='Sommet 2')
    ax.text(A0_test[0], A0_test[1], ' A0', fontsize=12, ha='left')
    ax.text(A1_test[0], A1_test[1], ' A1', fontsize=12, ha='left')
    ax.text(A2_test[0], A2_test[1], ' A2', fontsize=12, ha='left')
    # Mise en évidence de la face testée selon iface_test
    if iface_test == 0:
        # Face 0: opposée au sommet 0, donc arête entre sommets 1 et 2 (A1→A2)
        debut_face, fin_face = A1_test, A2_test
    elif iface_test == 1:
        # Face 1: opposée au sommet 1, donc arête entre sommets 2 et 0 (A2→A0)
        debut_face, fin_face = A2_test, A0_test
    elif iface_test == 2:
        # Face 2: opposée au sommet 2, donc arête entre sommets 0 et 1 (A0→A1)
        debut_face, fin_face = A0_test, A1_test
    else:
        raise ValueError(f"iface_test doit être 0, 1 ou 2, pas {iface_test}")

    ax.plot([debut_face[0], fin_face[0]], [debut_face[1], fin_face[1]], 'r-', linewidth=3, label=f'Face {iface_test} (A0→A1)')

    # Boucle sur tous les points de la face (ordre+1 points)
    for iloc_face in range(ordre + 1):
        # Récupération de l'indice global
        iglob = iface_iglob(ielt_test, iface_test, iloc_face, ordre, loctoglob_DG)
        # Récupération des coordonnées du DDL
        coord_dof = dof_coords[iglob]    
        # Tracé du point sur la figure
        ax.plot(coord_dof[0], coord_dof[1], 'mo', markersize=8)
        ax.text(coord_dof[0], coord_dof[1], f'  iloc_face={iloc_face}, iglob={iglob}', fontsize=10, color='magenta')

    # Configuration de l'affichage
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Élément {ielt_test}: Triangle et DDL de la face {iface_test} (ordre={ordre})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    plt.show()



def get_face_dof_pair(T, F, iloc_face, ordre, neighbors, neighbor_faces):
    """
    Retourne les deux numéros globaux du DDL iloc_face de la face F:
    - iglob_T : numéro global pour le triangle T
    - iglob_V : numéro global pour le triangle voisin V (-1 si pas de voisin)
    
    Paramètres:
    -----------
    T : int
        Numéro du triangle
    F : int
        Numéro de la face (0, 1, ou 2)
        - face 0 : arête opposée au sommet 0 (entre sommets 1 et 2)
        - face 1 : arête opposée au sommet 1 (entre sommets 2 et 0)
        - face 2 : arête opposée au sommet 2 (entre sommets 0 et 1)
    iloc_face : int
        Position du DDL sur la face (de 0 à ordre)
    ordre : int
        Ordre polynomial
    neighbors : ndarray
        Tableau des voisins (n_triangles x 3)
    neighbor_faces : ndarray
        Tableau des faces voisines (n_triangles x 3)
        
    Retourne:
    ---------
    iglob_T : int
        Numéro global du DDL pour le triangle T
    iglob_V : int
        Numéro global du DDL pour le triangle voisin V (ou -1 si pas de voisin)
    """
    # Nombre de DDL locaux par triangle
    Nloc = (ordre + 1) * (ordre + 2) // 2
    
    # Déterminer les coordonnées (m, n) dans le triangle de référence pour T
    if F == 0:
        # Face 0 opposée au sommet 0: arête entre sommets 1=(ordre,0) et 2=(0,ordre)
        m_T = ordre - iloc_face
        n_T = iloc_face
    elif F == 1:
        # Face 1 opposée au sommet 1: arête entre sommets 2=(0,ordre) et 0=(0,0)
        m_T = 0
        n_T = ordre - iloc_face
    elif F == 2:
        # Face 2 opposée au sommet 2: arête entre sommets 0=(0,0) et 1=(ordre,0)
        m_T = iloc_face
        n_T = 0
    else:
        raise ValueError(f"F doit être 0, 1 ou 2, pas {F}")
    
    # Conversion (m,n) -> indice local pour T
    iloc_T = loc2D_to_loc1D(m_T, n_T)
    
    # Conversion indice local -> indice global pour T (en DG)
    iglob_T = Nloc * T + iloc_T
    
    # Vérifier s'il y a un voisin
    V = neighbors[T, F]
    
    if V < 0:
        # Pas de voisin (bord)
        iglob_V = -1
    else:
        # Il y a un voisin, récupérer la face de V commune avec T
        F_V = neighbor_faces[T, F]
        
        # Attention: les faces sont orientées en sens opposé!
        # Si iloc_face va de 0 à ordre sur la face de T,
        # il correspond à ordre - iloc_face sur la face de V
        iloc_face_V = ordre - iloc_face
        
        # Déterminer les coordonnées (m, n) dans le triangle de référence pour V
        if F_V == 0:
            m_V = ordre - iloc_face_V
            n_V = iloc_face_V
        elif F_V == 1:
            m_V = 0
            n_V = ordre - iloc_face_V
        elif F_V == 2:
            m_V = iloc_face_V
            n_V = 0
        else:
            raise ValueError(f"F_V doit être 0, 1 ou 2, pas {F_V}")
        
        # Conversion (m,n) -> indice local pour V
        iloc_V = loc2D_to_loc1D(m_V, n_V)
        
        # Conversion indice local -> indice global pour V
        iglob_V = Nloc * V + iloc_V
    
    return iglob_T, iglob_V


def build_masse_frontiere_elt_DG(ielt, iface, ordre, loctoglob,points,triangles,M_ref_1D):
    """
    Construit la matrice de masse de frontière pour la face iface de l'élément ielt.
    
    La matrice de masse de frontière sur une arête est définie par:
    M^F_{ij} = ∫_F φ_i(s) φ_j(s) ds
    
    où F est l'arête (face) du triangle.
    
    Paramètres:
    -----------
    ielt : int
        Numéro du triangle
    iface : int
        Numéro de la face (0, 1, ou 2)
    ordre : int
        Ordre polynomial
    mesh : meshio.Mesh
        Maillage contenant les points et triangles
    loctoglob : ndarray
        Table de correspondance locale vers globale
        
    Retourne:
    ---------
    M_face : COOMatrix
        Matrice de masse de frontière (Nglob x Nglob) avec (ordre+1)^2 éléments non nuls
    """
    
    # Nombre de DDL globaux
    Nglob = loctoglob.shape[0]*loctoglob.shape[1]
    
    # Nombre de DDL sur la face
    n_dof_face = ordre + 1
    
 
    
    # Récupération des sommets du triangle physique
    pt1, pt2, pt3 = triangles[ielt]
    A0 = points[pt1]
    A1 = points[pt2]
    A2 = points[pt3]
    
    # Détermination des extrémités de la face physique
    if iface == 0:
        # Face opposée au sommet 0: arête A1→A2
        P_debut = A1
        P_fin = A2
    elif iface == 1:
        # Face opposée au sommet 1: arête A2→A0
        P_debut = A2
        P_fin = A0
    elif iface == 2:
        # Face opposée au sommet 2: arête A0→A1
        P_debut = A0
        P_fin = A1
    else:
        raise ValueError(f"iface doit être 0, 1 ou 2, pas {iface}")
    
    # Calcul de la longueur de l'arête physique
    longueur_arete = np.linalg.norm(P_fin - P_debut)
    
 
    
    # Transformation: ds = longueur * dξ pour ξ ∈ [0,1]
    # La matrice de masse physique est M_face = longueur * M_ref_1D
    M_face_locale = longueur_arete * M_ref_1D
    
    # Création de la matrice COO globale
    # Nombre d'éléments non nuls: (ordre+1)^2
    nnz = n_dof_face * n_dof_face
    M_face_globale = COOMatrix(Nglob, Nglob, nnz)
    
    # Assemblage dans la matrice globale
    for i_loc in range(n_dof_face):
        # Indice global du i-ème DDL sur la face
        i_glob = iface_iglob(ielt, iface, i_loc, ordre, loctoglob)
        
        for j_loc in range(n_dof_face):
            # Indice global du j-ème DDL sur la face
            j_glob = iface_iglob(ielt, iface, j_loc, ordre, loctoglob)
            
            # Ajout de la contribution à la matrice globale
            M_face_globale.ajout(i_glob, j_glob, M_face_locale[i_loc, j_loc])
    
    return M_face_globale





def plot_dof_neighbors_DG(
    ielt_test,
    triangles,
    points,
    ordre,
    neighbors,
    neighbor_faces,
    dof_coords,
    tolerance=1e-10,
    figsize=(14, 12),
    show=True,
    ax=None,
    verbose=True,
):
    """
    Visualize DOF coordinates for a triangle and its neighbors, face by face.

    Raises ValueError if matching face DOFs between neighboring triangles do not
    share the same coordinates (within tolerance).
    """
    x_V=0
    y_V=0
    if verbose:
        print("\n=== DOF coordinate visualization ===\n")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Central triangle
    pt0, pt1, pt2 = triangles[ielt_test]
    A0 = points[pt0]
    A1 = points[pt1]
    A2 = points[pt2]

    triangle_central = Polygon(
        [A0, A1, A2],
        fill=False,
        edgecolor="black",
        linewidth=3,
    )
    ax.add_patch(triangle_central)

    colors = ["red", "green", "blue"]
    face_names = ["Face 0", "Face 1", "Face 2"]

    for F in range(3):
        color = colors[F]
        V = neighbors[ielt_test, F]

        if V >= 0:
            pt0_v, pt1_v, pt2_v = triangles[V]
            A0_v = points[pt0_v]
            A1_v = points[pt1_v]
            A2_v = points[pt2_v]
            triangle_voisin = Polygon(
                [A0_v, A1_v, A2_v],
                fill=False,
                edgecolor=color,
                linewidth=2,
                linestyle="--",
                alpha=0.5,
            )
            ax.add_patch(triangle_voisin)

        for iloc_face in range(ordre + 1):
            iglob_T, iglob_V = get_face_dof_pair(
                ielt_test,
                F,
                iloc_face,
                ordre,
                neighbors,
                neighbor_faces,
            )

            x_T = dof_coords[iglob_T, 0]
            y_T = dof_coords[iglob_T, 1]

            if iglob_V >= 0:
                x_V = dof_coords[iglob_V, 0]
                y_V = dof_coords[iglob_V, 1]
                distance = np.sqrt((x_T - x_V) ** 2 + (y_T - y_V) ** 2)
                if distance > tolerance:
                    raise ValueError(
                        "DOF coordinates do not match: "
                        f"T={ielt_test}, F={F}, iloc_face={iloc_face}, "
                        f"iglob_T={iglob_T}, iglob_V={iglob_V}, "
                        f"distance={distance:.2e}"
                    )

            ax.plot(
                x_T,
                y_T,
                "o",
                color="white",
                markersize=14,
                markeredgecolor=color,
                markeredgewidth=2.5,
                label=f"{face_names[F]} (T)" if iloc_face == 0 else "",
                zorder=3,
            )
            ax.plot(
                x_T,
                y_T,
                "x",
                color=color,
                markersize=18,
                markeredgewidth=3.5,
                zorder=5,
            )
            ax.text(
                x_T - 0.002,
                y_T - 0.002,
                f"{iglob_T}",
                fontsize=10,
                fontweight="bold",
                ha="left",
                va="center",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.8,
                ),
            )

            if iglob_V >= 0:
                ax.plot(
                    x_V,
                    y_V,
                    "o",
                    color=color,
                    markersize=14,
                    markeredgecolor=color,
                    markeredgewidth=2.5,
                    fillstyle="left",
                    label=f"{face_names[F]} (V)" if iloc_face == 0 else "",
                    zorder=4,
                )
                ax.text(
                    x_V + 0.002,
                    y_V + 0.002,
                    f"{iglob_V}",
                    fontsize=9,
                    style="italic",
                    ha="right",
                    va="center",
                    color=color,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.7,
                    ),
                )
                ax.plot(
                    [x_T, x_V],
                    [y_T, y_V],
                    color=color,
                    linewidth=1.5,
                    linestyle=":",
                    alpha=0.6,
                )

    cx = (A0[0] + A1[0] + A2[0]) / 3
    cy = (A0[1] + A1[1] + A2[1]) / 3
    ax.text(
        cx,
        cy,
        f"T={ielt_test}",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        "Global DOF visualization - triangle "
        f"{ielt_test}\n"
        "Filled markers: DOF of T, empty markers: DOF of neighbors\n"
        "Dotted lines link matching face DOFs",
        fontsize=12,
    )
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)

    if show:
        plt.tight_layout()
        plt.show()

    if verbose:
        print("\nLegend:")
        print(f"  - Filled markers: DOF of triangle T = {ielt_test}")
        print("  - Empty markers: DOF of neighbor triangles")
        print("  - Dotted lines: matching DOFs across faces")
        print("  - Colors: red (face 0), green (face 1), blue (face 2)")

    return fig, ax



def build_boundary_mass_TV(T, F, ordre, triangles, points, neighbors, neighbor_faces, M_ref_1D):
    """
    Construit la matrice de masse de frontière entre l'élément T et son voisin V
    à travers la face F.
    
    La matrice représente: M_TV[i,j] = ∫_F φ_V^j * φ_T^i dl
    où φ_V^j est la j-ème fonction de base de V sur la face et φ_T^i la i-ème de T
    
    ATTENTION: Les numérotations des faces sont en sens opposé entre T et V
    Si iloc_T varie de 0 à ordre sur T, il correspond à iloc_V = ordre - iloc_T sur V
    
    Parameters:
    -----------
    T : int
        Numéro de l'élément
    F : int
        Numéro de la face (0, 1, ou 2)
    ordre : int
        Ordre polynomial
    triangles : ndarray
        Tableau des triangles (n_triangles x 3)
    points : ndarray
        Coordonnées des sommets (n_points x 2)
    neighbors : ndarray
        Tableau des voisins (n_triangles x 3)
    neighbor_faces : ndarray
        Tableau des faces voisines (n_triangles x 3)
    M_ref_1D : ndarray
        Matrice de masse de référence 1D sur [0,1] (ordre+1 x ordre+1)
        
    Returns:
    --------
    M_TV : COOMatrix
        Matrice de masse de frontière (Nglob x Nglob)
        ou None si pas de voisin (face de bord)
    """

    # Nombre de DDL globaux
    Nloc = (ordre + 1) * (ordre + 2) // 2
    n_triangles = len(triangles)
    Nglob = Nloc * n_triangles
    
    # Nombre de DDL sur une face
    n_dof_face = ordre + 1

    # Création de la matrice COO globale
    # Nombre d'éléments non nuls: (ordre+1)^2
    nnz = n_dof_face * n_dof_face
    M_TV = COOMatrix(Nglob, Nglob, nnz)

    # Vérifier s'il y a un voisin
    V = neighbors[T, F]
    if V < 0:
        return M_TV  # Face de bord, pas de voisin

    # Récupérer les sommets du triangle T
    pt0_T, pt1_T, pt2_T = triangles[T]
    A0_T = points[pt0_T]
    A1_T = points[pt1_T]
    A2_T = points[pt2_T]
    
    # Déterminer les extrémités de la face F de T
    if F == 0:  # Face opposée au sommet 0: entre sommets 1 et 2
        P1_T = A1_T
        P2_T = A2_T
    elif F == 1:  # Face opposée au sommet 1: entre sommets 2 et 0
        P1_T = A2_T
        P2_T = A0_T
    else:  # F == 2: Face opposée au sommet 2: entre sommets 0 et 1
        P1_T = A0_T
        P2_T = A1_T
    
    # Longueur de la face
    longueur_face = np.linalg.norm(P2_T - P1_T)
    
    # Matrice de masse physique sur la face: M_phys = longueur * M_ref_1D
    M_face_locale = longueur_face * M_ref_1D
    
    # Récupération de tous les ddls de la face nF de T et de la face correspondante F_V de V
    liste_T = np.zeros(n_dof_face)
    liste_V = np.zeros(n_dof_face)
    for iloc_face in range(n_dof_face):
        iglob_T, iglob_V = get_face_dof_pair(T, F, iloc_face, ordre, neighbors, neighbor_faces)
        liste_T[iloc_face] = iglob_T
        liste_V[iloc_face] = iglob_V
    
    # Assemblage dans la matrice globale
    # ATTENTION: orientation opposée entre T et V
    for iloc_T in range(n_dof_face):
        # Indice global du iloc_T-ème DDL de la face F de T
        iglob_T = liste_T[iloc_T]
        
        for jloc_V in range(n_dof_face):
            # Indice global du jloc_V-ème DDL de la face F_V de V
            jglob_V = liste_V[jloc_V]
            
            M_TV.ajout(iglob_T, jglob_V, M_face_locale[iloc_T, jloc_V])
    
    return M_TV

def build_matrice_masse_frontière_DG(mesh,ordre):
    M_ref_1D= build_masse_ref_1D(ordre)
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]        
    loctoglob_DG, n_glob_DG = build_loctoglob_DG(triangles, ordre)
    dof_coords = build_dof_coordinates_DG(mesh, ordre)
    neighbors, _,_=build_neighborhood_structure(triangles)
    # Calcul des dimensions
    Nloc = (ordre + 1) * (ordre + 2) // 2
    Nglob = Nloc * len(triangles)
    
    # Calcul du nombre de faces de bord
    n_faces_bord = np.sum(neighbors < 0) 
    nnz = n_faces_bord * (ordre + 1)**2    
    # Création de la matrice globale
    M_boundary = COOMatrix(Nglob, Nglob, nnz)
    
    
    # Boucle sur tous les triangles
    for T in range(len(triangles)):
        # Boucle sur les 3 faces de chaque triangle
        for F in range(3):
            V = neighbors[T, F]
            if V == -1:  # Face de bord            
                MTT = build_masse_frontiere_elt_DG(T, F, ordre, loctoglob_DG, points, triangles, M_ref_1D)
                # Ajouter à la matrice globale
                M_boundary = M_boundary + MTT
        
    return M_boundary


def plot_nodal_vector_moins_fonction_DG(U,func, mesh, ordre, title):
    Nloc = (ordre + 1) * (ordre + 2) // 2
    # Recuperation de la géométrie du maillage
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]        
    loctoglob_DG, n_glob_DG = build_loctoglob_DG(triangles, ordre)
    dof_coords = build_dof_coordinates_DG(mesh, ordre)
    U_func = build_nodal_vector_DG(func, mesh, ordre)
    U_diff = U - U_func
    plot_nodal_vector_DG(U_diff, mesh, ordre, title)


def nombre_dof_DG(mesh,ordre):
    Nloc = (ordre + 1) * (ordre + 2) // 2
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]
    Nglob_DG = Nloc * len(triangles)
    return Nglob_DG

def build_jump_matrix_DG(mesh,ordre:int, verbose=False):
    """
    Assemble la matrice de saut globale sans les pondération hyperboliques.
    
    Parameters:
    -----------
    ordre : int
        Ordre polynomial
    triangles : ndarray
        Tableau des triangles (n_triangles x 3)
    points : ndarray
        Coordonnées des sommets (n_points x 2)
    neighbors : ndarray
        Tableau des voisins (n_triangles x 3)
    neighbor_faces : ndarray
        Tableau des faces voisines (n_triangles x 3)
    mesh : meshio.Mesh
        Maillage
    loctoglob_DG : ndarray
        Table de correspondance locale vers globale pour DG
    M_ref_1D : ndarray
        Matrice de masse de référence 1D sur [0,1]
    verbose : bool, optional
        Afficher les statistiques d'assemblage (défaut: True)
        
    Returns:
    --------
    MAT_saut : COOMatrix
        Matrice de saut globale (Nglob x Nglob)
    """
    M_ref_1D = build_masse_ref_1D(ordre)
    points = mesh.points[:, :2]  # On ne garde que les coordonnées x et y
    triangles = np.asarray(mesh.cells_dict["triangle"]) # mesh.cells_dict["triangle"]
    neighbors, neighbor_faces, edges_to_triangles = build_neighborhood_structure(triangles)
    loctoglob_DG,Nglob_DG = build_loctoglob_DG(triangles, ordre)
    # Calcul des dimensions
    Nloc = (ordre + 1) * (ordre + 2) //2
    
    # Estimation du nombre d'éléments non nuls
    n_faces_int_estimee = 3 * len(triangles)
    nnz_estime = 2 * n_faces_int_estimee * (ordre + 1)**2
    
    if verbose:
        print("\n" + "="*70)
        print("=== Assemblage de la matrice de saut globale ===")
        print("="*70)
        print(f"\nNombre de triangles : {len(triangles)}")
        print(f"Nombre de faces intérieures estimé : {n_faces_int_estimee}")
        print(f"Nombre d'éléments non nuls estimé : {nnz_estime}")
    
    # Création de la matrice globale de saut
    MAT_saut = COOMatrix(Nglob_DG, Nglob_DG, nnz_estime)
    
    # Compteurs pour les statistiques
    n_faces_int = 0
    n_faces_bord = 0
    
    # Boucle sur tous les triangles
    for T in range(len(triangles)):
        # Boucle sur les 3 faces de chaque triangle
        for F in range(3):
            V = neighbors[T, F]
            
            if V >= 0:  # Face intérieure
                n_faces_int += 1
                
                # Construire MTT : ∫_F φ_T^i φ_T^j
                MTT = build_masse_frontiere_elt_DG(T, F, ordre, loctoglob_DG, points, triangles, M_ref_1D)
                
                # Construire MTV : ∫_F φ_V^j φ_T^i
                MTV = build_boundary_mass_TV(T, F, ordre, triangles, points, neighbors, neighbor_faces, M_ref_1D)
                
                # Ajouter : ∫_F (u_T - u_V) v̄_T = ∫_F u_T v̄_T - ∫_F u_V v̄_T
                MAT_saut = MAT_saut + MTT   # +MTT
                MAT_saut = MAT_saut - MTV   # -MTV
                
            else:  # Face de bord
                n_faces_bord += 1
    
    if verbose:
        print(f"\n=== Statistiques d'assemblage ===")
        print(f"Faces intérieures traitées : {n_faces_int}")
        print(f"Faces de bord ignorées : {n_faces_bord}")
        print(f"Total de faces : {n_faces_int + n_faces_bord}")
        print(f"\nMatrice MAT_saut :")
        print(f"  Taille : {MAT_saut.nb_lig} x {MAT_saut.nb_col}")
        print(f"  Éléments non nuls utilisés : {MAT_saut.l}")
        print(f"  Éléments non nuls alloués : {MAT_saut.nnz}")
        print(f"  Taux de remplissage : {100 * MAT_saut.l / MAT_saut.nnz:.2f}%")
        print("\nAssemblage de la matrice de saut globale terminé")
    
    return MAT_saut

def terme_source_DG(func, mesh, ordre: int):
    """
    Construit le terme source en méthode DG :
        F_i = ∫_T f(x,y) φ_i(x,y) dx
    (aucun couplage entre éléments)

    Parameters
    ----------
    func : callable
        f(x,y)
    mesh : meshio.Mesh
    ordre : int

    Returns
    -------
    F_DG : ndarray (Nglob_DG,)
    """

    points = mesh.points[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])

    NT = len(triangles)
    Nloc = (ordre + 1) * (ordre + 2) // 2
    Nglob = NT * Nloc

    F_DG = np.zeros(Nglob, dtype=np.complex128)

    # --- quadrature (UNE seule fois) ---
    ordreq = 2 * ordre
    wq, xq, yq = quadrature_triangle_ref_2D(ordreq + 1)
    Nq = len(wq)

    # --- pré-évaluation des bases ---
    Phi = np.zeros((Nloc, Nq))
    for i in range(ordre + 1):
        for j in range(ordre + 1 - i):
            k = loc2D_to_loc1D(i, j)
            Phi[k, :] = base(xq, yq, i, j, ordre)

    # ==========================================================
    # Boucle éléments (aucun assemblage = écriture directe)
    # ==========================================================

    for T, (pt0, pt1, pt2) in enumerate(triangles):

        A0 = points[pt0]
        A1 = points[pt1]
        A2 = points[pt2]

        # transformation affine
        x_phys = A0[0] + xq * (A1[0] - A0[0]) + yq * (A2[0] - A0[0])
        y_phys = A0[1] + xq * (A1[1] - A0[1]) + yq * (A2[1] - A0[1])

        f_q = func(x_phys, y_phys)

        # jacobien
        Jac = abs((A1[0]-A0[0])*(A2[1]-A0[1]) - (A2[0]-A0[0])*(A1[1]-A0[1]))

        # intégrale locale vectorisée
        F_loc = Jac * (Phi @ (wq * f_q))

        # écriture directe DG (aucun scatter)
        offset = T * Nloc
        F_DG[offset:offset + Nloc] = F_loc

    return F_DG

def build_masse_variable_DG(mass_func, mesh, ordre: int):
    """
    Matrice de masse DG avec coefficient variable m(x,y)

        M_ij = ∫_T m(x,y) φ_i φ_j dx

    Parameters
    ----------
    mass_func : callable
        m(x,y)
    mesh : meshio.Mesh
    ordre : int

    Returns
    -------
    Mat : COOMatrix
    """

    points = mesh.points[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])

    NT = len(triangles)
    Nloc = (ordre + 1) * (ordre + 2) // 2
    Ndof = NT * Nloc

    Mat = COOMatrix(Ndof, Ndof, NT * Nloc * Nloc)

    # --- quadrature (UNE seule fois) ---
    ordreq = 2 * ordre
    wq, xq, yq = quadrature_triangle_ref_2D(ordreq + 1)
    Nq = len(wq)

    # --- pré-évaluation des bases ---
    Phi = np.zeros((Nloc, Nq))
    for i in range(ordre + 1):
        for j in range(ordre + 1 - i):
            k = loc2D_to_loc1D(i, j)
            Phi[k, :] = base(xq, yq, i, j, ordre)

    # ==========================================================
    # Boucle éléments DG (bloc locaux indépendants)
    # ==========================================================

    for T, (pt0, pt1, pt2) in enumerate(triangles):

        A0 = points[pt0]
        A1 = points[pt1]
        A2 = points[pt2]

        # transformation affine
        x_phys = A0[0] + xq * (A1[0] - A0[0]) + yq * (A2[0] - A0[0])
        y_phys = A0[1] + xq * (A1[1] - A0[1]) + yq * (A2[1] - A0[1])

        m_q = mass_func(x_phys, y_phys)

        Jac = abs((A1[0]-A0[0])*(A2[1]-A0[1]) - (A2[0]-A0[0])*(A1[1]-A0[1]))

        # --- masse locale variable ---
        # Mloc = ∫ m φ_i φ_j
        Wm = wq * m_q
        Mloc = Jac * ((Phi * Wm) @ Phi.T)

        offset = T * Nloc
        for i in range(Nloc):
            ig = offset + i
            for j in range(Nloc):
                Mat.ajout(ig, offset + j, Mloc[i, j])

    return Mat




def build_mixte_variable_DG(rho_func, mesh, ordre):
    """
    Construit les deux matrices mixtes DG :

        Ax_ij = ∫ rho(x,y) (∂φ_j/∂x) φ_i
        Ay_ij = ∫ rho(x,y) (∂φ_j/∂y) φ_i

    Returns
    -------
    Matx, Maty : COOMatrix
    """

    points = mesh.points[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])

    NT = len(triangles)
    Nloc = (ordre + 1) * (ordre + 2) // 2
    Ndof = NT * Nloc

    Matx = COOMatrix(Ndof, Ndof, NT * Nloc * Nloc)
    Maty = COOMatrix(Ndof, Ndof, NT * Nloc * Nloc)

    # ----------------------------------------------------------
    # Quadrature (UNE seule fois)
    # ----------------------------------------------------------
    ordreq = 2 * ordre
    wq, xq, yq = quadrature_triangle_ref_2D(ordreq + 1)
    Nq = len(wq)

    # ----------------------------------------------------------
    # Pré-évaluation des bases de référence
    # ----------------------------------------------------------
    Phi = np.zeros((Nloc, Nq))
    dPhix_hat = np.zeros((Nloc, Nq))
    dPhiy_hat = np.zeros((Nloc, Nq))

    for i in range(ordre + 1):
        for j in range(ordre + 1 - i):
            k = loc2D_to_loc1D(i, j)

            Phi[k, :] = base(xq, yq, i, j, ordre)

            # dérivées sur le triangle de référence
            dPhix_hat[k, :] = derivative_base(xq, yq, m=i, n=j, ordre=ordre, var='x')
            dPhiy_hat[k, :] = derivative_base(xq, yq, m=i, n=j, ordre=ordre, var='y')

    # ==========================================================
    # Boucle éléments DG
    # ==========================================================
    for T, (pt0, pt1, pt2) in enumerate(triangles):

        A0 = points[pt0]
        A1 = points[pt1]
        A2 = points[pt2]

        # Jacobien affine
        J = np.column_stack((A1 - A0, A2 - A0))
        detJ = abs(np.linalg.det(J))
        JinvT = np.linalg.inv(J).T

        # transformation des points de quadrature
        x_phys = A0[0] + xq * (A1[0] - A0[0]) + yq * (A2[0] - A0[0])
        y_phys = A0[1] + xq * (A1[1] - A0[1]) + yq * (A2[1] - A0[1])

        rho_q = rho_func(x_phys, y_phys)

        # ------------------------------------------------------
        # Gradient physique
        # ------------------------------------------------------
        dPhi_dx = JinvT[0, 0] * dPhix_hat + JinvT[0, 1] * dPhiy_hat
        dPhi_dy = JinvT[1, 0] * dPhix_hat + JinvT[1, 1] * dPhiy_hat

        # poids pondérés
        W = wq * rho_q

        # ------------------------------------------------------
        # Matrices locales
        # ------------------------------------------------------
        Ax_loc = detJ * ((Phi * W) @ dPhi_dx.T)
        Ay_loc = detJ * ((Phi * W) @ dPhi_dy.T)

        # insertion bloc DG
        offset = T * Nloc
        for i in range(Nloc):
            ig = offset + i
            for j in range(Nloc):
                jg = offset + j
                Matx.ajout(ig, jg, Ax_loc[i, j])
                Maty.ajout(ig, jg, Ay_loc[i, j])

    return Matx, Maty








def build_masse_frontiere_variable_DG(rho_func, mesh, ordre: int, domaine="all"):
    """
    Masse de frontière DG avec coefficient variable rho(x,y)

        M_ij = ∫_Γ rho(x,y) φ_i φ_j.conj ds

    Parameters
    ----------
    rho_func : callable
        rho(x,y)
    mesh : meshio.Mesh
    ordre : int
    domaine : str
        "all" ou nom physique (Dirichlet, Neumann, ...)

    Returns
    -------
    Mat : COOMatrix
    """

    n_gauss = 2 * ordre + 1

    points = mesh.points[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])

    neighbors, neighbor_faces, edges_to_triangles, reference_BC, bc_name = \
        build_neighborhood_structure_with_bc(mesh)

    NT = len(triangles)
    Nloc = (ordre + 1) * (ordre + 2) // 2
    Ndof = NT * Nloc


    if domaine == "all":
        n_faces = np.sum(neighbors < 0)
    else:
        n_faces = np.sum(neighbors == reference_BC(domaine))
    nnz = n_faces * (ordre + 1)**2

    Mat = COOMatrix(Ndof, Ndof, nnz)

    # --------------------------------------------------
    # Quadrature 1D (UNE seule fois)
    # --------------------------------------------------
    xi_gauss, w_gauss = np.polynomial.legendre.leggauss(n_gauss)
    sq = 0.5 * (xi_gauss + 1)  # transformation de [-1,1] à [0,1]
    wq = 0.5 * w_gauss
    # sq : points de quadrature dans [0,1]

    n_dof_face = ordre + 1
    # ------------------------------------------
    # Evaluation des bases restreintes à la face
    # ------------------------------------------
    PhiF = np.zeros((n_dof_face, len(sq)))
    for iloc_face in range(n_dof_face):
        for k in range(len(sq)):
            PhiF[iloc_face, k] = base_1D(sq[k], iloc_face, ordre)
    # ==================================================
    # Boucle éléments / faces
    # ==================================================
    for T in range(NT):
        # ------------------------------------------
        # Calcul de l'indice du début du bloc de T dans la matrice globale
        # ------------------------------------------
        offset = T * Nloc
        pt0, pt1, pt2 = triangles[T]
        A0 = points[pt0]
        A1 = points[pt1]
        A2 = points[pt2]

        for F in range(3):

            V = neighbors[T, F]

            if domaine == "all":
                TEST = V <= -1
            else:
                TEST = V == reference_BC(domaine)

            if not TEST:
                continue

            # ------------------------------------------
            # Géométrie de la face
            # ------------------------------------------
            if F == 0:
                P0, P1 = A1, A2
            elif F == 1:
                P0, P1 = A2, A0
            else:
                P0, P1 = A0, A1

            edge = P1 - P0
            longueur = np.linalg.norm(edge)

            # ------------------------------------------
            # Points physiques de quadrature
            # x(ξ) = P0 + ξ (P1-P0)
            # ------------------------------------------
            x_phys = P0[0] + sq * edge[0]
            y_phys = P0[1] + sq * edge[1]

            rho_q = rho_func(x_phys, y_phys)

            W = wq * rho_q * longueur   # ds = longueur dξ
                        # ------------------------------------------
            # Masse locale variable sur la face
            # ------------------------------------------
            Mloc = (PhiF * W) @ PhiF.T


            for iloc_face in range(n_dof_face):
                # coordonnées barycentriques du point de face
                # reconstruction (m,n) équivalent à ta logique iface_iglob
                if F == 0:
                    m = ordre - iloc_face
                    n = iloc_face
                    ilocT = loc2D_to_loc1D(m,n)
                elif F == 1:
                    m = 0
                    n = ordre - iloc_face
                    ilocT = loc2D_to_loc1D(m,n)
                else:
                    m = iloc_face
                    n = 0
                    ilocT = loc2D_to_loc1D(m,n)
                for jloc_face in range(n_dof_face):
                    # coordonnées barycentriques du point de face
                    # reconstruction (m,n) équivalent à ta logique iface_iglob
                    if F == 0:
                        m = ordre - jloc_face
                        n = jloc_face
                        jlocT = loc2D_to_loc1D(m,n)
                    elif F == 1:
                        m = 0
                        n = ordre - jloc_face
                        jlocT = loc2D_to_loc1D(m,n)
                    else:
                        m = jloc_face
                        n = 0
                        jlocT = loc2D_to_loc1D(m,n)
                    ig = offset + ilocT
                    jg = offset + jlocT
                    Mat.ajout(ig, jg, Mloc[iloc_face, jloc_face])

    return Mat

