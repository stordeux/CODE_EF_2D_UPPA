import pygmsh
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from matplotlib.patches import Polygon

def verifier_et_corriger_orientation(mesh):
    """
    Vérifie que tous les triangles du maillage sont orientés positivement 
    (sens trigonométrique) et corrige l'orientation si nécessaire.
    
    Un triangle (A0, A1, A2) est orienté positivement si:
    det = (A1-A0) × (A2-A0) > 0
    
    Si det < 0, on échange A1 et A2 pour corriger l'orientation.
    
    Paramètres:
    -----------
    mesh : meshio.Mesh
        Maillage contenant les points et triangles
        
    Retourne:
    ---------
    nb_corriges : int
        Nombre de triangles dont l'orientation a été corrigée
    """
    
    # Récupération des points et triangles
    points = mesh.points[:, :2]
    triangles = mesh.cells_dict['triangle']
    
    nb_corriges = 0
    nb_total = len(triangles)
    
    print(f"\n=== Vérification de l'orientation des triangles ===")
    print(f"Nombre total de triangles: {nb_total}")
    
    for ielt in range(nb_total):
        # Récupération des indices des sommets
        pt0, pt1, pt2 = triangles[ielt]
        
        # Récupération des coordonnées
        A0 = points[pt0]
        A1 = points[pt1]
        A2 = points[pt2]
        
        # Calcul du déterminant pour vérifier l'orientation
        # det = (A1-A0) × (A2-A0) en 2D
        det = (A1[0] - A0[0]) * (A2[1] - A0[1]) - (A1[1] - A0[1]) * (A2[0] - A0[0])
        
        # Si det < 0, le triangle est orienté négativement
        if det < 0:
            # Correction: échange des sommets 1 et 2
            triangles[ielt][1], triangles[ielt][2] = triangles[ielt][2], triangles[ielt][1]
            nb_corriges += 1
            
            if nb_corriges <= 5:  # Afficher seulement les 5 premiers
                print(f"  Triangle {ielt}: orientation négative corrigée (det={det:.6e})")
        elif det == 0:
            print(f"  Triangle {ielt}: dégénéré (det=0)!")
    
    print(f"\nNombre de triangles corrigés: {nb_corriges}/{nb_total}")
    
    if nb_corriges > 0:
        print("Tous les triangles étaient déjà correctement orientés", False)
        print("✓ Les orientations ont été corrigées dans le maillage")
    else:
        print("Tous les triangles étaient déjà correctement orientés", True)
    
    return nb_corriges

def create_mesh_circle_in_square(radius=0.2, square_size=1.0, mesh_size=0.2):
    with pygmsh.geo.Geometry() as geom:
        # Définir les points du carré
        p1 = geom.add_point([-square_size / 2, -square_size / 2, 0], mesh_size)
        p2 = geom.add_point([ square_size / 2, -square_size / 2, 0], mesh_size)
        p3 = geom.add_point([ square_size / 2,  square_size / 2, 0], mesh_size)
        p4 = geom.add_point([-square_size / 2,  square_size / 2, 0], mesh_size)
        
        # Définir les lignes du carré
        l1 = geom.add_line(p1, p2)
        l2 = geom.add_line(p2, p3)
        l3 = geom.add_line(p3, p4)
        l4 = geom.add_line(p4, p1)
        
        # Créer un loop pour le carré
        square_loop = geom.add_curve_loop([l1, l2, l3, l4])
        
        # Définir le cercle (trou)
        circle = geom.add_circle([0, 0, 0], radius, mesh_size=mesh_size, make_surface=False)
        
        # Le cercle a déjà un curve_loop, utiliser directement
        circle_loop = circle.curve_loop
        
        # Créer la surface entre le carré et le cercle
        surface = geom.add_plane_surface(square_loop, holes=[circle_loop])
        
        # Générer le maillage
        mesh = geom.generate_mesh()
        mesh = verifier_et_corriger_orientation(mesh)


        # -------------------------------
        # AJOUT DES RÉFÉRENCES DE FRONTIÈRE
        # -------------------------------

        # Référence 1 : frontière extérieure (carré)
        geom.add_physical([l1, l2, l3, l4], label="-2")

        # Référence 2 : frontière intérieure (cercle)
        geom.add_physical(circle_loop.curves, label="-3")

        # (optionnel mais fortement conseillé)
        # Référence du domaine
        geom.add_physical([surface], label="10")

        # Générer le maillage
        mesh = geom.generate_mesh()
        
    return mesh

def triangle_area(p1, p2, p3):
    """
    Aire d'un triangle 2D défini par p1, p2, p3
    p1, p2, p3 : array-like (2,)
    """
    return 0.5 * abs(
        (p2[0] - p1[0]) * (p3[1] - p1[1])
        - (p2[1] - p1[1]) * (p3[0] - p1[0])
    )

def check_triangle_areas(nodes, triangles, tol=1e-12):
    areas = np.zeros(len(triangles))
    bad = []

    for k, tri in enumerate(triangles):
        p1, p2, p3 = nodes[tri]
        area = triangle_area(p1, p2, p3)
        areas[k] = area

        if area < tol:
            bad.append(k)

    print("Aire min :", areas.min())
    print("Aire max :", areas.max())
    print("Aires < tol :", len(bad))

    return areas, bad

def plot_mesh(mesh):
    triangles = mesh.cells_dict['triangle']
    points = mesh.points[:, :2]
    # Calcul du nombre d'éléments et de sommets
    num_elements = len(triangles)
    num_nodes = len(points)
    print(f"Nombre d'éléments : {num_elements}")
    print(f"Nombre de sommets : {num_nodes}")
    plt.triplot(points[:, 0], points[:, 1], triangles)
    plt.gca().set_aspect('equal')
    plt.title("Maillage du domaine avec un trou circulaire")
    plt.show()

def build_neighborhood_structure(triangles):
    """
    Construit la structure de voisinage du maillage
    
    Parameters:
    -----------
    triangles : array (n_triangles, 3)
        Indices des sommets de chaque triangle
        
    Returns:
    --------
    neighbors : array (n_triangles, 3)
        Pour chaque triangle, indices des 3 triangles voisins
        neighbors[i, j] = indice du triangle voisin par l'arête opposée au sommet j
        Si pas de voisin (bord), neighbors[i, j] = -1
    neighbor_faces : array (n_triangles, 3)
        Pour chaque face, numéro de la face correspondante dans le triangle voisin
        neighbor_faces[i, j] = numéro de face (0, 1 ou 2) dans le triangle voisin
        Si pas de voisin (bord), neighbor_faces[i, j] = -1
    edges_to_triangles : dict
        Dictionnaire {edge: [triangles]} où edge = tuple(min(v1,v2), max(v1,v2))
    """
    n_triangles = len(triangles)
    neighbors = -np.ones((n_triangles, 3), dtype=int)
    neighbor_faces = -np.ones((n_triangles, 3), dtype=int)
    
    # Dictionnaire pour stocker les arêtes
    # Clé : tuple (min_vertex, max_vertex)
    # Valeur : liste de (triangle_idx, local_edge_idx)
    edges_to_triangles = {}
    
    # Premier passage : recenser toutes les arêtes
    for tri_idx, triangle in enumerate(triangles):
        # Les 3 arêtes du triangle
        edges = [
            (triangle[1], triangle[2]),  # Arête opposée au sommet 0
            (triangle[2], triangle[0]),  # Arête opposée au sommet 1
            (triangle[0], triangle[1])   # Arête opposée au sommet 2
        ]
        
        for local_edge_idx, (v1, v2) in enumerate(edges):
            # Normaliser l'arête (min, max) pour la rendre unique
            edge = tuple(sorted([v1, v2]))
            
            if edge not in edges_to_triangles:
                edges_to_triangles[edge] = []
            edges_to_triangles[edge].append((tri_idx, local_edge_idx))
    
    # Deuxième passage : construire la table de voisinage
    for edge, tri_list in edges_to_triangles.items():
        if len(tri_list) == 2:
            # Arête interne : 2 triangles adjacents
            (tri1, edge1), (tri2, edge2) = tri_list
            neighbors[tri1, edge1] = tri2
            neighbors[tri2, edge2] = tri1
            # Stockage des numéros de faces correspondantes
            neighbor_faces[tri1, edge1] = edge2
            neighbor_faces[tri2, edge2] = edge1
        # Si len(tri_list) == 1 : arête de bord, reste à -1
    
    return neighbors, neighbor_faces, edges_to_triangles


def voisinage_reciproque(neighbors,triangles):
    message = ""
    TEST = False
    NT = len(triangles)
    for iT in range(len(triangles)):
        iV = neighbors[iT]
        for V in range(3):
            if iV[V] != -1:        
                TEST = iT in neighbors[iV[V]]
            if TEST == False:
                message += "Problème dans la table de voisinage\n"
                message += "Element " + str(iT) + " voisin " + str(V) + " : " + str(iV[V]) + "\n"
    if message == "":
        print("Table de voisinage correcte",True)
    else:
        print(message)
        print("Table de voisinage correcte",False)

def plot_un_trianlge_et_ses_voisins(points, triangles, neighbors, iT):
    # Test du maillage 
    pt1,pt2,pt3=triangles[iT]
    A1= points[pt1]
    A2= points[pt2]
    A3= points[pt3]

    # trace du triangle qui relie A1, A2, A3
    plt.plot([A1[0], A2[0]], [A1[1], A2[1]], 'k-')
    plt.plot([A2[0], A3[0]], [A2[1], A3[1]], 'k-')
    plt.plot([A3[0], A1[0]], [A3[1], A1[1]], 'k-')
    # Affichage de 1 2 3 sur les sommets
    plt.text(A1[0], A1[1], '0', fontsize=12, ha='right')
    plt.text(A2[0], A2[1], '1', fontsize=12,ha='left')
    plt.text(A3[0], A3[1], '2', fontsize=12,ha='center')

    for k in range(3):
        jT = neighbors[iT, k]
        if jT == -1:
            continue

        tr2 = triangles[jT]
        pt1,pt2,pt3=tr2
        A1= points[pt1]
        A2= points[pt2]
        A3= points[pt3]
        plt.plot([A1[0], A2[0]], [A1[1], A2[1]], 'r-')
        plt.plot([A2[0], A3[0]], [A2[1], A3[1]], 'r-')
        plt.plot([A3[0], A1[0]], [A3[1], A1[1]], 'r-')
        # Affichage de 1 2 3 sur les sommets
        G=(A1+A2+A3)/3
        plt.text(G[0], G[1], f'neigh {k}', fontsize=12, ha='center')

    plt.gca().set_aspect('equal')

def calcul_normale(A0,A1,A2,i):
    # Calcule la normale sortante à l'arête opposée au sommet i
    # A0, A1, A2 : coordonnées des sommets du triangle
    if i==0:
        P1=A1
        P2=A2
    elif i==1:
        P1=A2
        P2=A0
    else:
        P1=A0
        P2=A1
    T=P2-P1
    n=np.array([-T[1],T[0]])
    n=n/np.linalg.norm(n)
    # On vérifie que la normale est sortante
    G=(A0+A1+A2)/3
    M=(P1+P2)/2
    AM= M - G
    if np.dot(AM,n)<0:
        n=-n
    return n


# Tracé d'un triangle et de ses trois normales
def plot_triangle_with_normals(A0:ArrayLike, A1:ArrayLike, A2:ArrayLike, figsize=(10, 10)):
    """
    Trace un triangle et ses trois normales sortantes
    
    Parameters:
    -----------
    A0, A1, A2 : array ou tuple
        Coordonnées des trois sommets du triangle
    figsize : tuple
        Taille de la figure
    """
    A0 = np.array(A0,dtype=float)
    A1 = np.array(A1,dtype=float)
    A2 = np.array(A2,dtype=float)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Tracer le triangle
    triangle = Polygon([A0, A1, A2], fill=False, edgecolor='black', linewidth=2) # type: ignore
    ax.add_patch(triangle)
    
    # Tracer les sommets
    ax.plot([A0[0], A1[0], A2[0]], [A0[1], A1[1], A2[1]], 'ro', markersize=10)
    ax.text(A0[0], A0[1], '  A0', fontsize=14, ha='left')
    ax.text(A1[0], A1[1], '  A1', fontsize=14, ha='left')
    ax.text(A2[0], A2[1], '  A2', fontsize=14, ha='left')
    
    # Calculer et tracer les normales pour chaque arête
    colors = ['red', 'green', 'blue']
    labels = ['Normale opposée à A0', 'Normale opposée à A1', 'Normale opposée à A2']
    
    for i in range(3):
        # Calculer la normale
        n = calcul_normale(A0, A1, A2, i)
        
        # Déterminer le point d'application (milieu de l'arête)
        if i == 0:
            M = (A1 + A2) / 2
        elif i == 1:
            M = (A2 + A0) / 2
        else:
            M = (A0 + A1) / 2
        
        # Tracer la normale (vecteur de longueur 0.3)
        scale = 0.3
        ax.arrow(M[0], M[1], n[0]*scale, n[1]*scale, 
                head_width=0.05, head_length=0.05, 
                fc=colors[i], ec=colors[i], linewidth=2,
                label=labels[i])
        
        # Afficher les coordonnées de la normale
        ax.text(M[0] + n[0]*scale*1.2, M[1] + n[1]*scale*1.2, 
               f'n{i}\n({n[0]:.2f}, {n[1]:.2f})', 
               fontsize=10, ha='center', color=colors[i])
    
    # Centre de gravité
    G = (A0 + A1 + A2) / 3
    ax.plot(G[0], G[1], 'k*', markersize=15, label='Centre de gravité')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_title('Triangle et ses normales sortantes', fontsize=16)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def plot_structure_voisinage(mesh, ielt_test=None, ax=None):
    """Test de la structure de voisinage avec visualisation des normales."""
    print("\n=== Test de la structure de voisinage ===\n")
    points = mesh.points[:, :2]
    triangles = mesh.cells_dict["triangle"]
    # Construction de la structure de voisinage
    neighbors, neighbor_faces, edges_to_triangles = build_neighborhood_structure(triangles)
    # Tirage d'un element au hasard (eviter les elements de bord)
    # Chercher un element qui a 3 voisins
    if ielt_test is None:
        for i in range(len(triangles)):
            if np.all(neighbors[i] >= 0):  # Toutes les faces ont un voisin
                ielt_test = i
                break

    if ielt_test is None:
        # Si aucun element n'a 3 voisins, prendre un au hasard
        ielt_test = np.random.randint(0, len(triangles))
        print("Aucun triangle avec 3 voisins trouve, tirage aleatoire")

    print(f"Triangle teste : {ielt_test}")

    # Recuperation des sommets du triangle
    pt0, pt1, pt2 = triangles[ielt_test]
    A0 = points[pt0]
    A1 = points[pt1]
    A2 = points[pt2]

    print(f"Sommets : A0={pt0}, A1={pt1}, A2={pt2}")
    print(f"Voisins : {neighbors[ielt_test]}")
    print(f"Faces voisines : {neighbor_faces[ielt_test]}")

    # Creation de la figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure

    # Trace du triangle central en gras
    triangle_central = Polygon([A0, A1, A2], fill=False, edgecolor='black', linewidth=3)  # type: ignore
    ax.add_patch(triangle_central)

    # Annotation des sommets du triangle central
    ax.plot(*A0, 'ro', markersize=12, label='Triangle central', zorder=10)
    ax.plot(*A1, 'ro', markersize=12, zorder=10)
    ax.plot(*A2, 'ro', markersize=12, zorder=10)
    ax.text(A0[0], A0[1], f'  A0 (T{ielt_test})', fontsize=11, fontweight='bold')
    ax.text(A1[0], A1[1], f'  A1 (T{ielt_test})', fontsize=11, fontweight='bold')
    ax.text(A2[0], A2[1], f'  A2 (T{ielt_test})', fontsize=11, fontweight='bold')

    # Centre du triangle central pour les normales
    cx_central = (A0[0] + A1[0] + A2[0]) / 3
    cy_central = (A0[1] + A1[1] + A2[1]) / 3
    _ = (cx_central, cy_central)

    # Couleurs pour les 3 faces
    couleurs_faces = ['red', 'green', 'blue']
    noms_faces = ['Face 0 (opp. A0)', 'Face 1 (opp. A1)', 'Face 2 (opp. A2)']

    print(f"\n--- Analyse des 3 faces du triangle {ielt_test} ---")

    # Boucle sur les 3 faces du triangle central
    for iface in range(3):
        couleur = couleurs_faces[iface]

        # Calcul de la normale sortante du triangle central par la face iface
        n_central = calcul_normale(A0, A1, A2, iface)

        # Recuperation du voisin
        ielt_voisin = neighbors[ielt_test, iface]
        iface_voisin = neighbor_faces[ielt_test, iface]

        print(f"\n{noms_faces[iface]}:")
        print(f"  Voisin : T{ielt_voisin}")

        if ielt_voisin >= 0:
            print(f"  Face commune dans le voisin : Face {iface_voisin}")

            # Recuperation des sommets du triangle voisin
            pt0_v, pt1_v, pt2_v = triangles[ielt_voisin]
            A0_v = points[pt0_v]
            A1_v = points[pt1_v]
            A2_v = points[pt2_v]

            # Trace du triangle voisin
            triangle_voisin = Polygon(
                [A0_v, A1_v, A2_v],
                fill=False,  # type: ignore
                edgecolor=couleur,
                linewidth=2,
                alpha=0.7,
                linestyle='--',
            )
            ax.add_patch(triangle_voisin)

            # Centre du triangle voisin
            cx_voisin = (A0_v[0] + A1_v[0] + A2_v[0]) / 3
            cy_voisin = (A0_v[1] + A1_v[1] + A2_v[1]) / 3
            ax.text(
                cx_voisin,
                cy_voisin,
                f'T{ielt_voisin}',
                fontsize=10,
                ha='center',
                va='center',
                color=couleur,
                fontweight='bold',
            )

            # Calcul de la normale sortante du voisin par sa face commune
            n_voisin = calcul_normale(A0_v, A1_v, A2_v, iface_voisin)

            print(f"  Normale centrale : n={n_central}")
            print(f"  Normale voisin   : n={n_voisin}")
            print(f"  Produit scalaire : {np.dot(n_central, n_voisin):.6f} (attendu ~= -1)")

            # Point de depart des normales : milieu de la face commune
            if iface == 0:
                milieu = (A1 + A2) / 2
            elif iface == 1:
                milieu = (A2 + A0) / 2
            else:
                milieu = (A0 + A1) / 2

            # Longueur de la fleche (proportionnelle a la taille du triangle)
            scale = 0.15 * np.linalg.norm(A1 - A0)

            # Fleche normale du triangle central (sortante)
            ax.arrow(
                milieu[0],
                milieu[1],
                scale * n_central[0],
                scale * n_central[1],
                head_width=scale * 0.3,
                head_length=scale * 0.2,
                fc=couleur,
                ec=couleur,
                linewidth=2,
                label=f'{noms_faces[iface]} (T{ielt_test})' if iface == 0 else None,
            )

            # Fleche normale du triangle voisin (sortante)
            ax.arrow(
                milieu[0],
                milieu[1],
                scale * n_voisin[0],
                scale * n_voisin[1],
                head_width=scale * 0.3,
                head_length=scale * 0.2,
                fc=couleur,
                ec=couleur,
                linewidth=2,
                alpha=0.5,
                linestyle='--',
            )
        else:
            print("  Arete de bord (pas de voisin)")

            # Point de depart de la normale : milieu de l'arete de bord
            if iface == 0:
                milieu = (A1 + A2) / 2
            elif iface == 1:
                milieu = (A2 + A0) / 2
            else:
                milieu = (A0 + A1) / 2

            # Longueur de la fleche
            scale = 0.15 * np.linalg.norm(A1 - A0)

            # Fleche normale (sortante vers l'exterieur du domaine)
            ax.arrow(
                milieu[0],
                milieu[1],
                scale * n_central[0],
                scale * n_central[1],
                head_width=scale * 0.3,
                head_length=scale * 0.2,
                fc=couleur,
                ec=couleur,
                linewidth=2,
                label=f'{noms_faces[iface]} (bord)',
            )

    # Configuration de l'affichage
    ax.set_aspect('equal')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f'Structure de voisinage - Triangle {ielt_test} et ses voisins\n'
        + 'Les normales opposees doivent etre de sens contraire'
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    fig.tight_layout()
    plt.show()

    return ielt_test

import numpy as np

def build_spatial_grid(mesh, nx=None, ny=None):
    """
    Construit une grille spatiale uniforme pour accélérer la localisation.
    Version robuste aux erreurs d'arrondi flottant.
    """

    pts = mesh.points[:, :2]
    tris = np.asarray(mesh.cells_dict["triangle"])

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)

    NT = len(tris)

    if nx is None:
        nx = int(np.sqrt(NT))
    if ny is None:
        ny = nx

    hx = (xmax - xmin) / nx
    hy = (ymax - ymin) / ny

    # Tolérance géométrique (très petite, relative au pas)
    epsx = 1e-12
    epsy = 1e-12

    grid = [[[] for _ in range(ny)] for _ in range(nx)]

    for T, (i0, i1, i2) in enumerate(tris):
        P = pts[[i0, i1, i2]]

        txmin, tymin = P.min(axis=0)
        txmax, tymax = P.max(axis=0)

        # floor obligatoire + léger élargissement de la bounding box
        ix0 = int(np.floor((txmin - xmin)/hx - epsx))
        ix1 = int(np.floor((txmax - xmin)/hx + epsx))

        iy0 = int(np.floor((tymin - ymin)/hy - epsy))
        iy1 = int(np.floor((tymax - ymin)/hy + epsy))

        ix0 = max(ix0, 0); ix1 = min(ix1, nx-1)
        iy0 = max(iy0, 0); iy1 = min(iy1, ny-1)

        for ix in range(ix0, ix1+1):
            for iy in range(iy0, iy1+1):
                grid[ix][iy].append(T)

    return {
        "grid": grid,
        "xmin": xmin, "ymin": ymin,
        "hx": hx, "hy": hy,
        "nx": nx, "ny": ny
    }


def candidate_triangles(x, y, spgrid):
    """
    Retourne la liste des triangles candidats pour le point (x,y).
    Version robuste aux erreurs d'arrondi flottant.
    """

    xmin = spgrid["xmin"]; ymin = spgrid["ymin"]
    hx   = spgrid["hx"];   hy   = spgrid["hy"]
    nx   = spgrid["nx"];   ny   = spgrid["ny"]

    # floor obligatoire (et sans epsilon)
    ix = int(np.floor((x - xmin)/hx))
    iy = int(np.floor((y - ymin)/hy))

    # Clamp pour éviter les débordements en bord de domaine
    ix = max(0, min(ix, nx-1))
    iy = max(0, min(iy, ny-1))

    return spgrid["grid"][ix][iy]




