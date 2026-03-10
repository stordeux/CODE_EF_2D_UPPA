import pygmsh
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from matplotlib.patches import Polygon
import meshio, tempfile, os

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

def create_mesh_circle_in_square_old(radius=0.2, square_size=1.0, mesh_size=0.2):
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

def create_mesh_polygon_with_hole(outer_points, inner_points, mesh_size=0.05):
    """
    Génère un maillage 2D d’un polygone contenant un trou polygonal.

    Géométrie
    ---------
    Ω = polygone extérieur
    trou = polygone intérieur

    Frontières physiques
    --------------------
    - "FOURIER" : frontière extérieure
    - "NEUMANN" : frontière du trou
    - "OMEGA"   : domaine

    Paramètres
    ----------
    outer_points : list[(x,y)]
        Sommets du polygone extérieur (ordre le long du bord).
    inner_points : list[(x,y)]
        Sommets du polygone intérieur.
    mesh_size : float
        Taille caractéristique du maillage.

    Retour
    ------
    mesh : meshio.Mesh
    """

    import gmsh
    import meshio
    import tempfile
    import os

    gmsh.initialize()
    gmsh.model.add("polygon_with_hole")

    geo = gmsh.model.geo

    # --------------------------------------------------
    # 1. Polygone extérieur
    # --------------------------------------------------

    outer_gmsh_pts = [
        geo.addPoint(x, y, 0, mesh_size)
        for x, y in outer_points
    ]

    outer_lines = []
    n = len(outer_gmsh_pts)

    for i in range(n):
        p1 = outer_gmsh_pts[i]
        p2 = outer_gmsh_pts[(i + 1) % n]
        outer_lines.append(geo.addLine(p1, p2))

    outer_loop = geo.addCurveLoop(outer_lines)

    # --------------------------------------------------
    # 2. Polygone intérieur (trou)
    # --------------------------------------------------

    inner_gmsh_pts = [
        geo.addPoint(x, y, 0, mesh_size)
        for x, y in inner_points
    ]

    inner_lines = []
    m = len(inner_gmsh_pts)

    for i in range(m):
        p1 = inner_gmsh_pts[i]
        p2 = inner_gmsh_pts[(i + 1) % m]
        inner_lines.append(geo.addLine(p1, p2))

    inner_loop = geo.addCurveLoop(inner_lines)

    # --------------------------------------------------
    # 3. Surface perforée
    # --------------------------------------------------

    surface = geo.addPlaneSurface([outer_loop, inner_loop])

    geo.synchronize()

    # --------------------------------------------------
    # 4. Groupes physiques
    # --------------------------------------------------

    tag_fourier = gmsh.model.addPhysicalGroup(1, outer_lines)
    gmsh.model.setPhysicalName(1, tag_fourier, "FOURIER")

    tag_neumann = gmsh.model.addPhysicalGroup(1, inner_lines)
    gmsh.model.setPhysicalName(1, tag_neumann, "NEUMANN")

    tag_domain = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, tag_domain, "OMEGA")

    # --------------------------------------------------
    # 5. Génération du maillage
    # --------------------------------------------------

    gmsh.model.mesh.generate(2)

    tmp = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
    tmp.close()

    gmsh.write(tmp.name)
    gmsh.finalize()

    mesh = meshio.read(tmp.name)
    os.remove(tmp.name)

    verifier_et_corriger_orientation(mesh)

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

def plot_mesh(mesh,secondes=None):
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
    if secondes is not None:
        # fermeture automatique après  secondes secondes
        timer = plt.gcf().canvas.new_timer(interval=1000*secondes)
        timer.add_callback(plt.close)
        timer.start()
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


def create_mesh_circle_in_square(radius=0.2, square_size=1.0, mesh_size=0.05):
    """
    Génère un maillage 2D d’un carré contenant un trou circulaire centré.

    Géométrie :
        Ω = carré de côté `square_size`
        trou = disque de rayon `radius` centré en (0,0)

    Frontières physiques définies :
        - "FOURIER" : bord extérieur du carré
        - "NEUMANN" : bord du trou circulaire
        - "OMEGA"   : domaine 2D

    Paramètres
    ----------
    radius : float
        Rayon du trou circulaire.
    square_size : float
        Longueur du côté du carré.
    mesh_size : float
        Taille caractéristique du maillage (gmsh).

    Retour
    ------
    mesh : meshio.Mesh
        Maillage triangulaire avec groupes physiques conservés.
    """

    import gmsh
    import meshio
    import tempfile
    import os

    # ============================================================
    # Initialisation de gmsh
    # ============================================================
    gmsh.initialize()
    gmsh.model.add("square_with_hole")

    # ============================================================
    # 1. Définition de la géométrie du carré
    # ============================================================
    # On construit explicitement les 4 sommets et les 4 arêtes.
    # Ces arêtes formeront la frontière extérieure du domaine.

    geo = gmsh.model.geo

    p1 = geo.addPoint(-square_size/2, -square_size/2, 0, mesh_size)
    p2 = geo.addPoint( square_size/2, -square_size/2, 0, mesh_size)
    p3 = geo.addPoint( square_size/2,  square_size/2, 0, mesh_size)
    p4 = geo.addPoint(-square_size/2,  square_size/2, 0, mesh_size)

    l1 = geo.addLine(p1, p2)
    l2 = geo.addLine(p2, p3)
    l3 = geo.addLine(p3, p4)
    l4 = geo.addLine(p4, p1)

    # Boucle fermée représentant le bord extérieur
    outer_loop = geo.addCurveLoop([l1, l2, l3, l4])

    # ============================================================
    # 2. Construction du cercle intérieur
    # ============================================================
    # IMPORTANT :
    # On utilise 4 arcs de cercle GEO au lieu de addCircle
    # pour garantir une bonne compatibilité avec meshio
    # et conserver correctement les tags de frontières.

    pc = geo.addPoint(0, 0, 0, mesh_size)

    p5 = geo.addPoint( radius, 0, 0, mesh_size)
    p6 = geo.addPoint( 0, radius, 0, mesh_size)
    p7 = geo.addPoint(-radius, 0, 0, mesh_size)
    p8 = geo.addPoint( 0,-radius, 0, mesh_size)

    c1 = geo.addCircleArc(p5, pc, p6)
    c2 = geo.addCircleArc(p6, pc, p7)
    c3 = geo.addCircleArc(p7, pc, p8)
    c4 = geo.addCircleArc(p8, pc, p5)

    # Boucle représentant le bord du trou
    inner_loop = geo.addCurveLoop([c1, c2, c3, c4])

    # ============================================================
    # 3. Définition de la surface perforée
    # ============================================================
    # gmsh construit la surface comme :
    # surface = outer_loop - inner_loop
    # c'est-à-dire un carré auquel on retire le disque.

    surface = geo.addPlaneSurface([outer_loop, inner_loop])

    geo.synchronize()

    # ============================================================
    # 4. Groupes physiques
    # ============================================================
    # Les groupes physiques servent à identifier les frontières
    # lors de l’assemblage FEM (conditions aux limites).

    # Bord extérieur : condition de Fourier
    tag_fourier = gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4])
    gmsh.model.setPhysicalName(1, tag_fourier, "FOURIER")

    # Bord intérieur : condition de Neumann
    tag_neumann = gmsh.model.addPhysicalGroup(1, [c1, c2, c3, c4])
    gmsh.model.setPhysicalName(1, tag_neumann, "NEUMANN")

    # Domaine 2D
    tag_domain = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, tag_domain, "OMEGA")

    # ============================================================
    # 5. Génération du maillage triangulaire
    # ============================================================
    gmsh.model.mesh.generate(2)

    # ============================================================
    # 6. Écriture temporaire du fichier .msh
    # ============================================================
    # Étape importante :
    # meshio lit correctement les groupes physiques seulement
    # si le maillage est écrit puis relu depuis un fichier.

    tmp = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
    tmp.close()

    gmsh.write(tmp.name)
    gmsh.finalize()

    # Lecture via meshio avec conservation des tags gmsh
    mesh = meshio.read(tmp.name)
    os.remove(tmp.name)

    # ============================================================
    # 7. Correction éventuelle de l’orientation des triangles
    # ============================================================
    # Garantit une orientation cohérente pour les intégrales FEM
    verifier_et_corriger_orientation(mesh)

    return mesh

def create_mesh_from_polygon(points, mesh_size=0.05):
    """
    Génère un maillage 2D triangulaire à partir d'un polygone.

    Géométrie
    ---------
    Ω = domaine polygonal défini par la liste de sommets `points`.

    Frontières physiques
    --------------------
    - "FOURIER" : frontière extérieure du domaine
    - "OMEGA"   : domaine 2D

    Paramètres
    ----------
    points : list of tuple
        Liste des sommets du polygone [(x1,y1), (x2,y2), ...]
        Les points doivent être ordonnés le long du bord.
    mesh_size : float
        Taille caractéristique du maillage.

    Retour
    ------
    mesh : meshio.Mesh
        Maillage triangulaire avec groupes physiques.
    """

    import gmsh
    import meshio
    import tempfile
    import os

    gmsh.initialize()
    gmsh.model.add("polygon")

    geo = gmsh.model.geo

    # ============================================================
    # 1. Création des points
    # ============================================================

    gmsh_points = []
    for x, y in points:
        gmsh_points.append(geo.addPoint(x, y, 0, mesh_size))

    # ============================================================
    # 2. Création des arêtes
    # ============================================================

    lines = []
    n = len(gmsh_points)

    for i in range(n):
        p1 = gmsh_points[i]
        p2 = gmsh_points[(i + 1) % n]   # fermeture automatique
        lines.append(geo.addLine(p1, p2))

    # Boucle fermée
    loop = geo.addCurveLoop(lines)

    # Surface du domaine
    surface = geo.addPlaneSurface([loop])

    geo.synchronize()

    # ============================================================
    # 3. Groupes physiques
    # ============================================================

    tag_fourier = gmsh.model.addPhysicalGroup(1, lines)
    gmsh.model.setPhysicalName(1, tag_fourier, "FOURIER")

    tag_domain = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, tag_domain, "OMEGA")

    # ============================================================
    # 4. Génération du maillage
    # ============================================================

    gmsh.model.mesh.generate(2)

    # ============================================================
    # 5. Sauvegarde temporaire (pour conserver les tags gmsh)
    # ============================================================

    tmp = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
    tmp.close()

    gmsh.write(tmp.name)
    gmsh.finalize()

    mesh = meshio.read(tmp.name)
    os.remove(tmp.name)

    # Correction orientation triangles
    verifier_et_corriger_orientation(mesh)

    return mesh

def build_bc_from_gmsh(mesh, name_to_code):
    """
    Construit le dictionnaire
        bc_from_edge[(i,j)] = code BC
    à partir des Physical Groups Gmsh.
    """

    lines = mesh.cells_dict["line"]
    tags  = mesh.cell_data_dict["gmsh:physical"]["line"]
    field = mesh.field_data

    # mapping ID gmsh -> code interne (-1/-2/-3)
    gmsh_id_to_code = {}

    for name, code in name_to_code.items():
        gmsh_id = field[name][0]
        gmsh_id_to_code[gmsh_id] = code

    bc_from_edge = {}

    for (i, j), tag in zip(lines, tags):
        edge = tuple(sorted((int(i), int(j))))
        bc_from_edge[edge] = gmsh_id_to_code[tag]

    return bc_from_edge

def inject_bc_into_neighbors(neighbors, edges_to_triangles, bc_from_edge):
    """
    Injecte les conditions aux limites dans la table de voisinage.

    Parameters
    ----------
    neighbors : (NT,3) ndarray
        Table de voisinage topologique (-1 = bord)

    edges_to_triangles : dict
        edge -> [(triangle, face_locale), ...]

    bc_from_edge : dict
        edge -> code BC (-2, -3, ...)

    Returns
    -------
    neighbors : ndarray modifié (in-place)
    """

    for edge, tri_list in edges_to_triangles.items():

        # arête frontière = un seul triangle attaché
        if len(tri_list) == 1:
            (iT, iF) = tri_list[0]

            if edge not in bc_from_edge:
                raise ValueError(f"Boundary edge {edge} has no BC assigned")

            neighbors[iT, iF] = bc_from_edge[edge]

    return neighbors




def build_neighborhood_structure_with_bc_old(mesh, name_to_code):
    """
    Construit la connectivité de voisinage d’un maillage triangulaire
    et remplace les faces de bord par les codes de conditions aux limites.

    Cette fonction réalise TROIS étapes distinctes :

    1) Construction de la connectivité purement topologique du maillage
       (indépendante de Gmsh) :

           - détection des arêtes globales
           - identification des triangles adjacents
           - remplissage de la table de voisinage

    2) Lecture des Physical Groups Gmsh définis sur les arêtes (dim = 1)
       afin d’associer chaque arête frontière à une condition aux limites.

    3) Injection de ces conditions aux limites dans la table de voisinage :
       une face frontière n’est plus marquée -1 mais par un code utilisateur
       (ex: -2 = Fourier, -3 = Neumann).

    Parameters
    ----------
    mesh : meshio.Mesh
        Maillage issu de Gmsh contenant :
            - des triangles (cells_dict["triangle"])
            - des arêtes physiques (cells_dict["line"])
            - les tags Gmsh (cell_data_dict["gmsh:physical"])

    name_to_code : dict[str, int]
        Mapping entre le nom Gmsh de la frontière et le code interne utilisé
        par le solveur.
        Exemple :
            {
                "FOURIER": -2,
                "NEUMANN": -3,
            }

    Returns
    -------
    neighbors : ndarray (NT,3)
        Table de voisinage par triangle.

        neighbors[iT, iF] =
            indice du triangle voisin si la face est interne
            code de BC (<0) si la face est frontière

        Convention locale :
            iF = 0 : face opposée au sommet 0  (edge v1–v2)
            iF = 1 : face opposée au sommet 1  (edge v2–v0)
            iF = 2 : face opposée au sommet 2  (edge v0–v1)

    neighbor_faces : ndarray (NT,3)
        Correspondance des faces entre triangles adjacents.

        Si neighbors[iT,iF] = jT >= 0 alors :
            neighbor_faces[iT,iF] = jF
        où jF est la face locale de jT correspondant à la même arête.

        Si la face est frontière :
            neighbor_faces[iT,iF] = -1

        Cette table est indispensable pour les flux DG
        (accès direct à la trace du voisin).

    edges_to_triangles : dict
        Structure duale du maillage :

            edge = (min(i,j), max(i,j))  →  [(iT,iF), (jT,jF)]

        Elle associe une arête globale aux triangles qui la portent.

        Cas possibles :
            len(list) == 2  → arête interne
            len(list) == 1  → arête frontière

        Cette structure est utilisée pour :
            - détecter les frontières
            - injecter les BC
            - intégrer une seule fois par face (méthodes DG)
            - debug topologique
    """

    triangles = np.asarray(mesh.cells_dict["triangle"])
    lines     = mesh.cells_dict["line"]
    tags      = mesh.cell_data_dict["gmsh:physical"]["line"]
    field     = mesh.field_data

    NT = len(triangles)

    neighbors = -np.ones((NT,3), dtype=int)
    neighbor_faces = -np.ones((NT,3), dtype=int)

    edges_to_triangles = {}

    # -------------------------------------------------
    # 1 Construire la connectivité pure (topologique)
    # -------------------------------------------------
    for iT, tri in enumerate(triangles):

        edges = [
            (tri[1], tri[2]),  # face 0
            (tri[2], tri[0]),  # face 1
            (tri[0], tri[1])   # face 2
        ]

        for iF, (i, j) in enumerate(edges):

            edge = tuple(sorted((int(i), int(j))))

            if edge not in edges_to_triangles:
                edges_to_triangles[edge] = []

            edges_to_triangles[edge].append((iT, iF))

    # -------------------------------------------------
    # 2 Remplir les voisins internes
    # -------------------------------------------------
    for edge, tri_list in edges_to_triangles.items():

        if len(tri_list) == 2:
            (t1, f1), (t2, f2) = tri_list

            neighbors[t1, f1] = t2
            neighbors[t2, f2] = t1

            neighbor_faces[t1, f1] = f2
            neighbor_faces[t2, f2] = f1

    # -------------------------------------------------
    # 3 Construire le mapping gmsh_id -> code interne
    # -------------------------------------------------
    gmsh_id_to_code = {}

    for name, code in name_to_code.items():
        gmsh_id = field[name][0]
        gmsh_id_to_code[gmsh_id] = code

    # -------------------------------------------------
    # 4 Associer chaque arête frontière à sa BC Gmsh
    # -------------------------------------------------
    bc_from_edge = {}

    for (i, j), tag in zip(lines, tags):
        edge = tuple(sorted((int(i), int(j))))
        bc_from_edge[edge] = gmsh_id_to_code[tag]

    # -------------------------------------------------
    # 5 Injecter les BC dans neighbors
    # -------------------------------------------------
    for edge, tri_list in edges_to_triangles.items():

        if len(tri_list) == 1:  # frontière
            (iT, iF) = tri_list[0]

            if edge not in bc_from_edge:
                raise ValueError(f"Frontière non classée par Gmsh : {edge}")

            neighbors[iT, iF] = bc_from_edge[edge]

    return neighbors, neighbor_faces, edges_to_triangles


def build_boundary_conditions(mesh):
    """
    Construit automatiquement :

        bc_from_edge[(i,j)] = code_interne
        reference_BC("NAME") -> code_interne
        bc_name(code)        -> "NAME"
    """

    if "line" not in mesh.cells_dict:
        raise ValueError("Le mesh ne contient pas d'arêtes (type 'line').")

    lines = mesh.cells_dict["line"]
    tags  = mesh.cell_data_dict["gmsh:physical"]["line"]

    # --- tag gmsh -> nom ---
    tag_to_name = {
        int(tag): name
        for name, (tag, dim) in mesh.field_data.items()
        if dim == 1
    }

    if not tag_to_name:
        raise ValueError("Aucune frontière physique détectée.")

    # --- nom -> code interne (-1,-2,...) ---
    name_to_code = {
        name: -(k + 2)
        for k, name in enumerate(sorted(tag_to_name.values()))
    }

    # --- code -> nom (inverse indispensable !) ---
    code_to_name = {v: k for k, v in name_to_code.items()}

    # --- construire bc_from_edge ---
    bc_from_edge = {}
    for (i, j), tag in zip(lines, tags):
        edge = tuple(sorted((int(i), int(j))))
        bc_from_edge[edge] = name_to_code[tag_to_name[int(tag)]]

    # --- helpers ---
    def reference_BC(name):
        return name_to_code[name.upper()]

    def bc_name(code):
        return code_to_name[int(code)]

    return bc_from_edge, reference_BC, bc_name


def build_neighborhood_structure_with_bc(mesh):
    """
    Construit la structure de voisinage du maillage et y injecte automatiquement
    les conditions aux limites définies dans Gmsh.

    Cette fonction constitue l'interface entre la description géométrique Gmsh
    (Physical Groups) et la structure topologique utilisée par le solveur EF/DG.

    Elle réalise successivement :
        1) Reconstruction de la connectivité pure du maillage triangulaire
           (indépendamment de Gmsh) :
               - identification des triangles voisins
               - construction de la table edges_to_triangles

        2) Lecture automatique des Physical Groups de dimension 1 (frontières)
           présents dans le mesh Gmsh via :
               mesh.field_data
               mesh.cell_data_dict["gmsh:physical"]

        3) Attribution AUTOMATIQUE d'un code interne négatif à chaque type
           de condition limite détecté :
               -1, -2, -3, ...
           (aucun dictionnaire utilisateur n'est nécessaire).

        4) Construction du dictionnaire bc_from_edge :
               bc_from_edge[(i,j)] = code_BC
           qui associe chaque arête frontière à sa condition limite.

        5) Injection de ces codes directement dans la table de voisinage :
               neighbors[iT,iF] = jT    si face interne
               neighbors[iT,iF] = -k    si face frontière de type k

    -------------------------------------------------------------------------
    Signification des objets retournés
    -------------------------------------------------------------------------

    neighbors : ndarray (NT,3)
        Structure principale utilisée par les méthodes EF/DG.

        neighbors[iT,iF] donne ce qu'il y a "en face" de la face iF du triangle iT :

            >= 0  : indice du triangle voisin (face interne)
            <  0  : code de condition limite (face frontière)

        Cette unique structure encode à la fois :
            - la topologie du maillage
            - la nature physique des frontières

    neighbor_faces : ndarray (NT,3)
        neighbor_faces[iT,iF] = jF est le numéro de la face dans le triangle voisin
        correspondant à la face iF de iT.

        Indispensable pour :
            - flux numériques DG
            - intégrales de saut / moyenne
            - orientation cohérente des normales

    edges_to_triangles : dict
        Dictionnaire dual donnant, pour chaque arête géométrique,
        les triangles qui la portent :

            (i,j) -> [(triangle, face_locale), ...]

        Utile pour :
            - assemblages par face
            - post-traitement
            - debug topologique

    reference_BC : function
        Fonction utilitaire permettant d'obtenir le code interne associé
        à un nom de frontière Gmsh :

            reference_BC("Neumann")  -> -2
            reference_BC("Fourier")  -> -1

        Permet d'écrire un code solveur lisible sans dépendre de Gmsh.

    bc_name : function
        Fonction inverse donnant le nom physique associé à un code interne :

            bc_name(-2) -> "NEUMANN"

        Utilisée pour affichage, visualisation, diagnostics.

    -------------------------------------------------------------------------
    Remarque conceptuelle
    -------------------------------------------------------------------------
    Après appel à cette fonction, le solveur ne dépend plus de Gmsh.
    Toute l'information nécessaire est contenue dans la structure discrète :

        (triangles, neighbors, neighbor_faces)

    ce qui correspond exactement à la description mathématique du squelette
    du maillage utilisé dans les formulations EF/DG.
    """
    triangles = mesh.cells_dict["triangle"]

    neighbors, neighbor_faces, edges_to_triangles = \
        build_neighborhood_structure(triangles)

    bc_from_edge, reference_BC, bc_name = \
        build_boundary_conditions(mesh)

    neighbors = inject_bc_into_neighbors(
        neighbors,
        edges_to_triangles,
        bc_from_edge
    )

    return neighbors, neighbor_faces, edges_to_triangles, reference_BC, bc_name


def plot_mesh_with_bc(mesh,secondes=2):
    """
    Trace le maillage en coloriant automatiquement les différentes
    conditions aux limites définies dans Gmsh.
    """

    # --- Construction voisinage + BC ---
    neighbors, neighbor_faces, edges_to_triangles, reference_BC, bc_name = build_neighborhood_structure_with_bc(mesh)

    triangles = mesh.cells_dict["triangle"]
    points = mesh.points[:, :2]

    print(f"Nombre d'éléments : {len(triangles)}")
    print(f"Nombre de sommets : {len(points)}")

    fig, ax = plt.subplots(figsize=(6, 6))

    # --------------------------------------------------
    # 1. Tracé du maillage (fond neutre)
    # --------------------------------------------------
    ax.triplot(points[:, 0], points[:, 1], triangles,
               color="0.8", linewidth=0.8)

    # --------------------------------------------------
    # 2. Détection automatique des BC présentes
    # --------------------------------------------------
    bc_codes = sorted({v for v in neighbors.flatten() if v < 0})

    # palette simple mais robuste
    cmap = plt.get_cmap("tab10")

    code_to_color = {
        code: cmap(i % 10)
        for i, code in enumerate(bc_codes)
    }

    drawn = set()

    # --------------------------------------------------
    # 3. Parcours des faces frontière
    # --------------------------------------------------
    for iT, tri in enumerate(triangles):

        for iF in range(3):

            code = neighbors[iT, iF]

            if code >= 0:
                continue  # face interne

            color = code_to_color[code]

            # sommets de la face opposée à iF
            if iF == 0:
                i1, i2 = tri[1], tri[2]
            elif iF == 1:
                i1, i2 = tri[2], tri[0]
            else:
                i1, i2 = tri[0], tri[1]

            P1 = points[i1]
            P2 = points[i2]

            label = None
            if code not in drawn:
                label = bc_name(code)
                drawn.add(code)

            ax.plot([P1[0], P2[0]],
                    [P1[1], P2[1]],
                    color=color,
                    linewidth=2.5,
                    label=label)

    # --------------------------------------------------
    # 4. Finition
    # --------------------------------------------------
    ax.set_aspect("equal")
    ax.set_title("Maillage avec conditions aux limites (auto)")
    ax.legend()
    if secondes > 0:
        plt.show(block=False)
        plt.pause(secondes)
        plt.close()
    else :
        plt.show()


def compute_element_sizes(mesh):
    """
    Calcule une taille caractéristique h_K pour chaque triangle.

    On utilise : h_K = sqrt(4*|K|/pi)
    (diamètre du disque équivalent, robuste pour SIPG)
    """

    points = mesh.points[:, :2]
    triangles = mesh.cells_dict["triangle"]

    NT = len(triangles)
    hK = np.zeros(NT)

    for iT, (i0, i1, i2) in enumerate(triangles):
        A0 = points[i0]
        A1 = points[i1]
        A2 = points[i2]

        area = triangle_area(A0, A1, A2)

        hK[iT] = np.sqrt(4 * area / np.pi)

    return hK

def compute_h_min(mesh):
    """
    Calcule h_min = min_K h_K, la plus petite taille d'élément du maillage.

    Utile pour le critère de stabilité explicite (CFL).
    """

    hK = compute_element_sizes(mesh)
    return hK.min()