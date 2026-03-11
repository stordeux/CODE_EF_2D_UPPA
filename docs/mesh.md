# Documentation des routines de création de maillages

## Objectif

Ce document présente les principales routines de génération et d’analyse de maillages 2D du module `mesh.py`, ainsi que trois exemples d’utilisation.

L’idée générale est la suivante :

* décrire une géométrie simple avec **Gmsh** ;
* générer un maillage triangulaire 2D ;
* conserver les **groupes physiques** pour identifier les frontières ;
* relire le maillage avec **meshio** ;
* corriger si besoin l’orientation des triangles ;
* exploiter ensuite le maillage dans les routines EF / DG.

---

## Convention sur les frontières physiques

Les maillages créés dans ce module utilisent des noms physiques Gmsh pour identifier les différentes parties du bord.

Les noms les plus utilisés sont :

* `FOURIER` : frontière extérieure du domaine ;
* `NEUMANN` : frontière intérieure ou obstacle ;
* `OMEGA` : domaine 2D lui-même.

Ces noms sont ensuite relus automatiquement et convertis en codes internes négatifs dans la structure de voisinage.

---

## 1. Vérification de l’orientation des triangles

### `verifier_et_corriger_orientation(mesh)`

Cette routine vérifie que tous les triangles sont orientés positivement (sens trigonométrique) et corrige les triangles mal orientés en échangeant deux sommets.

### Rôle

Une orientation cohérente est importante pour :

* le calcul des normales sortantes ;
* les intégrales de bord ;
* les formulations DG ;
* la cohérence des assemblages élémentaires.

### Principe

Pour chaque triangle `(A0, A1, A2)`, on calcule le déterminant :

```python
det = (A1[0] - A0[0]) * (A2[1] - A0[1]) - (A1[1] - A0[1]) * (A2[0] - A0[0])
```

* si `det > 0`, l’orientation est correcte ;
* si `det < 0`, le triangle est inversé ;
* si `det = 0`, le triangle est dégénéré.

---

## 2. Création d’un cercle dans un carré

### `create_mesh_circle_in_square(radius=0.2, square_size=1.0, mesh_size=0.05)`

Cette routine construit un domaine de type :

[
\Omega = \text{carré} \setminus \text{disque}
]

avec :

* bord extérieur du carré tagué `FOURIER` ;
* bord du trou circulaire tagué `NEUMANN` ;
* surface du domaine taguée `OMEGA`.

### Paramètres

* `radius` : rayon du trou circulaire ;
* `square_size` : longueur du côté du carré ;
* `mesh_size` : taille caractéristique du maillage Gmsh.

### Particularité importante

Le cercle est construit à l’aide de **quatre arcs de cercle** et non avec un unique `addCircle`, afin de bien conserver les informations physiques lors du passage par Gmsh puis MeshIO.

### Étapes de la routine

1. création des sommets et arêtes du carré ;
2. création des quatre arcs du cercle ;
3. construction de la surface perforée ;
4. définition des groupes physiques ;
5. génération du maillage ;
6. écriture temporaire du fichier `.msh` ;
7. relecture avec `meshio` ;
8. correction de l’orientation.

### Usage minimal

```python
mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
```

---

## 3. Création d’un polygone simple

### `create_mesh_from_polygon(points, mesh_size=0.05)`

Cette routine génère un maillage 2D à partir d’un polygone simple décrit par la liste ordonnée de ses sommets.

### Paramètres

* `points` : liste des sommets `[(x1,y1), (x2,y2), ...]` ordonnés le long du bord ;
* `mesh_size` : taille caractéristique du maillage.

### Groupes physiques

* toute la frontière extérieure reçoit le tag `FOURIER` ;
* le domaine reçoit le tag `OMEGA`.

### Principe

La fonction :

* crée les points Gmsh ;
* crée automatiquement les segments entre points consécutifs ;
* ferme le polygone ;
* génère la surface ;
* attribue les groupes physiques ;
* génère et relit le maillage.

### Exemple

```python
points = [
    (-4,0), (0,0), (0,1.5), (2,1.5), (2,0), (6,0),
    (6,4), (2,4), (2,2.5), (0,2.5), (0,4), (-4,4)
]
mesh = create_mesh_from_polygon(points, mesh_size=0.2)
```

Cette routine est particulièrement pratique pour les tests unitaires sur des géométries polygonales.

---

## 4. Création d’un polygone avec trou polygonal

### `create_mesh_polygon_with_hole(outer_points, inner_points, mesh_size=0.05)`

Cette routine construit un domaine polygonal perforé :

[
\Omega = \text{polygone extérieur} \setminus \text{polygone intérieur}
]

### Paramètres

* `outer_points` : sommets du polygone extérieur ;
* `inner_points` : sommets du polygone intérieur ;
* `mesh_size` : taille caractéristique du maillage.

### Groupes physiques

* frontières du polygone extérieur : `FOURIER` ;
* frontières du polygone intérieur : `NEUMANN` ;
* domaine : `OMEGA`.

### Usage typique

Cette routine permet de créer des obstacles polygonaux non circulaires, par exemple :

* carré avec trou carré ;
* hexagone avec étoile intérieure ;
* polygone concave avec inclusion polygonale.

### Exemple d’idée géométrique

```python
outer = [(-1,-1), (1,-1), (1,1), (-1,1)]
inner = [(-0.3,-0.3), (0.3,-0.3), (0.3,0.3), (-0.3,0.3)]
mesh = create_mesh_polygon_with_hole(outer, inner, mesh_size=0.05)
```

---

## 5. Visualisation des maillages

### `plot_mesh(mesh, secondes=0)`

Affiche le maillage triangulaire avec `matplotlib`.

### Comportement

* si `secondes is 0`, la fenêtre reste ouverte ;
* sinon, elle se ferme automatiquement après le nombre de secondes demandé.

### Exemple

```python
plot_mesh(mesh, secondes=2)
```

---

### `plot_mesh_with_bc(mesh, secondes=2)`

Affiche le maillage en coloriant les arêtes de frontière selon les conditions aux limites détectées automatiquement dans les groupes physiques Gmsh.

### Fonctionnement

La routine :

1. construit la structure de voisinage enrichie avec les BC ;
2. détecte les codes de BC présents ;
3. attribue une couleur à chaque type de frontière ;
4. trace uniquement les faces de bord avec leur étiquette.

### Intérêt

Cette routine est très utile pour vérifier visuellement que :

* les tags `FOURIER` et `NEUMANN` sont correctement posés ;
* la lecture Gmsh / MeshIO est correcte ;
* la structure de voisinage est cohérente.

### Exemple

```python
plot_mesh_with_bc(mesh, secondes=2)
```

---

## 6. Construction de la structure de voisinage

### `build_neighborhood_structure(triangles)`

Construit la connectivité topologique du maillage triangulaire.

### Sorties

* `neighbors[iT, iF]` : triangle voisin de la face `iF` du triangle `iT` ;
* `neighbor_faces[iT, iF]` : numéro local de la face correspondante dans le voisin ;
* `edges_to_triangles` : dictionnaire associant chaque arête globale aux triangles adjacents.

### Convention locale des faces

Pour un triangle `(v0, v1, v2)` :

* face 0 = arête opposée à `v0`, donc `(v1, v2)` ;
* face 1 = arête opposée à `v1`, donc `(v2, v0)` ;
* face 2 = arête opposée à `v2`, donc `(v0, v1)`.

Cette convention est fondamentale pour les normales et les flux DG.

---

## 7. Gestion automatique des conditions aux limites

### `build_boundary_conditions(mesh)`

Construit automatiquement :

* `bc_from_edge[(i,j)] = code_interne` ;
* `reference_BC(name)` : nom → code ;
* `bc_name(code)` : code → nom.

Les groupes physiques Gmsh de dimension 1 sont lus via :

* `mesh.field_data` ;
* `mesh.cell_data_dict["gmsh:physical"]`.

Les noms de frontière sont ensuite transformés en codes internes négatifs :

* `-2`, `-3`, `-4`, etc.

---

### `inject_bc_into_neighbors(neighbors, edges_to_triangles, bc_from_edge)`

Injecte les codes de conditions aux limites directement dans la table de voisinage.

Ainsi, une face frontière n’est plus marquée seulement par `-1`, mais par un code identifiant son type physique.

---

### `build_neighborhood_structure_with_bc(mesh)`

C’est la routine centrale pour relier la géométrie Gmsh au solveur EF / DG.

Elle :

1. construit la connectivité topologique ;
2. lit les groupes physiques ;
3. attribue des codes internes ;
4. injecte ces codes dans `neighbors`.

### Résultat

Après cette étape :

* `neighbors[iT, iF] >= 0` signifie « face interne » ;
* `neighbors[iT, iF] < 0` signifie « face frontière de type donné ».

Le solveur n’a alors plus besoin d’interroger Gmsh : toute l’information utile est contenue dans la structure discrète.

---

## 8. Outils géométriques complémentaires

### `triangle_area(p1, p2, p3)`

Calcule l’aire d’un triangle 2D.

### `check_triangle_areas(nodes, triangles, tol=1e-12)`

Vérifie les aires des triangles et signale les éléments dégénérés ou presque dégénérés.

### `calcul_normale(A0, A1, A2, i)`

Calcule la normale sortante à la face opposée au sommet `i`.

### `plot_triangle_with_normals(...)`

Visualise un triangle et ses normales sortantes.

### `plot_structure_voisinage(mesh, ielt_test=None, ax=None)`

Affiche un triangle, ses voisins et les normales de part et d’autre des faces communes.

Ces routines sont particulièrement utiles pour le débogage des formulations DG.

---

## 9. Localisation spatiale

### `build_spatial_grid(mesh, nx=None, ny=None)`

Construit une grille spatiale uniforme pour accélérer la recherche de triangles candidats.

### `candidate_triangles(x, y, spgrid)`

Retourne la liste des triangles candidats pour un point donné.

Ces routines sont utiles lorsqu’on veut localiser rapidement un point dans un maillage, par exemple pour de l’interpolation ou du post-traitement.

---

## 10. Tailles caractéristiques des éléments

### `compute_element_sizes(mesh)`

Calcule une taille caractéristique `h_K` pour chaque triangle, définie ici par

[
h_K = \sqrt{\frac{4|K|}{\pi}}
]

c’est-à-dire le diamètre du disque de même aire.

### `compute_h_min(mesh)`

Retourne

$$
h_{\min} = \min_K h_K
$$

utile pour les critères CFL ou les pénalisations de type SIPG.

---

## 11. Exemples fournis

### Exemple 1 — cercle dans un carré

Le script `exemple_mesh_1.py` crée un carré percé d’un trou circulaire, corrige l’orientation puis affiche :

* le maillage ;
* le maillage colorié par conditions aux limites.

Code principal :

```python
mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
nb_corriges = verifier_et_corriger_orientation(mesh)
plot_mesh(mesh, secondes=2)
plot_mesh_with_bc(mesh, secondes=2)
```

---

### Exemple 2 — polygone simple

Le script `exemple_mesh_2.py` définit explicitement une liste de sommets polygonaux, puis génère le maillage correspondant.

Code principal :

```python
mesh = create_mesh_from_polygon(points, mesh_size=0.2)
plot_mesh(mesh, secondes=2)
plot_mesh_with_bc(mesh, secondes=2)
```

---

### Exemple 3 — étoile dans un hexagone régulier

Le script `exemple_mesh_3.py` construit :

* un hexagone régulier extérieur ;
* une étoile intérieure ;
* un maillage polygonal avec trou.

Code principal :

```python
outer = regular_hexagon(1.0)
inner = star_polygon()
mesh = create_mesh_polygon_with_hole(outer, inner, mesh_size=0.05)
plot_mesh(mesh, secondes=2)
plot_mesh_with_bc(mesh, secondes=0)
```

Cet exemple est intéressant pour tester :

* les géométries non convexes ;
* les coins aigus ;
* la visualisation des BC ;
* les futurs assemblages DG sur des obstacles polygonaux.

---

## 12. Résumé des principales routines de création

Les routines de création de maillage actuellement disponibles sont donc :

* `create_mesh_circle_in_square(...)`
* `create_mesh_from_polygon(...)`
* `create_mesh_polygon_with_hole(...)`

Elles couvrent déjà trois cas importants :

1. domaine perforé à trou circulaire ;
2. domaine polygonal simple ;
3. domaine polygonal avec obstacle polygonal.

---

## 13. Remarque de conception

L’organisation actuelle sépare assez bien trois niveaux :

1. **création géométrique** du maillage ;
2. **lecture / enrichissement topologique** avec les BC ;
3. **visualisation / diagnostic**.

C’est une bonne base pour intégrer ensuite les routines d’assemblage EF, CG, DG ou SIPDG.

---

## 14. Idées d’extensions naturelles

Parmi les extensions possibles :

* une routine `create_mesh_polygon_with_holes(...)` pour plusieurs trous ;
* un contrôle automatique du sens d’orientation des polygones d’entrée ;
* une option pour raffiner localement près des obstacles ;
* des maillages courbes d’ordre supérieur ;
* l’ajout de tags physiques plus variés (`DIRICHLET`, `ROBIN`, etc.).

---

## Conclusion

Le fichier `mesh.py` fournit déjà un petit noyau très utile pour :

* créer rapidement des géométries de test ;
* conserver proprement les conditions aux limites ;
* analyser la connectivité du maillage ;
* visualiser les structures nécessaires aux formulations EF / DG.

Ces routines sont particulièrement adaptées à la construction d’exemples et de tests unitaires pour un code de méthodes aux éléments finis.
