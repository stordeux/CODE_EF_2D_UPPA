# Créer un maillage 2D avec Gmsh : polygone avec trou

Ce document explique comment générer un maillage 2D à l’aide de **Gmsh** dans le code d’éléments finis, à partir de la fonction :

```python
create_mesh_polygon_with_hole(outer_points, inner_points, mesh_size=0.05)
```

Cette fonction construit un domaine polygonal plan comportant un **trou polygonal**, génère le maillage, associe des **groupes physiques** utiles pour les conditions aux limites, puis renvoie un objet `meshio.Mesh` compatible avec le reste du code.

---

## 1. Objectif

On souhaite mailler un domaine du type

\[
\Omega = \text{polygone extérieur} \setminus \text{polygone intérieur}.
\]

Autrement dit :
- le **contour extérieur** définit la frontière externe du domaine ;
- le **contour intérieur** définit un trou ;
- l’intérieur du domaine est maillé en triangles.

La fonction crée en plus les groupes physiques suivants :

- **`FOURIER`** : frontière extérieure ;
- **`NEUMANN`** : frontière du trou ;
- **`OMEGA`** : domaine 2D.

Ces noms sont importants, car ils peuvent ensuite être utilisés dans les routines d’assemblage ou pour sélectionner des parties du bord selon les conditions aux limites.

---

## 2. Signature de la fonction

```python
def create_mesh_polygon_with_hole(outer_points, inner_points, mesh_size=0.05):
```

### Paramètres

- `outer_points : list[(x, y)]`  
  Liste des sommets du polygone extérieur, dans l’ordre du contour.

- `inner_points : list[(x, y)]`  
  Liste des sommets du polygone intérieur (le trou), également dans l’ordre du contour.

- `mesh_size : float`  
  Taille caractéristique du maillage transmise à Gmsh. Plus cette valeur est petite, plus le maillage est fin.

### Retour

- `mesh : meshio.Mesh`  
  Maillage lu via `meshio`, prêt à être exploité dans le code.

---

## 3. Principe de construction

La fonction suit les étapes suivantes.

### 3.1. Initialisation de Gmsh

```python
gmsh.initialize()
gmsh.model.add("polygon_with_hole")
geo = gmsh.model.geo
```

On initialise Gmsh, on crée un modèle géométrique, puis on travaille avec le noyau géométrique `geo`.

---

### 3.2. Création du polygone extérieur

```python
outer_gmsh_pts = [
    geo.addPoint(x, y, 0, mesh_size)
    for x, y in outer_points
]
```

Chaque sommet du contour extérieur est créé comme point Gmsh.

Ensuite, on relie les sommets successifs par des segments :

```python
outer_lines = []
n = len(outer_gmsh_pts)

for i in range(n):
    p1 = outer_gmsh_pts[i]
    p2 = outer_gmsh_pts[(i + 1) % n]
    outer_lines.append(geo.addLine(p1, p2))
```

L’écriture `(i + 1) % n` permet de refermer automatiquement le polygone en reliant le dernier sommet au premier.

Enfin, on crée la boucle fermée associée :

```python
outer_loop = geo.addCurveLoop(outer_lines)
```

---

### 3.3. Création du trou polygonal

Le même procédé est appliqué au contour intérieur :

```python
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
```

Le contour intérieur délimite le trou.

---

### 3.4. Création de la surface perforée

```python
surface = geo.addPlaneSurface([outer_loop, inner_loop])
```

Cette instruction demande à Gmsh de créer une surface plane définie par :
- une boucle extérieure ;
- une boucle intérieure retirée de la surface.

On obtient donc un domaine 2D perforé.

Après cela, on synchronise la géométrie :

```python
geo.synchronize()
```

Cette étape est indispensable avant d’ajouter des groupes physiques ou de générer le maillage.

---

### 3.5. Définition des groupes physiques

```python
tag_fourier = gmsh.model.addPhysicalGroup(1, outer_lines)
gmsh.model.setPhysicalName(1, tag_fourier, "FOURIER")

tag_neumann = gmsh.model.addPhysicalGroup(1, inner_lines)
gmsh.model.setPhysicalName(1, tag_neumann, "NEUMANN")

tag_domain = gmsh.model.addPhysicalGroup(2, [surface])
gmsh.model.setPhysicalName(2, tag_domain, "OMEGA")
```

Ici :
- la dimension `1` correspond aux entités de bord (segments) ;
- la dimension `2` correspond à la surface.

Ces groupes physiques permettent ensuite d’identifier facilement :
- le bord extérieur ;
- le bord du trou ;
- le domaine intérieur.

---

### 3.6. Génération du maillage

```python
gmsh.model.mesh.generate(2)
```

Le paramètre `2` indique que l’on génère un maillage surfacique 2D.

---

### 3.7. Écriture, lecture et nettoyage

La fonction écrit temporairement le maillage dans un fichier `.msh`, puis le relit avec `meshio` :

```python
tmp = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
tmp.close()

gmsh.write(tmp.name)
gmsh.finalize()

mesh = meshio.read(tmp.name)
os.remove(tmp.name)
```

Cela permet de récupérer un objet `meshio.Mesh` sans conserver de fichier intermédiaire sur le disque.

---

### 3.8. Correction éventuelle de l’orientation

```python
verifier_et_corriger_orientation(mesh)
```

Cette étape est importante pour garantir que les triangles sont orientés de manière cohérente dans le reste du code.

---

## 4. Fonction complète

```python
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
```

---

## 5. Exemple d’utilisation

Voici un exemple simple : un carré contenant un trou carré.

```python
outer_points = [
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
]

inner_points = [
    (0.4, 0.4),
    (0.6, 0.4),
    (0.6, 0.6),
    (0.4, 0.6),
]

mesh = create_mesh_polygon_with_hole(
    outer_points=outer_points,
    inner_points=inner_points,
    mesh_size=0.05,
)
```

Dans cet exemple :
- `outer_points` décrit le contour extérieur du domaine ;
- `inner_points` décrit le trou ;
- `mesh_size=0.05` fixe une finesse de maillage modérée.

---

## 6. Recommandations pratiques

### Ordre des sommets

Les sommets doivent être fournis dans l’ordre du contour, sans croisement d’arêtes.

En pratique, il est préférable que :
- le contour extérieur soit donné dans un ordre cohérent ;
- le contour intérieur soit également donné dans l’ordre du contour.

Même si Gmsh accepte souvent plusieurs conventions, une orientation incohérente ou un polygone auto-intersectant peut provoquer des erreurs géométriques.

### Taille de maillage

- une valeur de `mesh_size` plus petite produit un maillage plus fin ;
- une valeur trop petite augmente rapidement le nombre de triangles et donc le coût des calculs.

### Groupes physiques

Les noms `FOURIER`, `NEUMANN` et `OMEGA` doivent rester cohérents avec les conventions du reste du code si vous voulez réutiliser directement les routines d’assemblage existantes.

---

## 7. Dépendances

Cette fonction nécessite au minimum :

```bash
pip install gmsh meshio
```

Selon votre environnement, il peut aussi être utile d’avoir :

```bash
pip install numpy matplotlib
```

pour visualiser ou exploiter ensuite le maillage.

---

## 8. Résumé

La fonction `create_mesh_polygon_with_hole` permet de :

1. définir un contour extérieur polygonal ;
2. définir un contour intérieur polygonal jouant le rôle de trou ;
3. construire automatiquement la surface perforée dans Gmsh ;
4. associer des groupes physiques adaptés au code d’éléments finis ;
5. générer un maillage triangulaire 2D ;
6. renvoyer ce maillage au format `meshio.Mesh`.

C’est une base simple et robuste pour construire des géométries polygonales avec obstacle intérieur dans le cadre des simulations par éléments finis.
