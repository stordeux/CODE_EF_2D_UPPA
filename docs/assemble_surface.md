# assemble_surface

La fonction `assemble_surface` assemble une matrice globale associée à une intégrale de bord de la forme

$$
A_{ij}=\int_{\Gamma} f(x,y)\, O_v(\phi_i)\, O_u(\phi_j)\, ds
$$

où

- $f(x,y)$ est une fonction donnée,
- $O_u$ et $O_v$ sont des opérateurs appliqués aux fonctions de base,
- $\phi_i$ sont les fonctions de base locales,
- $\Gamma$ désigne tout ou partie du bord du domaine.

La matrice est assemblée en parcourant les faces de bord du maillage triangulaire et en intégrant les contributions locales par quadrature sur les arêtes.

---

## Signature

assemble_surface(mesh, ordre, func, operatoru, operatorv, methode="CG", domaine="all")

---

## Paramètres

### mesh

Maillage triangulaire (`meshio.Mesh`).

Le maillage doit contenir :

mesh.points  
mesh.cells_dict["triangle"]

---

### ordre

Ordre polynomial des éléments finis.

Le nombre de degrés de liberté locaux est

$$
N_{loc}=\frac{(ordre+1)(ordre+2)}{2}.
$$

---

### func

Fonction $f(x,y)$ utilisée dans l'intégrale de bord.

Elle doit avoir la forme

def func(x, y):
return ...

et accepter des vecteurs numpy.

Exemple :

func = lambda x, y: 1.0

---

### operatoru

Opérateur appliqué à la fonction de base associée à la variable $u$.

Valeurs possibles :

- "u" : fonction de base $u$
- "dxu" : dérivée $\partial_x u$
- "dyu" : dérivée $\partial_y u$
- "dnu" : dérivée normale $\partial_n u$

---

### operatorv

Opérateur appliqué à la fonction test $v$.

Valeurs possibles :

- "v" : fonction test $v$
- "dxv" : dérivée $\partial_x v$
- "dyv" : dérivée $\partial_y v$
- "dnv" : dérivée normale $\partial_n v$

---

### methode

Type de méthode d’éléments finis utilisée pour l’assemblage :

- "CG" : éléments finis continus (Continuous Galerkin)
- "DG" : éléments finis discontinus (Discontinuous Galerkin)

La connectivité locale-globale des degrés de liberté est adaptée automatiquement en fonction de la méthode choisie.

---

### domaine

Partie du bord sur laquelle l’intégrale est assemblée.

Valeurs typiques :

- "all" : tout le bord
- "FOURIER"
- "DIRICHLET"
- "NEUMANN"

---

## Retour

La fonction retourne une matrice creuse de type `COOMatrix` de taille

$$
N_{glob}\times N_{glob}
$$

où $N_{glob}$ est le nombre total de degrés de liberté du problème.

---

## Principe de l’algorithme

Pour chaque triangle $T$ du maillage, et pour chaque face $F$ appartenant au domaine de bord choisi :

1. calcul de la géométrie de la face
2. calcul de la longueur de l’arête
3. évaluation des fonctions de base sur la face
4. transformation éventuelle des gradients vers les coordonnées physiques
5. construction de la matrice locale

$$
M_{loc}
=
|e|
\sum_q
w_q\,f(x_q,y_q)\,
O_v(\phi_i(x_q,y_q))\,
O_u(\phi_j(x_q,y_q))
$$

6. insertion de cette contribution dans la matrice globale.

L’assemblage global utilise la table

LoctoGlob

qui dépend du choix de la méthode :

- CG : degrés de liberté partagés entre éléments
- DG : degrés de liberté locaux à chaque élément

---

## Exemple

Assemblage d’une matrice de masse de bord :

func = lambda x, y: 1.0

Mbord = assemble_surface(
mesh,
ordre=2,
func=func,
operatoru="u",
operatorv="v",
methode="CG",
domaine="all"
)

---

## Remarques

- La quadrature sur les faces est calculée une seule fois sur la face de référence.
- Les dérivées normales sont obtenues à partir de la normale extérieure à la face.
- L’assemblage global est réalisé via la structure creuse `COOMatrix`.

