# `assemble_volume`

La fonction `assemble_volume` assemble une matrice globale associée à une intégrale volumique de la forme

$$
A_{ij} = \int_{\Omega} f(x,y)\, O_v(\phi_i)\, O_u(\phi_j)\, dx
$$

où

- $f(x,y)$ est une fonction donnée,
- $O_u$ et $O_v$ sont des opérateurs appliqués aux fonctions de base,
- $\phi_i$ sont les fonctions de base locales.

La matrice est assemblée en parcourant tous les éléments du maillage triangulaire et en intégrant les contributions locales par quadrature.

---

## Signature

```python
assemble_volume(mesh, ordre, func, operatoru, operatorv, methode="CG")
```

Paramètres
mesh

Maillage triangulaire (meshio.Mesh).

Le maillage doit contenir

```python
mesh.points
mesh.cells_dict["triangle"]
```

ordre

Ordre polynomial des éléments finis.

Le nombre de degrés de liberté locaux est
$$N_{loc}=\frac{(ordre+1)(ordre+2)}{2}.$$

func

Fonction f(x,y) utilisée dans l'intégrale.

Elle doit avoir la forme :

```python
def func(x, y):
    return ...
```

et accepter des vecteurs numpy.

Exemple :

```python
func = lambda x, y: 1.0
```

### `operatoru`

Opérateur appliqué à la fonction de base associée à la variable $u$.

Valeurs possibles :

| opérateur | signification          |
| --------- | ---------------------- |
| `"u"`     | fonction de base $u$   |
| `"dxu"`   | dérivée $\partial_x u$ |
| `"dyu"`   | dérivée $\partial_y u$ |

---

### `operatorv`

Opérateur appliqué à la fonction test $v$.

Valeurs possibles :

| opérateur | signification          |
| --------- | ---------------------- |
| `"v"`     | fonction test $v$      |
| `"dxv"`   | dérivée $\partial_x v$ |
| `"dyv"`   | dérivée $\partial_y v$ |

---

### `methode`

Type de méthode d’éléments finis utilisée pour l’assemblage :

| valeur | description                                           |
| ------ | ----------------------------------------------------- |
| `"CG"` | éléments finis continus (_Continuous Galerkin_)       |
| `"DG"` | éléments finis discontinus (_Discontinuous Galerkin_) |

La connectivité locale-globale des degrés de liberté est adaptée automatiquement en fonction de la méthode choisie.

## Retour

La fonction retourne une matrice creuse de type `COOMatrix` de taille

$$
N_{glob} \times N_{glob}
$$

où $N_{glob}$ est le nombre total de degrés de liberté du problème.

---

## Principe de l’algorithme

Pour chaque triangle T du maillage :

1. calcul du jacobien affine de la transformation référence → élément,
2. évaluation des fonctions de base aux points de quadrature,
3. transformation des gradients vers les coordonnées physiques,
4. construction de la matrice locale

$$
M_{loc} =
|J| \sum_q w_q f(x_q,y_q) O_v(\phi_i(x_q,y_q)) O_u(\phi_j(x_q,y_q))
$$

5. insertion de la matrice locale dans la matrice globale.

L’assemblage global utilise la table

LoctoGlob

qui dépend du choix de la méthode :

- CG : degrés de liberté partagés entre éléments
- DG : degrés de liberté locaux à chaque élément

---

## Exemple

### Assemblage de la matrice de masse

func = lambda x, y: 1

M = assemble_volume(
mesh,
ordre=2,
func=func,
operatoru="u",
operatorv="v",
methode="CG"
)

### Assemblage de la matrice de rigidité

Kx = assemble_volume(mesh, ordre, lambda x, y: 1, "dxu", "dxv")
Ky = assemble_volume(mesh, ordre, lambda x, y: 1, "dyu", "dyv")

K = Kx + Ky

---

## Remarques

- La quadrature est calculée une seule fois sur le triangle de référence afin d’améliorer les performances.
- Les gradients des fonctions de base sont transformés vers les coordonnées physiques à l’aide du jacobien de l’élément.
- L’assemblage global est réalisé via la structure creuse COOMatrix.
