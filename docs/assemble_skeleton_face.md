# `assemble_skeleton_par_face`

La fonction `assemble_skeleton_par_face` assemble une matrice globale associée à une **intégrale sur les faces internes du maillage** (squelette), avec une **sommation sur les faces internes**.

La forme générale assemblée est

$$
A_{ij} = \sum_{F\in \mathcal{F}_{int}}
\int_{F}
c(x,y)\,
O_v(v_i)\,
O_u(u_j)\,
ds
$$

où

- $\mathcal{F}_{int}$ désigne l’ensemble des **faces internes du maillage**,
- $c(x,y)$ est un coefficient scalaire,
- $O_u$ et $O_v$ sont des opérateurs appliqués aux traces des fonctions de base,
- $u_j$ et $v_i$ sont les fonctions de base associées aux degrés de liberté.

Cette routine est utilisée dans les méthodes **Discontinuous Galerkin (DG)** pour construire les termes de squelette.

---

## Signature

```python
assemble_skeleton_par_face(
    mesh,
    ordre,
    coef,
    operatoru="sautu",
    operatorv="sautv",
    methode="DG"
)
```

---

## Paramètres

### `mesh`

Maillage triangulaire (`meshio.Mesh`).

Le maillage doit contenir

```python
mesh.points
mesh.cells_dict["triangle"]
```

---

### `ordre`

Ordre polynomial des éléments finis.

Le nombre de degrés de liberté locaux est

$$
N_{loc}=\frac{(ordre+1)(ordre+2)}{2}.
$$

---

### `coef`

Coefficient apparaissant dans l’intégrale de squelette.

Il peut être :

- une constante,
- une fonction

```python
coef(x, y)
```

acceptant des vecteurs numpy.

Exemple :

```python
coef = 1.0
```

ou

```python
coef = lambda x, y: 1 + x
```

---

### `operatoru`

Opérateur appliqué à la trace de la fonction inconnue $u$.

Valeurs possibles (exemples principaux) :

| opérateur     | signification                   |
| ------------- | ------------------------------- |
| `"uT"`        | trace côté élément courant      |
| `"uV"`        | trace côté élément voisin       |
| `"sautu"`     | saut $u_T-u_V$                  |
| `"dxuT"`      | dérivée $\partial_x u$ côté $T$ |
| `"dyuT"`      | dérivée $\partial_y u$ côté $T$ |
| `"dnuT"`      | dérivée normale côté $T$        |
| `"dtuT"`      | dérivée tangentielle côté $T$   |
| `"moyu"`      | moyenne des traces              |
| `"moynablau"` | moyenne des gradients           |
| `"uTnT"`      | flux $u_T n_T$                  |
| `"sautDGu"`   | saut de flux                    |

Les opérateurs analogues existent pour le côté voisin (`uV`, `dxuV`, etc.).

---

### `operatorv`

Opérateur appliqué à la fonction test $v$.

Les mêmes opérateurs sont disponibles :

| opérateur     | signification          |
| ------------- | ---------------------- |
| `"vT"`        | trace côté $T$         |
| `"vV"`        | trace côté voisin      |
| `"sautv"`     | saut $v_T-v_V$         |
| `"dxvT"`      | dérivée $\partial_x v$ |
| `"dyvT"`      | dérivée $\partial_y v$ |
| `"dnvT"`      | dérivée normale        |
| `"dtvT"`      | dérivée tangentielle   |
| `"moyv"`      | moyenne des traces     |
| `"moynablav"` | moyenne des gradients  |
| `"vTnT"`      | flux $v_T n_T$         |
| `"sautDGv"`   | saut de flux           |

---

### `methode`

Type de méthode d’éléments finis :

| valeur | description                |
| ------ | -------------------------- |
| `"DG"` | éléments finis discontinus |
| `"CG"` | éléments finis continus    |

La connectivité locale-globale est adaptée automatiquement.

---

## Retour

La fonction retourne une matrice creuse

```
COOMatrix
```

de taille

$$
N_{glob} \times N_{glob}
$$

où $N_{glob}$ est le nombre total de degrés de liberté.

---

## Principe de l’algorithme

Pour chaque **face interne** du maillage :

1. identification des deux éléments adjacents

$$
T \quad \text{et} \quad V
$$

2. calcul de la **normale extérieure** $n_T$ du côté de l’élément $T$,

3. construction de la **tangente**

$$
t = (-n_y,n_x),
$$

4. évaluation des fonctions de base sur la face, des deux côtés de l’interface,

5. transformation des gradients vers les coordonnées physiques,

6. construction des **quatre blocs locaux**

$$
M_{TT},\quad
M_{TV},\quad
M_{VT},\quad
M_{VV},
$$

7. insertion dans la matrice globale.

Chaque face interne contribue donc à **quatre blocs élémentaires**, mais **elle n’est traitée qu’une seule fois** dans `assemble_skeleton_par_face`.

---

## Exemple

### Terme de pénalisation SIPDG

```python
coef = lambda x, y: 10

A = assemble_skeleton_par_face(
    mesh,
    ordre,
    coef,
    operatoru="sautu",
    operatorv="sautv",
    methode="DG"
)
```

---

## Remarques

- **Les faces internes sont traitées une seule fois** dans `assemble_skeleton_par_face`.
- L’orientation des points de quadrature entre éléments voisins est automatiquement corrigée.
- Les opérateurs vectoriels sont contractés via un **produit scalaire**.
- L’assemblage global est réalisé via la structure creuse `COOMatrix`.
