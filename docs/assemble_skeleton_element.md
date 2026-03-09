# `assemble_skeleton_par_element`

La fonction `assemble_skeleton_par_element` assemble une matrice globale associée à une **intégrale sur les faces internes du maillage** (squelette) avec une sommation sur les éléments.

La forme générale assemblée est

$$
A_{ij} = \sum_{T\in\mathcal{T}}\sum_{F\in\mathcal{F}_T\cap\mathcal{F}_{int}}
\int_{F}
c(x,y)\,
O_v(v_i)\,
O_u(u_j)\,
ds
$$

où

- $F$ désigne une **face interne du maillage**,
- $c(x,y)$ est un coefficient scalaire,
- $O_u$ et $O_v$ sont des opérateurs appliqués aux traces des fonctions de base,
- $u_j$ et $v_i$ sont les fonctions de base associées aux degrés de liberté.

Cette routine est utilisée dans les méthodes **Discontinuous Galerkin (DG)** pour construire les termes de squelette.

---

## Signature

```python
assemble_skeleton_par_element(
    mesh,
    ordre,
    coef,
    operatoru="uT",
    operatorv="vT",
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

- une constante
- une fonction

```python
coef(x,y)
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

| opérateur   | signification                  |
| ----------- | ------------------------------ |
| `"uT"`      | trace côté élément courant     |
| `"uV"`      | trace côté élément voisin      |
| `"dxuT"`    | dérivée $\partial_x u$ côté T  |
| `"dyuT"`    | dérivée $\partial_y u$ côté T  |
| `"dnuT"`    | dérivée normale                |
| `"dtuT"`    | dérivée tangentielle           |
| `"graduT"`  | gradient complet               |
| `"uTnT"`    | flux $u n$                     |
| `"uTnx"`    | $u n_x$                        |
| `"uTny"`    | $u n_y$                        |
| `"(M.n)uT"` | produit scalaire $(M\cdot n)u$ |

Les opérateurs analogues existent pour le côté voisin (`uV`, `dxuV`, etc.).

---

### `operatorv`

Opérateur appliqué à la fonction test $v$.

Les mêmes opérateurs sont disponibles :

| opérateur  | signification          |
| ---------- | ---------------------- |
| `"vT"`     | trace côté T           |
| `"vV"`     | trace côté voisin      |
| `"dxvT"`   | dérivée $\partial_x v$ |
| `"dyvT"`   | dérivée $\partial_y v$ |
| `"dnvT"`   | dérivée normale        |
| `"dtvT"`   | dérivée tangentielle   |
| `"gradvT"` | gradient complet       |
| `"vTnT"`   | flux $v n$             |

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

où

- $N_{glob}$ est le nombre total de degrés de liberté.

---

## Principe de l’algorithme

Pour chaque **face interne** du maillage :

1. identification des deux éléments adjacents

$$
T \quad \text{et} \quad V
$$

2. calcul de la **normale extérieure**

$$
n_T
$$

3. construction de la **tangente**

$$
t = (-n_y,n_x)
$$

4. évaluation des fonctions de base sur la face

5. transformation des gradients vers les coordonnées physiques

6. construction des **quatre blocs locaux**

$$
M_{TT},\quad
M_{TV},\quad
M_{VT},\quad
M_{VV}
$$

7. insertion dans la matrice globale.

Chaque face interne contribue donc à **quatre blocs élémentaires**.

---

## Exemple

### Terme de pénalisation SIPDG

```python
coef = lambda x,y: 10

A = assemble_skeleton_par_element(
    mesh,
    ordre,
    coef,
    operatoru="uT",
    operatorv="vT",
    methode="DG"
)
```

---

## Remarques

- Les faces internes sont traitées **deux fois**.
- L’orientation des points de quadrature entre éléments voisins est automatiquement corrigée.
- Les opérateurs vectoriels sont contractés via un **produit scalaire**.
- L’assemblage global est réalisé via la structure creuse `COOMatrix`.
