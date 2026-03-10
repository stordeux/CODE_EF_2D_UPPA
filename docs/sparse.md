# Sparse matrices (sparse.py)

Le module sparse.py fournit une implémentation simple de matrices creuses basée sur le format COO (Coordinate format).
Cette structure est particulièrement adaptée à l’assemblage de matrices issues de méthodes d’éléments finis.

## La classe principale est : COOMatrix

Elle permet :
- l’assemblage progressif de matrices creuses
- les opérations algébriques de base
- la résolution de systèmes linéaires
- des produits matrice–vecteur efficaces

## PRINCIPE DU FORMAT COO

Une matrice creuse est stockée sous forme de trois tableaux :

rows  : indices de lignes
cols  : indices de colonnes
data  : valeurs des coefficients

Chaque coefficient non nul est représenté par un triplet

(rows[k], cols[k], data[k])

Contrairement à d'autres formats, les doublons sont autorisés.

Par exemple :

(0,3,2.0)
(0,3,3.14)

Les deux contributions seront additionnées lors de la conversion dense.
Cette propriété est très utile pour l’assemblage EF.


## CRÉATION D'UNE MATRICE

Exemple Python :

    from mes_packages import *
    A = COOMatrix(nblig=4,nb_col=4,nnz=30)

Paramètres :

nb_lig : nombre de lignes
nb_col : nombre de colonnes
nnz    : capacité maximale de coefficients


## AJOUT DE COEFFICIENTS

Exemple :
    A.ajout(i,j,val)
    A.ajout(1,0,1.0)
    A.ajout(0,1,1.0)
    A.ajout(0,3,2.0)
    A.ajout(0,3,3.14)

Les doublons sont autorisés.


## CONVERSION EN MATRICE DENSE

    A_dense = A.to_dense()
    print(A_dense)

Les doublons sont automatiquement sommés.

------------------------------------------------------------

ADDITION DE MATRICES

Addition standard :

    A = B + C

Cette opération :
- crée une nouvelle matrice
- ne modifie pas B ni C

Addition en place :

    B += C

Cette opération modifie directement B.


## SOUSTRACTION

    A = B - C

Crée une nouvelle matrice correspondant à la différence.


## MULTIPLICATION PAR UN SCALAIRE

    B = 100 * A

Tous les coefficients sont multipliés.


# PRODUIT MATRICE-VECTEUR

    u = np.array([[1],[2],[3],[4]])
    Au = A @ u

# PRODUIT VECTEUR-MATRICE

    vT = u.T
    vTA = A.produit_gauche(vT)


# FORME SESQUILINÉAIRE

    z = A.sesquilinear_form(v,u)

calcule

v* A u

où v* est le conjugué hermitien.


# RÉSOLUTION DE SYSTÈMES LINÉAIRES

Pour résoudre

A x = b

on peut écrire

    x = A.solve(b)


# FACTORISATION LU

    A.lu()
    x = A.solveLU(b)

Cela permet de résoudre plusieurs systèmes avec la même matrice.

# VISUALISATION DE LA STRUCTURE CREUSE

    A.spy()

Affiche la position des coefficients non nuls.

Cette visualisation est utile pour :
- vérifier l’assemblage
- inspecter la structure des couplages
- analyser la sparsité d'une matrice


# EXEMPLE COMPLET

Un exemple d’utilisation se trouve dans :

examples/exemple_sparse.py

Cet exemple illustre :
- addition de matrices
- addition en place
- soustraction
- multiplication par un scalaire
- produit matrice-vecteur
- produit vecteur-matrice
- forme sesquilinéaire
- résolution de systèmes linéaires
- visualisation spy()
"""