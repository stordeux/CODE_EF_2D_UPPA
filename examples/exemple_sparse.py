import numpy as np
from mes_packages import *


def build_matrix():
    """
    Petite matrice COO d'exemple.
    """
    A = COOMatrix(4, 4, 30)
    A.ajout(1, 0, 1.0)
    A.ajout(0, 1, 1.0)
    A.ajout(2, 2, 1.0)
    A.ajout(3, 2, 1.5)
    A.ajout(3, 3, 1.0)
    A.ajout(2, 3, 1.0)
    A.ajout(1, 3, 1.0)
    A.ajout(3, 1, 5.0)
    A.ajout(0, 3, 2.0)
    A.ajout(0, 3, np.pi)   # doublon COO volontaire
    return A


def build_tridiag_matrix(complexe=False):
    """
    Matrice tridiagonale d'exemple pour les produits et la résolution.
    """
    A = COOMatrix(4, 4, 30)
    A.ajout(0, 0, 2)
    A.ajout(1, 1, 2)
    A.ajout(2, 2, 2)
    A.ajout(3, 3, 2)
    A.ajout(0, 1, -1)
    A.ajout(1, 0, -1)
    A.ajout(1, 2, -1)
    A.ajout(2, 1, -1)
    A.ajout(2, 3, -1)
    A.ajout(3, 2, -1)

    if complexe:
        A.ajout(3, 3, 1j)

    return A


# ============================================================
# Exemple 1 : addition
# ============================================================

B = build_matrix()
C = build_matrix()

A = B + C

print("=== Exemple : A = B + C ===")
print("B dense =")
print(B.to_dense())
print("\nC dense =")
print(C.to_dense())
print("\nA = B + C =")
print(A.to_dense())

# B et C ne sont pas modifiées
print("\nB après B + C =")
print(B.to_dense())
print("\nC après B + C =")
print(C.to_dense())


# ============================================================
# Exemple 2 : addition en place
# ============================================================

B = build_matrix()
C = build_matrix()

print("\n=== Exemple : B += C ===")
print("B avant =")
print(B.to_dense())

B += C

print("\nB après B += C =")
print(B.to_dense())


# ============================================================
# Exemple 3 : soustraction
# ============================================================

B = build_matrix()
C = build_matrix()

A = B - C

print("\n=== Exemple : A = B - C ===")
print(A.to_dense())


# ============================================================
# Exemple 4 : multiplication par un scalaire
# ============================================================

A = build_tridiag_matrix()

B = 100 * A

print("\n=== Exemple : B = 100 * A ===")
print("A dense =")
print(A.to_dense())
print("\nB dense =")
print(B.to_dense())


# ============================================================
# Exemple 5 : produit matrice-vecteur à droite
# ============================================================

A = build_tridiag_matrix(complexe=True)
u = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=complex)

Au = A @ u

print("\n=== Exemple : A @ u ===")
print("A dense =")
print(A.to_dense())
print("\nu =")
print(u)
print("\nA @ u =")
print(Au)


# ============================================================
# Exemple 6 : produit vecteur-matrice à gauche
# ============================================================

vT = u.T
vTA = A.produit_gauche(vT)

print("\n=== Exemple : v^T A ===")
print("v^T =")
print(vT)
print("\nv^T A =")
print(vTA)


# ============================================================
# Exemple 7 : forme sesquilinéaire v^* A u
# ============================================================

v = np.array([1, 2, 3, 4], dtype=complex)
u = np.array([1, 1, 1, 1], dtype=complex)

z = A.sesquilinear_form(v, u)

print("\n=== Exemple : v^* A u ===")
print("v =", v)
print("u =", u)
print("v^* A u =", z)


# ============================================================
# Exemple 8 : résolution linéaire
# ============================================================

A = build_tridiag_matrix()
F = np.array([[2.0], [4.0], [6.0], [7.0]])

print("\n=== Exemple : résolution A X = F ===")
print("A dense =")
print(A.to_dense())
print("\nF =")
print(F)

X = A.solve(F)

print("\nX =")
print(X)
print("\nA X =")
print(A.to_dense() @ X)


# ============================================================
# Exemple 9 : visualisation de la structure creuse
# ============================================================

print("\n=== Exemple : spy() ===")
A.spy()