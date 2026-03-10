import numpy as np
import matplotlib.pyplot as plt
import builtins

_print = builtins.print

def print(*args, **kwargs):
    def colorize(arg):
        if isinstance(arg, (bool, np.bool_)):
            if arg:  # True
                return f'\033[94m{arg}\033[0m'  # Bleu
            else:    # False
                return f'\033[91m{arg}\033[0m'  # Rouge
        return arg
    
    args = [colorize(arg) for arg in args]
    _print(*args, **kwargs)

builtins.print = print

from mes_packages import *


# Construction de deux matrices creuses identiques pour tester l'addition
Mat=COOMatrix(4,4,30)
Mat.ajout(1,0,1)
Mat.ajout(0,1,1.0)
Mat.ajout(2,2,1.0)
Mat.ajout(3,2,1.5) 
Mat.ajout(3,3,1.0)
Mat.ajout(2,3,1.0)
Mat.ajout(1,3,1.0)
Mat.ajout(3,1,5.0)
Mat.ajout(0,3,2.0)
Mat.ajout(0,3,np.pi)
Mat1dense = Mat.to_dense() 

Mat2=COOMatrix(4,4,30)
Mat2.ajout(1,0,1)
Mat2.ajout(0,1,1.0)
Mat2.ajout(2,2,1.0)
Mat2.ajout(3,2,1.5) 
Mat2.ajout(3,3,1.0)
Mat2.ajout(2,3,1.0)
Mat2.ajout(1,3,1.0)
Mat2.ajout(3,1,5.0)
Mat2.ajout(0,3,2.0)
Mat2.ajout(0,3,np.pi)
Mat2dense = Mat2.to_dense()

# Verification de l'addition en comparant la version dense
print('Verification du +')
_=Mat+Mat2 
Mat3dense = Mat.to_dense()
TEST= np.allclose(Mat3dense, Mat1dense + Mat2dense)
print('Somme correcte ?', TEST)

# Verification de la soustraction
Mat=COOMatrix(4,4,30)
Mat.ajout(1,0,1)
Mat.ajout(0,1,1.0)
Mat.ajout(2,2,1.0)
Mat.ajout(3,2,1.5) 
Mat.ajout(3,3,1.0)
Mat.ajout(2,3,1.0)
Mat.ajout(1,3,1.0)
Mat.ajout(3,1,5.0)
Mat.ajout(0,3,2.0)
Mat1dense = Mat.to_dense() 

Mat2=COOMatrix(4,4,30)
Mat2.ajout(1,0,1)
Mat2.ajout(0,1,1.0)
Mat2.ajout(2,2,1.0)
Mat2.ajout(3,2,1.5) 
Mat2.ajout(3,3,1.0)
Mat2.ajout(2,3,1.0)
Mat2.ajout(1,3,1.0)
Mat2.ajout(3,1,5.0)
Mat2.ajout(0,3,2.0)
Mat2dense = Mat2.to_dense()

print('Verification du -')
Mat = Mat-Mat2
Mat3dense = Mat.to_dense()
TEST= np.allclose(Mat3dense, Mat1dense - Mat2dense)
print('Difference correcte ?', TEST)

# Verification de la resolution via decomposition LU
Mat=COOMatrix(4,4,30)
Mat.ajout(0,0,2)
Mat.ajout(1,1,2)
Mat.ajout(2,2,2)
Mat.ajout(3,3,2)
Mat.ajout(0,1,-1)
Mat.ajout(1,0,-1)
Mat.ajout(1,2,-1)
Mat.ajout(2,1,-1)
Mat.ajout(2,3,-1)
Mat.ajout(3,2,-1)


F=np.zeros((4,1))
F[0]=2
F[1]=4
F[2]=6
F[3]=7
A=Mat.to_dense()
print(A)

Mat.lu()
X=Mat.solveLU(F)
print('TEST du LU')
F0=A@X
TEST = np.allclose(F0, F)
print('AX-F=0 par lu puis solvelu',TEST)
X0=Mat.solve(F)
F0=A@X0
TEST = np.allclose(F0, F)
print('AX-F=0 par solve',TEST)

TEST = np.allclose(X, X0)
print('X(solvelu) == X(solve) ?', TEST)


## Test de la forme sesquilineaire V* A U
print('\n=== Test de la forme sesquilineaire V^* A U ===')
V = np.array([1, 2, 3, 4])
U = np.array([1, 1, 1, 1])

# Calcul via la methode sparse puis comparaison a la version dense
result_sparse = Mat.sesquilinear_form(V, U)
result_dense = np.conj(V) @ A @ U
TEST = np.allclose(result_sparse, result_dense)
print('Test de la forme sesquilineaire')
print('Forme sesquilineaire correcte ?', TEST)


## Test de la multiplication par un scalaire
A=COOMatrix(4,4,30)
A.ajout(0,0,2)
A.ajout(1,1,2)
A.ajout(2,2,2)
A.ajout(3,3,2)
A.ajout(3,3,2)
A.ajout(0,1,-1)
A.ajout(1,0,-1)
A.ajout(1,2,-1)
A.ajout(2,1,-1)
A.ajout(2,3,-1)
A.ajout(3,2,-1)

B=100*A
Adense=A.to_dense()
Bdense=B.to_dense()
TEST = np.allclose(Bdense, 100*Adense)
print('\n=== Test de la multiplication par un scalaire ===')
print('Test de la multiplication par un scalaire :', TEST)
## Test du produit matrice-vecteur (a droite)
print('\n=== Test du produit matrice-vecteur a droite ===')
A=COOMatrix(4,4,30)
A.ajout(0,0,2)
A.ajout(1,1,2)
A.ajout(2,2,2)
A.ajout(3,3,2)
A.ajout(0,1,-1)
A.ajout(1,0,-1)
A.ajout(1,2,-1)
A.ajout(2,1,-1)
A.ajout(2,3,-1)
A.ajout(3,2,-1)
A.ajout(3,3,1j)

F=np.zeros((4,1))
F[0]=1
F[1]=2
F[2]=3
F[3]=4
A_dense=A.to_dense()
print(A_dense)
F_result_dense=A_dense@F
F_result_sparse=A@F
TEST = np.allclose(F_result_sparse, F_result_dense)
print('Test du produit matrice-vecteur :', TEST)

print('\n=== Test du produit matrice-vecteur a gauche ===')

F_result_dense=F.T@A_dense
F_result_sparse=A.produit_gauche(F.T)
TEST = np.allclose(F_result_sparse, F_result_dense)
print('Test du produit vecteur-matrice :', TEST)

_=Mat.spy()
