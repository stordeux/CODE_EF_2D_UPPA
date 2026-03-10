from mes_packages import COOMatrix
import numpy as np

def test_add_and_iadd_consistency():
    Mat = COOMatrix(4, 4, 30)
    Mat.ajout(1, 0, 1)
    Mat.ajout(0, 1, 1.0)
    Mat.ajout(2, 2, 1.0)
    Mat.ajout(3, 2, 1.5)
    Mat.ajout(3, 3, 1.0)
    Mat.ajout(2, 3, 1.0)
    Mat.ajout(1, 3, 1.0)
    Mat.ajout(3, 1, 5.0)
    Mat.ajout(0, 3, 2.0)
    Mat.ajout(0, 3, np.pi)

    Mat2 = COOMatrix(4, 4, 30)
    Mat2.ajout(1, 0, 1)
    Mat2.ajout(0, 1, 1.0)
    Mat2.ajout(2, 2, 1.0)
    Mat2.ajout(3, 2, 1.5)
    Mat2.ajout(3, 3, 1.0)
    Mat2.ajout(2, 3, 1.0)
    Mat2.ajout(1, 3, 1.0)
    Mat2.ajout(3, 1, 5.0)
    Mat2.ajout(0, 3, 2.0)
    Mat2.ajout(0, 3, np.pi)

    Mat1dense = Mat.to_dense()
    Mat2dense = Mat2.to_dense()

    # Test du +
    A = Mat + Mat2
    Adense = A.to_dense()

    # Test du +=
    Mat += Mat2
    Mat3dense = Mat.to_dense()

    expected = Mat1dense + Mat2dense

    assert np.allclose(Adense, expected), "A = Mat + Mat2 incorrect"
    assert np.allclose(Mat3dense, expected), "Mat += Mat2 incorrect"
    assert np.allclose(Adense, Mat3dense), "+ et += ne donnent pas le même résultat"
    assert np.allclose(Mat2.to_dense(), Mat2dense), "Mat2 a été modifiée"


def test_subtraction_inplace():
    # Vérification de la soustraction
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

    Mat=Mat-Mat2
    Mat3dense = Mat.to_dense()
    TEST= np.allclose(Mat3dense, Mat1dense - Mat2dense)
    assert TEST, "Subtraction inplace failed"

def test_lu_and_solve():
    # Vérification du LU 
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

    Mat.lu()
    X=Mat.solveLU(F)
    F0=A@X
    TEST = np.allclose(F0, F)
    X0=Mat.solve(F)
    F0=A@X0
    TEST = np.allclose(F0, F)

    TEST = np.allclose(X, X0)
    assert TEST, "LU and Solve failed"

def test_sesquilinear_form():
    Mat = COOMatrix(4,4,30)
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

    A = Mat.to_dense()

    V = np.array([1, 2, 3, 4])
    U = np.array([1, 1, 1, 1])

    result_sparse = Mat.sesquilinear_form(V, U)
    result_dense = np.conj(V) @ A @ U
    assert np.allclose(result_sparse, result_dense), "Sesquilinear form failed"

# Test de la multiplication par un scalaire
def test_scalar_multiplication():
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
    assert TEST, "Scalar multiplication failed"
# Test du produit matrice-vecteur
def test_matvec_right_and_left():
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

    assert np.allclose(A @ F, A_dense @ F)
    assert np.allclose(A.produit_gauche(F.T), F.T @ A_dense)


