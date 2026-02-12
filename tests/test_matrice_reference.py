import numpy as np
from scipy.linalg import null_space

from mes_packages import Mref,Krefmixte,Kref,vecteur_nodal_reference,build_masse_locale,build_masse_ref_1D

def test_matrice_masse():
    """Test simple de la matrice de masse locale"""
    ordre = 3
    Mhat = Mref(ordre)
    fx = lambda x, y: x
    fy = lambda x, y: y
    fz = lambda x, y: 1 - x - y
    f1 = lambda x, y: 1.0
    fxy = lambda x, y: x * y
    U1hat=vecteur_nodal_reference(f1, ordre)
    Uxhat=vecteur_nodal_reference(fx, ordre)
    Uyhat=vecteur_nodal_reference(fy, ordre)
    Uzhat=vecteur_nodal_reference(fz, ordre)
    Uxyhat=vecteur_nodal_reference(fxy, ordre)
    assert abs(U1hat.T @ Mhat @ U1hat - 0.5) < 1e-10, "intégrale de 1 sur le triangle = 0.5"
    assert abs(Uxhat.T @ Mhat @ U1hat - 1/6) < 1e-10, "intégrale de x sur le triangle = 1/6"
    assert abs(Uyhat.T @ Mhat @ U1hat - 1/6) < 1e-10, "intégrale de y sur le triangle = 1/6"
    assert abs(Uxyhat.T @ Mhat @ U1hat - 1/24) < 1e-10, "intégrale de xy sur le triangle = 1/24"
    assert abs(Uxhat.T @ Mhat @ Uyhat - 1/24) < 1e-10, "intégrale de xy sur le triangle = 1/24"
    assert abs(Uxhat.T @ Mhat @ Uxhat - 1/12) < 1e-10, "intégrale de x^2 sur le triangle = 1/12"
    assert abs(Uyhat.T @ Mhat @ Uyhat - 1/12) < 1e-10, "intégrale de y^2 sur le triangle = 1/12"
    assert np.allclose(Mhat, Mhat.T), "Matrice de masse locale symétrique"
    
def test_matrice_rigidite():
    """Tests des matrices de rigidité Krefx et Krefy"""
    tol = 1e-10
    ordre = 3
    Krefx, Krefy = Krefmixte(ordre)
    fx = lambda x, y: x
    fy = lambda x, y: y
    fz = lambda x, y: 1 - x - y
    f1 = lambda x, y: 1.0
    fxy = lambda x, y: x*y
    U1hat=vecteur_nodal_reference(f1, ordre)
    Uxhat=vecteur_nodal_reference(fx, ordre)
    Uyhat=vecteur_nodal_reference(fy, ordre)
    Uzhat=vecteur_nodal_reference(fz, ordre)
    Uxyhat=vecteur_nodal_reference(fxy, ordre)
    # Test 2
    assert abs(U1hat.T @ Krefy @ U1hat) < tol, "∫ 1 * ∂(1)/∂y = 0"
    # Test 3
    assert abs(Uxhat.T @ Krefx @ U1hat) < tol, "∫ x * ∂(1)/∂x = -1/2"
    # Test 4
    assert abs(Uyhat.T @ Krefy @ U1hat) < tol, "∫ y * ∂(1)/∂y = -1/2"
    # Test 5
    assert abs(U1hat.T @ Krefx @ Uxhat - 0.5) < tol, "∫ 1 * ∂(x)/∂x = 1/2"
    # Test 6
    assert abs(U1hat.T @ Krefy @ Uyhat - 0.5) < tol, "∫ 1 * ∂(y)/∂y = 1/2"
    # Test 7
    assert abs(Uxhat.T @ Krefy @ U1hat) < tol, "∫ x * ∂(1)/∂y = 0"
    # Test 8
    assert abs(Uyhat.T @ Krefx @ U1hat) < tol, "∫ y * ∂(1)/∂x = 0"
    # Test 9
    assert abs(Uxhat.T @ Krefx @ Uxhat - 1/6) < tol, "∫ x * ∂(x)/∂x = 1/6"    
    # Test 10
    assert abs(Uyhat.T @ Krefy @ Uyhat - 1/6) < tol, "∫ y * ∂(y)/∂y = 1/6"

def test_matrices_rigidite_complete():
    """Tests complets des matrices de rigidité"""
    tol = 1e-10
    ordre =3
    Krefxx,Krefxy,Krefyx,Krefyy = Kref(ordre)
    fx = lambda x, y: x
    fy = lambda x, y: y
    f1 = lambda x, y: 1.0
    U1hat=vecteur_nodal_reference(f1, ordre)
    Uxhat=vecteur_nodal_reference(fx, ordre)
    Uyhat=vecteur_nodal_reference(fy, ordre)
    
    
    # Vérifications de symétrie
    assert np.allclose(Krefxx, Krefxx.T), "Krefxx doit être symétrique"
    assert np.allclose(Krefyy, Krefyy.T), "Krefyy doit être symétrique"
    assert np.allclose(Krefyx, Krefxy.T), "Krefxy doit être la transposée de Krefyx"
    
    # Tests sur la fonction constante
    assert abs(U1hat.T @ Krefyy @ U1hat) < tol, "∫ (∂1/∂y)·(∂1/∂y) = 0"
    
    # Tests sur les dérivées croisées
    assert abs(Uxhat.T @ Krefxx @ U1hat) < tol, "∫ (∂x/∂x)·(∂1/∂x) = 0"
    assert abs(Uyhat.T @ Krefyy @ U1hat) < tol, "∫ (∂y/∂y)·(∂1/∂y) = 0"
    assert abs(U1hat.T @ Krefxx @ Uxhat) < tol, "∫ (∂1/∂x)·(∂x/∂x) = 0"
    assert abs(U1hat.T @ Krefyy @ Uyhat) < tol, "∫ (∂1/∂y)·(∂y/∂y) = 0"
    assert abs(Uxhat.T @ Krefxy @ U1hat) < tol, "∫ (∂x/∂x)·(∂1/∂y) = 0"
    assert abs(Uyhat.T @ Krefyx @ U1hat) < tol, "∫ (∂y/∂y)·(∂1/∂x) = 0"
    
    # Tests sur les termes diagonaux
    assert abs(Uxhat.T @ Krefxx @ Uxhat - 0.5) < tol, "∫ (∂x/∂x)·(∂x/∂x) = 1/2"
    assert abs(Uyhat.T @ Krefyy @ Uyhat - 0.5) < tol, "∫ (∂y/∂y)·(∂y/∂y) = 1/2"
    
    # Tests sur les termes mixtes
    assert abs(Uxhat.T @ Krefxy @ Uyhat - 0.5) < tol, "∫ (∂x/∂x)·(∂y/∂y) = 1/2"
    assert abs(Uyhat.T @ Krefxy @ Uxhat) < tol, "∫ (∂y/∂x)·(∂x/∂y) = 0"
    
def test_matrice_masse_1D():
    """Tests de la matrice de masse 1D de référence"""
    
    for ordre_test in [1, 2, 3]:
        M1D = build_masse_ref_1D(ordre_test)
        assert np.allclose(np.sum(M1D), 1.0), f"Somme Ok pour ordre {ordre_test}"
        assert np.allclose(M1D, M1D.T), f"Symétrie Ok pour ordre {ordre_test}"
        assert np.all(np.linalg.eigvals(M1D) > 0), f"Positivité des valeurs propres pour ordre {ordre_test}"

def test_noyau_Mixte():
    for ordre in range(1, 5):
        Krefx, Krefy = Krefmixte(ordre) 
        # Calcul du noyau (null space)
        noyau_Krefx = null_space(Krefx, rcond=1e-10)
        noyau_Krefy = null_space(Krefy, rcond=1e-10)
        assert noyau_Krefx.shape[1] == ordre+1, f"Le noyau de Krefx n'est pas de dimension  {ordre+1}"
        assert noyau_Krefy.shape[1] == ordre+1, f"Le noyau de Krefy n'est pas de dimension  {ordre+1}"

def test_Mlocal():
    for ordre in range(1, 5):
        Mlocref = Mref(ordre)
        A1= (0, 0)
        A2= (3, 0)
        A3= (0, 4)
        A1= np.array(A1)
        A2= np.array(A2)
        A3= np.array(A3)
        Mloc= build_masse_locale(ordre,A1, A2, A3)
        J = np.column_stack((A2 - A1, A3 - A1))
        detJ = np.linalg.det(J)
        assert abs(detJ-12) < 1e-10, "Le déterminant de J doit être égal à 12 pour le triangle (0,0),(3,0),(0,4)"
        # Caclcul des vlaeurs propres de Mloc
        values, vecteurs_propres = np.linalg.eig(Mloc)
        # Vérification que toutes les valeurs propres sont strictement positives
        toutes_positives = np.all(values > 0)
        assert toutes_positives, "La matrice de masse locale doit être définie positive"
        assert np.allclose(Mloc, detJ * Mlocref), "Mloc doit être égal à det(J) * Mref"

def test_Mlocal2():
    for ordre in range(1, 5):
        A1= (0, 0)
        A2= (1, 0)
        A3= (0, 1)
        Mloc = build_masse_locale(ordre,A1,A2,A3)
        assert np.allclose(Mloc , Mref(ordre)), "Mloc doit être égal à Mref pour le triangle de référence"
