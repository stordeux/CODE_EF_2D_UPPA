import numpy as np
from .base import base, derivative_base, loc2D_to_loc1D, loc1D_to_loc2D,base_1D
from .quadrature import integrate_triangle_2D_product
from numpy.polynomial.legendre import leggauss

def Mref(ordre:int) -> np.ndarray:
    """
    Construit la matrice de masse de référence sur le triangle de référence.
    Elle contient les intégrales du produit de toutes les paires de fonctions de base.
    """
    # Calcul du nombre de fonctions de base locales pour l'ordre polynomial donné
    # Formule: (ordre+2)*(ordre+1)/2 = nombre total de nœuds d'interpolation
    Nloc = (ordre+2) * (ordre+1)//2
    
    # Création d'une matrice Nloc x Nloc initialisée à zéro
    Mloc = np.zeros((Nloc, Nloc))
    xi, w = leggauss(ordre+1)  # Points et poids de quadrature de Gauss-Legendre    
    # Boucle sur les indices 2D (m1, n1) de la première fonction de base
    for m1 in range(ordre+1):
        for n1 in range(ordre+1 - m1):
            # Conversion des indices 2D en indice 1D linéaire pour la première fonction
            i = loc2D_to_loc1D(m1, n1)

            # Boucle sur les indices 2D (m2, n2) de la deuxième fonction de base
            for m2 in range(ordre+1):
                for n2 in range(ordre+1 - m2):
                    # Conversion des indices 2D en indice 1D linéaire pour la deuxième fonction
                    j = loc2D_to_loc1D(m2, n2)
                    
                    # Calcul de l'intégrale du produit des deux fonctions de base
                    # sur le triangle de référence avec sommets (0,0), (1,0), (0,1)
                    Mloc[i, j] = integrate_triangle_2D_product(
                        lambda x, y: base(x, y, m1, n1, ordre),      # Fonction de base i
                        lambda x, y: base(x, y, m2, n2, ordre),      # Fonction de base j
                        (0, 0), (1, 0), (0, 1),                       # Sommets du triangle
                        xi, w)                                          # Points et poids de quadrature    
    return Mloc

def Krefmixte(ordre:int) -> tuple[np.ndarray, np.ndarray] :
    xi, w = leggauss(ordre+1)  # Points et poids de quadrature de Gauss-Legendre
    Nloc = (ordre+2) * (ordre+1)//2
    Klocx = np.zeros((Nloc, Nloc))
    Klocy = np.zeros((Nloc, Nloc))
    for m1 in range(ordre+1):
        for n1 in range(ordre+1 - m1):
            i = loc2D_to_loc1D(m1, n1)
            for m2 in range(ordre+1):
                for n2 in range(ordre+1 - m2):
                    j = loc2D_to_loc1D(m2, n2)
                    # Intégrale de la dérivée partielle de la fonction de base i * fonction de base j sur le triangle de référence
                    Klocx[i, j] = integrate_triangle_2D_product(
                        lambda x, y: base(x, y, m1, n1, ordre),
                        lambda x, y: derivative_base(x, y, m2, n2, ordre, var='x'),
                        (0, 0), (1, 0), (0, 1),
                        xi, w
                    )
                    Klocy[i, j] = integrate_triangle_2D_product(
                        lambda x, y: base(x, y, m1, n1, ordre),
                        lambda x, y: derivative_base(x, y, m2, n2, ordre, var='y'),
                        (0, 0), (1, 0), (0, 1),
                        xi, w
                    )
    return Klocx, Klocy

def Kref(ordre:int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xi, w = leggauss(ordre+1)  # Points et poids de quadrature de Gauss-Legendre
    Nloc = (ordre+2) * (ordre+1)//2
    Kxx = np.zeros((Nloc, Nloc))
    Kxy = np.zeros((Nloc, Nloc))
    Kyx = np.zeros((Nloc, Nloc))
    Kyy = np.zeros((Nloc, Nloc))
    
    for m1 in range(ordre+1):
        for n1 in range(ordre+1 - m1):
            i = loc2D_to_loc1D(m1, n1)
            for m2 in range(ordre+1):
                for n2 in range(ordre+1 - m2):
                    j = loc2D_to_loc1D(m2, n2)
                    
                    # Kxx : d(base_i)/dx * d(base_j)/dx
                    Kxx[i, j] = integrate_triangle_2D_product(
                        lambda x, y: derivative_base(x, y, m1, n1, ordre, var='x'),
                        lambda x, y: derivative_base(x, y, m2, n2, ordre, var='x'),
                        (0, 0), (1, 0), (0, 1),
                        xi, w
                    )
                    
                    # Kxy : d(base_i)/dx * d(base_j)/dy
                    Kxy[i, j] = integrate_triangle_2D_product(
                        lambda x, y: derivative_base(x, y, m1, n1, ordre, var='x'),
                        lambda x, y: derivative_base(x, y, m2, n2, ordre, var='y'),
                        (0, 0), (1, 0), (0, 1),
                        xi, w
                    )
                    
                    # Kyx : d(base_i)/dy * d(base_j)/dx
                    Kyx[i, j] = integrate_triangle_2D_product(
                        lambda x, y: derivative_base(x, y, m1, n1, ordre, var='y'),
                        lambda x, y: derivative_base(x, y, m2, n2, ordre, var='x'),
                        (0, 0), (1, 0), (0, 1),
                        xi, w
                    )
                    
                    # Kyy : d(base_i)/dy * d(base_j)/dy
                    Kyy[i, j] = integrate_triangle_2D_product(
                        lambda x, y: derivative_base(x, y, m1, n1, ordre, var='y'),
                        lambda x, y: derivative_base(x, y, m2, n2, ordre, var='y'),
                        (0, 0), (1, 0), (0, 1),
                        xi, w
                    )
                    
    return Kxx, Kxy, Kyx, Kyy

# Calcul de la matrice de masse locale 
def build_masse_locale(ordre,A1,A2,A3):
    Mloc = Mref(ordre)
    # Calcul de la transformation affine du triangle de référence au triangle réel
    A1 = np.array(A1)
    A2 = np.array(A2)
    A3 = np.array(A3)
    # Calcul de la matrice jacobienne de la transformation affine
    J = np.column_stack((A2 - A1, A3 - A1))
    detJ = abs(np.linalg.det(J))
    return detJ*Mloc

def build_masse_ref_1D(ordre):
    """
    Calcule la matrice de masse de référence 1D sur [-1, 1]
    
    M_{ij} = ∫_{-1}^{1} φ_i(ξ) φ_j(ξ) dξ
    
    Parameters:
    -----------
    ordre : int
        Ordre des polynômes de Lagrange
        
    Returns:
    --------
    M : array (ordre+1, ordre+1)
        Matrice de masse 1D de référence
    """
    n_dof = ordre + 1
    M = np.zeros((n_dof, n_dof))
    
    # Points et poids de Gauss pour l'intégration
    n_gauss = ordre + 2  # Assez de points pour intégrer exactement
    xi_gauss, w_gauss = leggauss(n_gauss)
    
    for i in range(n_dof):
        for j in range(n_dof):
            # Intégration par quadrature de Gauss
            integral = 0.0
            for k in range(n_gauss):
                xi = xi_gauss[k]
                phi_i = base_1D((1+xi)/2, i, ordre)
                phi_j = base_1D((1+xi)/2, j, ordre)
                integral += w_gauss[k] * phi_i * phi_j / 2
            
            M[i, j] = integral
    
    return M

def build_mixte_locale(ordre, A1, A2, A3):
    # Calcul de la transformation affine du triangle de référence au triangle réel
    A1 = np.array(A1)
    A2 = np.array(A2)
    A3 = np.array(A3)
    
    # Calcul de la matrice jacobienne de la transformation affine
    J = np.column_stack((A2 - A1, A3 - A1))
    detJ = abs(np.linalg.det(J))
    J_inv = np.linalg.inv(J)
    
    # Matrices de référence
    Krefx, Krefy = Krefmixte(ordre)
    
    
    Klocx = detJ * (J_inv[0, 0] * Krefx + J_inv[1, 0] * Krefy)
    Klocy = detJ * (J_inv[0, 1] * Krefx + J_inv[1, 1] * Krefy)
    
    return Klocx, Klocy

def build_rigidite_locale(ordre:int, A1, A2, A3, equation: str)-> np.ndarray:    
    # Calcul de la transformation affine du triangle de référence au triangle réel
    A1 = np.array(A1)
    A2 = np.array(A2)
    A3 = np.array(A3)
    
    # Calcul de la matrice jacobienne de la transformation affine
    J = np.column_stack((A2 - A1, A3 - A1))
    detJ = abs(np.linalg.det(J))
    J_inv = np.linalg.inv(J)
    
    # Matrices de référence
    Krefxx, Krefxy, Krefyx, Krefyy = Kref(ordre)
    
    Klocxx = detJ * (J_inv[0, 0]**2 * Krefxx + J_inv[0, 0]*J_inv[1, 0]*(Krefxy + Krefyx) + J_inv[1, 0]**2 * Krefyy)
    Klocxy = detJ * (J_inv[0, 0]*J_inv[0, 1] * Krefxx + J_inv[0, 0]*J_inv[1, 1]*Krefxy + J_inv[1, 0]*J_inv[0, 1]*Krefyx + J_inv[1, 0]*J_inv[1, 1]*Krefyy)
    Klocyy = detJ * (J_inv[0, 1]**2 * Krefxx + J_inv[0, 1]*J_inv[1, 1]*(Krefxy + Krefyx) + J_inv[1, 1]**2 * Krefyy)
    if equation == 'laplace':
        Kloc = Klocxx + Klocyy
        return Kloc
    if equation == 'xx':
        return Klocxx
    if equation == 'xy':
        return Klocxy
    if equation == 'yx':
        return Klocxy.T
    if equation == 'yy':
        return Klocyy
    raise ValueError(f"equation inconnue: {equation}")

