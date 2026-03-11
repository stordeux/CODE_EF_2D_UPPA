"""
Conventions mathématiques utilisées dans ce module
--------------------------------------------------

On travaille sur le triangle de référence

    T_hat = { (x,y) ; x >= 0, y >= 0, x+y <= 1 }

de sommets :
    A0 = (0,0), A1 = (1,0), A2 = (0,1).

On rappelle que l’intérieur du triangle peut aussi être décrit à l’aide des
coordonnées barycentriques par

    x > 0, y > 0, z > 0, avec z = 1 - x - y.

Pour un ordre polynomial `ordre >= 1`, les nœuds d’interpolation sont les
points équidistants du triangle :

    (x_{m,n}, y_{m,n}) = (m/ordre, n/ordre),

où les entiers `m` et `n` vérifient

    m >= 0, n >= 0, m+n <= ordre,

ou encore, en posant

    p = ordre - m - n,

    m >= 0, n >= 0, p >= 0, avec m+n+p = ordre.

Le triplet `(m,n,p)` décrit alors les coordonnées barycentriques discrètes
du nœud :

    (m/ordre, n/ordre, p/ordre).

La fonction `base(x,y,m,n,ordre)` désigne la fonction de base de Lagrange
associée au nœud `(m,n)`. Elle vérifie la propriété d’interpolation :

    phi_{m,n}(x_{m',n'}, y_{m',n'}) = delta_{m,m'} delta_{n,n'}.

La numérotation locale 1D est définie par

    loc2D_to_loc1D(m,n) = i
    avec i = (m+n)(m+n+1)/2 + n.

Cette numérotation correspond à un rangement par diagonales de somme
constante `m+n`, puis par ordre croissant de `n`.

Pour l’ordre 3, on obtient la correspondance suivante entre `(m,n)` et `i` :

    (0,3) [i=9]

    (0,2) [i=5]   (1,2) [i=8]

    (0,1) [i=2]   (1,1) [i=4]   (2,1) [i=7]

    (0,0) [i=0]   (1,0) [i=1]   (2,0) [i=3]   (3,0) [i=6]
"""


import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from numpy.typing import NDArray


# Définition des fonctions de base sur le triangle de référence
def base(x,y, m:int,n:int,ordre:int):
    """
    Évalue la fonction de base de Lagrange associée au nœud `(m,n)`
    sur le triangle de référence.

    Le triangle de référence est
        T_hat = { (x,y) ; x >= 0, y >= 0, x+y <= 1 }.

    Pour un ordre polynomial `ordre`, le nœud `(m,n)` correspond au point
        (m/ordre, n/ordre),
    avec `m >= 0`, `n >= 0` et `m+n <= ordre`.

    Parameters
    ----------
    x, y : float
        Coordonnées du point d'évaluation.
    m, n : int
        Indices du nœud associé à la fonction de base.
    ordre : int
        Ordre polynomial.

    Returns
    -------
    float
        Valeur de la fonction de base au point `(x,y)`.
    """
    P = 1
    z = np.linspace(0,1,ordre+1)
    for i in range(m):
            P *= (x - z[i]) / (z[m] - z[i])            
    for j in range(n):
            P *= (y - z[j]) / (z[n] - z[j])
    for k in range(ordre-n-m):
            P *= (1 - x - y - z[k]) / (1 - z[m] - z[n] - z[k])
    return P
# Dérivée des fonctions de base sur le triangle de référence
def derivative_base(x, y, m=0, n=0, ordre=3, var='x'):
    """
    Calcule la dérivée partielle exacte de la fonction de base
    sur le triangle de référence

    Parameters:
    -----------
    x, y : float
        Point où calculer la dérivée
    m, n : int
        Indices de la fonction de base
    ordre : int
        Ordre de la fonction de base
    var : str
        Variable par rapport à laquelle dériver ('x' ou 'y')
        
    Returns:
    --------
    deriv : float
        Valeur de la dérivée partielle
    """
    z = np.linspace(0, 1, ordre + 1)
    deriv = 0.0
    
    # On utilise la règle de dérivation du produit
    # d/dx[∏ f_i] = ∑_k [f'_k * ∏_{i≠k} f_i]
    
    if var == 'x':
        # Dérivée par rapport à x
        # Produit en x
        for i_prime in range(m):
            term = 1.0 / (z[m] - z[i_prime])
            for i in range(m):
                if i != i_prime:
                    term *= (x - z[i]) / (z[m] - z[i])
            # Produit en y
            for j in range(n):
                term *= (y - z[j]) / (z[n] - z[j])
            # Produit en (1-x-y)
            for k in range(ordre - n - m):
                term *= (1 - x - y - z[k]) / (1 - z[m] - z[n] - z[k])
            deriv += term
        
        # Contribution de la dérivée de (1-x-y)
        for k_prime in range(ordre - n - m):
            term = -1.0 / (1 - z[m] - z[n] - z[k_prime])
            # Produit en x
            for i in range(m):
                term *= (x - z[i]) / (z[m] - z[i])
            # Produit en y
            for j in range(n):
                term *= (y - z[j]) / (z[n] - z[j])
            # Produit en (1-x-y)
            for k in range(ordre - n - m):
                if k != k_prime:
                    term *= (1 - x - y - z[k]) / (1 - z[m] - z[n] - z[k])
            deriv += term
            
    elif var == 'y':
        # Dérivée par rapport à y
        # Contribution de la dérivée de y
        for j_prime in range(n):
            term = 1.0 / (z[n] - z[j_prime])
            # Produit en x
            for i in range(m):
                term *= (x - z[i]) / (z[m] - z[i])
            # Produit en y
            for j in range(n):
                if j != j_prime:
                    term *= (y - z[j]) / (z[n] - z[j])
            # Produit en (1-x-y)
            for k in range(ordre - n - m):
                term *= (1 - x - y - z[k]) / (1 - z[m] - z[n] - z[k])
            deriv += term
        
        # Contribution de la dérivée de (1-x-y)
        for k_prime in range(ordre - n - m):
            term = -1.0 / (1 - z[m] - z[n] - z[k_prime])
            # Produit en x
            for i in range(m):
                term *= (x - z[i]) / (z[m] - z[i])
            # Produit en y
            for j in range(n):
                term *= (y - z[j]) / (z[n] - z[j])
            # Produit en (1-x-y)
            for k in range(ordre - n - m):
                if k != k_prime:
                    term *= (1 - x - y - z[k]) / (1 - z[m] - z[n] - z[k])
            deriv += term
    else:
        raise ValueError("var doit être 'x' ou 'y'")
    
    return deriv
def plot_function_on_triangle(func, A0, A1, A2, n_points=50,SHOW='show'):
    """
    Trace une fonction sur un triangle
    
    Parameters:
    -----------
    func : function
        Fonction à tracer func(x, y)
    A0, A1, A2 : tuple
        Sommets du triangle
    n_points : int
        Nombre de points pour la discrétisation
    """
    A1 = np.array(A1)
    A2 = np.array(A2)
    A0 = np.array(A0)
    
    # Génération des points sur le triangle
    s_vals = np.linspace(0, 1, n_points)
    t_vals = np.linspace(0, 1, n_points)
    
    X_vals = []
    Y_vals = []
    Z_vals = []
    
    for s in s_vals:
        for t in t_vals:
            if s + t <= 1:  # On reste dans le triangle
                point = A0 + s * (A1 - A0) + t * (A2 - A0)
                x, y = point[0], point[1]
                z = func(x, y)
                X_vals.append(x)
                Y_vals.append(y)
                Z_vals.append(z)
    
    # Tracé 3D
    fig = plt.figure(figsize=(12, 5))
    
    # Vue 3D
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(X_vals, Y_vals, Z_vals, c=Z_vals, cmap='viridis', s=1)# type: ignore[arg-type]
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Vue 3D')
    plt.colorbar(scatter, ax=ax1)
    
    # Vue 2D (contour)
    ax2 = fig.add_subplot(122)
    scatter2 = ax2.tricontourf(X_vals, Y_vals, Z_vals, levels=20, cmap='viridis')
    ax2.plot([A1[0], A2[0]], [A1[1], A2[1]], 'k-', linewidth=2)
    ax2.plot([A2[0], A0[0]], [A2[1], A0[1]], 'k-', linewidth=2)
    ax2.plot([A0[0], A1[0]], [A0[1], A1[1]], 'k-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Vue 2D (contours)')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    if SHOW == 'show':
        plt.show()

def loc2D_to_loc1D(m:int,n:int) -> int:    
    """
    Convertit les indices (m,n) d'une fonction de base 2D
    en un indice unique pour une liste 1D
    
    Parameters:
    -----------
    m, n : int
        Indices de la fonction de base 2D
        
    Returns:
    --------
    index : int
        Indice unique correspondant dans une liste 1D
    """
    l = (m + n) * (m + n + 1) // 2 + n
    return l

def loc1D_to_loc2D(index:int) -> tuple[int, int]:
    """
    Convertit un indice 1D en coordonnées (m, n) dans le triangle de référence
    
    Inverse de loc2D_to_loc1D
    
    Parameters:
    -----------
    index : int
        Indice unique dans une liste 1D
        
    Returns:
    --------
    m, n : tuple of int
        Indices de la fonction de base 2D
    """
    # On cherche k tel que k(k+1)/2 <= index < (k+1)(k+2)/2
    # où k = m + n
    k = int((-1 + np.sqrt(1 + 8 * index)) / 2)
    
    # Ajustement si nécessaire
    if k * (k + 1) // 2 > index:
        k -= 1
    
    # n est la position dans la k-ième rangée
    n = index - k * (k + 1) // 2
    
    # m se déduit de m + n = k
    m = k - n
    
    return m, n

def vecteur_nodal_reference(f: Callable[[float,float], float], ordre: int) -> NDArray[np.float64]:
    """
    Évalue une fonction f aux nœuds du triangle de référence
    et retourne le vecteur nodal correspondant.
    
    Le triangle de référence a pour sommets :
    - A0 = (0, 0)
    - A1 = (1, 0)  
    - A2 = (0, 1)
    
    Parameters:
    -----------
    f : Callable[[float, float], float]
        Fonction à évaluer f(x, y) aux noeuds du triangle de référence
    ordre : int
        Ordre polynomial (détermine le nombre de nœuds)
        
    Returns:
    --------
    nodes : float
        Tableau contenant les valeurs évaluées aux nœuds
        Dimension : (Nloc, ) avec Nloc = (ordre+1)*(ordre+2)//2
    """
    N = (ordre + 1) * (ordre + 2) // 2  # Nombre total de nœuds
    out = np.zeros((N,),dtype=float)  # Initialisation du tableau des nœuds
    
    for m in range(ordre + 1):
        for n in range(ordre + 1 - m):
            # Coordonnées du nœud dans le triangle de référence
            x = m / ordre
            y = n / ordre
            i = loc2D_to_loc1D(m, n)
            out[i] = f(x, y)
    return out

def fonction_from_vecteur_nodal(vecteur_nodal: np.ndarray, ordre: int ) -> Callable:
    """
    Crée une fonction interpolante à partir d'un vecteur nodal
    en utilisant les fonctions de base polynomiales.
    
    La fonction retournée est : f(x,y) = Σ u_i * φ_i(x,y)
    où u_i sont les valeurs nodales et φ_i les fonctions de base.
    
    Parameters:
    -----------
    vecteur_nodal : np.ndarray
        Vecteur des valeurs aux nœuds (shape: (Nloc,))
        où Nloc = (ordre+1)*(ordre+2)//2
    ordre : int, optional
        Ordre polynomial des fonctions de base (par défaut: valeur globale ordre)
        
    Returns:
    --------
    f_interpole : Callable
        Fonction interpolante f(x, y) acceptant des scalaires ou des arrays
    """
    Nloc = (ordre + 1) * (ordre + 2) // 2
    
    # Vérification de la dimension
    assert len(vecteur_nodal) == Nloc, f"Le vecteur nodal doit avoir {Nloc} éléments, reçu {len(vecteur_nodal)}"
    
    # S'assurer que vecteur_nodal est un array 1D
    vecteur_nodal_flat = np.asarray(vecteur_nodal).flatten()
    
    def f_interpole(x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        """
        Fonction interpolée évaluée en (x, y)
        
        Parameters:
        -----------
        x, y : float ou np.ndarray
            Coordonnées où évaluer la fonction
            
        Returns:
        --------
        z : float ou np.ndarray
            Valeur(s) de la fonction interpolée
        """
        # Fonction scalaire de base
        def f_scalar(x_val: float, y_val: float) -> float:
            resultat = 0.0
            for m in range(ordre + 1):
                for n in range(ordre + 1 - m):
                    i = loc2D_to_loc1D(m, n)
                    phi_i = base(x_val, y_val, m, n, ordre)
                    # S'assurer que tout est scalaire
                    u_i = float(vecteur_nodal_flat[i])
                    phi_i_scalar = float(phi_i) if np.ndim(phi_i) > 0 else phi_i
                    resultat += u_i * phi_i_scalar
            return resultat
        
        # Gestion scalaire/vectorielle automatique
        f_vec = np.vectorize(f_scalar, otypes=[float])
        result = f_vec(x, y)
        
        # Retourner un scalaire si l'entrée était scalaire
        if np.ndim(result) == 0:
            return float(result)
        else:
            return result
    
    return f_interpole

def base_1D(xi: float | np.ndarray, i: int, ordre: int) -> float | np.ndarray:
    """
    Fonction de base de Lagrange 1D sur [0, 1]
    
    Parameters:
    -----------
    xi : float ou array
        Point(s) d'évaluation sur [0, 1]
    i : int
        Indice de la fonction de base (0 à ordre)
    ordre : int
        Ordre des polynômes
        
    Returns:
    --------
    phi : float ou array
        Valeur de la fonction de base
    """
    # Points de Lagrange équidistants sur [0, 1]
    points = np.linspace(0, 1, ordre + 1)
    xi_i = points[i]
    
    # Calcul du polynôme de Lagrange
    phi = 1.0
    for j in range(ordre + 1):
        if j != i:
            xi_j = points[j]
            phi *= (xi - xi_j) / (xi_i - xi_j)
    
    return phi