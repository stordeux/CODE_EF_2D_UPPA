import numpy as np

def integrate_segment_2D_old(f, A1, A2, xi, w):
    """
    Intègre une fonction f(x,y) sur un segment 2D [A1, A2]
    en utilisant la quadrature de Gauss
    
    Parameters:²
    -----------
    f : function
        Fonction à intégrer f(x,y)
    A1 : tuple ou array
        Point initial (x1, y1)
    A2 : tuple ou array
        Point final (x2, y2)
    xi : array
        Points de Gauss sur [-1,1]
    w : array
        Poids de Gauss
        
    Returns:
    --------
    I : float
        Valeur de l'intégrale
    """
    A1 = np.array(A1)
    A2 = np.array(A2)
    
    # Longueur du segment
    L = np.linalg.norm(A2 - A1)
    
    # Jacobien de la transformation
    J = L / 2
    
    # Calcul de l'intégrale
    I = 0.0
    for i in range(len(xi)):
        # Transformation du point de Gauss sur le segment réel
        # xi_i ∈ [-1,1] -> point sur [A1, A2]
        t = (1 + xi[i]) / 2  # t ∈ [0,1]
        point = A1 + t * (A2 - A1)
        x, y = point[0], point[1]
        
        # Contribution à l'intégrale
        I += w[i] * f(x, y) * J
    
    return I

def integrate_segment_2D(f, A1, A2, xi, w):
    """
    Intègre f(x,y) le long du segment 2D [A1, A2] (intégrale curviligne) :
        I = ∫_{segment} f(x,y) ds
    via quadrature de Gauss sur [-1,1].

    Paramètres
    ----------
    f  : callable
         f(x,y) (peut être complexe). Idéalement accepte x et y sous forme d'arrays NumPy.
    A1 : (2,) array-like
         Point initial (x1, y1)
    A2 : (2,) array-like
         Point final (x2, y2)
    xi : (n,) array-like
         Points de Gauss sur [-1,1]
    w  : (n,) array-like
         Poids de Gauss

    Retour
    ------
    I : complex
        Valeur de l'intégrale (complexe même si f est réelle)
    """
    A1 = np.asarray(A1, dtype=float)
    A2 = np.asarray(A2, dtype=float)
    xi = np.asarray(xi, dtype=float)
    w  = np.asarray(w, dtype=float)

    assert xi.shape == w.shape, "xi et w doivent avoir la même taille"
    assert A1.shape == (2,) and A2.shape == (2,), "A1 et A2 doivent être des points 2D (2,)"

    d = A2 - A1
    L = np.linalg.norm(d)
    if L == 0.0:
        return 0.0 + 0.0j

    J = L / 2.0  # Jacobien ds = (L/2) dxi

    # Vectorisation : xi -> t in [0,1], puis points sur le segment
    t = (1.0 + xi) / 2.0                         # (n,)
    pts = A1[None, :] + t[:, None] * d[None, :]  # (n,2)
    x = pts[:, 0]
    y = pts[:, 1]

    # Si f n'est pas vectorisée, on retombe proprement sur une évaluation point-par-point
    try:
        vals = f(x, y)
        vals = np.asarray(vals)
        if vals.shape != x.shape:
            raise ValueError
    except Exception:
        vals = np.array([f(float(xi_), float(yi_)) for xi_, yi_ in zip(x, y)], dtype=complex)

    return J * np.sum(w * vals)

def integrate_triangle_2D(f, A1, A2, A3, xi, w):
    """
    Intègre une fonction f(x,y) sur un triangle 2D [A1, A2, A3]
    en utilisant la quadrature de Gauss
    
    Parameters:
    -----------
    f : function
        Fonction à intégrer f(x,y)
    A1 : tuple ou array
        Premier sommet (x1, y1)
    A2 : tuple ou array
        Deuxième sommet (x2, y2)
    A3 : tuple ou array
        Troisième sommet (x3, y3)
    xi : array
        Points de Gauss sur [-1,1]
    w : array
        Poids de Gauss
        
    Returns:
    --------
    I : float
        Valeur de l'intégrale
    """
    A1 = np.array(A1)
    A2 = np.array(A2)
    A3 = np.array(A3)
    
    # Jacobien : aire du triangle * (1-s) * (1/4 pour transformation [-1,1]² -> [0,1]²)
    aire = abs((A2[0] - A1[0]) * (A3[1] - A1[1]) - (A3[0] - A1[0]) * (A2[1] - A1[1])) / 2
    
    # Calcul de l'intégrale par double quadrature de Gauss
    I = 0.0
    for i in range(len(xi)):
        for j in range(len(xi)):
            # Transformation des points de Gauss [-1,1] vers [0,1]
            s = (1 + xi[i]) / 2
            t = (1 + xi[j]) / 2
            
            # Transformation vers le triangle réel
            point = A1 + s * (A2 - A1) + t * (1 - s) * (A3 - A1)
            x, y = point[0], point[1]
            
            # Jacobien complet : aire * (1-s) * 1/4
            J = aire * (1 - s)/2 
            
            # Contribution à l'intégrale
            I += w[i] * w[j] * f(x, y) * J
    
    return I

def integrate_triangle_2D_product(f,g, A1, A2, A3, xi, w):
    """
    Intègre une fonction f(x,y) sur un triangle 2D [A1, A2, A3]
    en utilisant la quadrature de Gauss
    
    Parameters:
    -----------
    f : function
        Fonction à intégrer f(x,y)
    g : function
        Deuxième fonction à intégrer g(x,y)
    A1 : tuple ou array
        Premier sommet (x1, y1)
    A2 : tuple ou array
        Deuxième sommet (x2, y2)
    A3 : tuple ou array
        Troisième sommet (x3, y3)
    xi : array
        Points de Gauss sur [-1,1]
    w : array
        Poids de Gauss
        
    Returns:
    --------
    I : float
        Valeur de l'intégrale
    """
    A1 = np.array(A1)
    A2 = np.array(A2)
    A3 = np.array(A3)
    
    # Jacobien : aire du triangle * (1-s) * (1/4 pour transformation [-1,1]² -> [0,1]²)
    aire = abs((A2[0] - A1[0]) * (A3[1] - A1[1]) - (A3[0] - A1[0]) * (A2[1] - A1[1])) / 2
    
    # Calcul de l'intégrale par double quadrature de Gauss
    I = 0.0
    for i in range(len(xi)):
        for j in range(len(xi)):
            # Transformation des points de Gauss [-1,1] vers [0,1]
            s = (1 + xi[i]) / 2
            t = (1 + xi[j]) / 2
            
            # Transformation vers le triangle réel
            point = A1 + s * (A2 - A1) + t * (1 - s) * (A3 - A1)
            x, y = point[0], point[1]
            
            # Jacobien complet : aire * (1-s) * 1/4
            J = aire * (1 - s)/2 
            
            # Contribution à l'intégrale
            I += w[i] * w[j] * f(x, y) * g(x, y) * J
    
    return I


