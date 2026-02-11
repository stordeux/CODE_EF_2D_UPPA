import numpy as np
from numpy.polynomial.legendre import leggauss
from mes_packages import quadrature


def test_integrate_triangle_2D_simple():
    # Fonction à intégrer
    def f(x, y, m=1, n=1):
        return x**m * y**n

    from numpy.polynomial.legendre import leggauss
    ordre = 4
    xi, w = leggauss(ordre)

    A1 = (0, 0)
    A2 = (1, 0)
    A3 = (0, 1)

    I = quadrature.integrate_triangle_2D(lambda x, y: f(x, y, m=1, n=2), A1, A2, A3, xi, w)
    # L'intégrale exacte de x*y^2 sur le triangle unité est 1/24
    expected = 1/60
    assert np.isclose(I, expected, rtol=1e-6), f"Résultat inattendu: {I} (attendu: {expected})"

def test_integrate_segment_2D():
    # Fonction à intégrer
    def f(x,y):
        return x + y

    ordre = 4
    xi,w = leggauss(ordre)

    # Points du segment
    A1 = (0, 0)
    A2 = (3, 2)

    # Intégration avec l'ancienne fonction
    I_old = quadrature.integrate_segment_2D_old(f, A1, A2, xi, w)

    # Intégration avec la nouvelle fonction
    I_new = quadrature.integrate_segment_2D(f, A1, A2, xi, w)

    # Vérification de l'égalité des résultats
    assert np.isclose(I_old, I_new), f"Les intégrales ne correspondent pas: {I_old} != {I_new}"




