import numpy as np
from mes_packages.base import base, derivative_base, loc2D_to_loc1D, loc1D_to_loc2D
from mes_packages.matrice_reference import build_masse_ref_1D 

def test_base():
    ordre =3
    for m1 in range(ordre+1):
        for n1 in range(ordre+1-m1):
            for m2 in range(ordre+1):
                for n2 in range(ordre+1-m2):
                    # Test de l'orthogonalité
                    x = m2 / ordre
                    y = n2 / ordre
                    val = base(x, y, m1, n1, ordre)
                    if m1 == m2 and n1 == n2:
                        assert np.isclose(val, 1.0), f"Base non normalisée pour m={m1}, n={n1}: {val} != 1" 
                    else:
                        assert np.isclose(val, 0.0), f"Base non orthogonale pour m={m1}, n={n1} et m={m2}, n={n2}: {val} != 0"

def test_derive_base():
    # Test de la dérivée par rapport à x
    x = 0.3
    y = 0.2
    ordre = 3
    for m in range(ordre+1):
        for n in range(ordre+1-m):
            val_base = base(x, y, m, n, ordre)
            val_deriv_x = derivative_base(x, y, m, n, ordre, var='x')
            # Calcul de la dérivée numérique par différence finie
            h = 1e-6
            val_plus = base(x + h, y, m, n, ordre)
            val_minus = base(x - h, y, m, n, ordre)
            deriv_numerique_x = (val_plus - val_minus) / (2 * h)
            assert np.isclose(val_deriv_x, deriv_numerique_x, rtol=1e-5), f"Dérivée x incorrecte pour m={m}, n={n}: {val_deriv_x} != {deriv_numerique_x}"

def test_loc1D_to_loc2D():
    # Test de la fonction inverse
    print("\n=== Test de loc1D_to_loc2D ===")
    for idx in range(10):
        m, n = loc1D_to_loc2D(idx)
        idx_back = loc2D_to_loc1D(m, n)
        assert idx == idx_back, f"Erreur de conversion pour index {idx}: (m,n)=({m},{n}) -> {idx_back} != {idx}"


# Test de la matrice de masse 1D
print("=== Test de la matrice de masse 1D de référence ===\n")

def test_masse_ref_1D():
    for ordre_test in [1, 2, 3]:
        M1D = build_masse_ref_1D(ordre_test)
        TEST = np.allclose(np.sum(M1D), 1.0) 
        assert TEST, f"Somme des éléments de M1D pour ordre {ordre_test} incorrecte: {np.sum(M1D)} != 1"
        TEST = np.allclose(M1D, M1D.T) 
        assert TEST, f"Matrice de masse 1D pour ordre {ordre_test} n'est pas symétrique"
        TEST = np.all(np.linalg.eigvals(M1D) > 0)
        assert TEST, f"Matrice de masse 1D pour ordre {ordre_test} n'est pas définie positive"