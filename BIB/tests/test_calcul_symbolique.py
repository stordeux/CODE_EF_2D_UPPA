import numpy as np
import sympy as sp

from mes_packages.calcul_symbolique import build_f_and_grads

def test_build_f_and_grads():
    ## Test de cette fonction
    x, y = sp.symbols('x y')
    f_sym =  sp.sin(10* y)
    f,fx,fy = build_f_and_grads(f_sym, (x,y))
    TEST = np.isclose(np.sin(10*0.5),f(0,0.5))
    TESTx = np.isclose(0,fx(0,0.5))
    TESTy = np.isclose(10*np.cos(10*0.5),fy(0,0.5))
    print("Test fonction f correcte :", TEST)
    print("Test dérivée partielle en x correcte :", TESTx)
    print("Test dérivée partielle en y correcte :", TESTy)