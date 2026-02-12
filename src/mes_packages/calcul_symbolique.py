import sympy as sp

def build_f_and_grads(f_symf_, vars):
    x,y=vars
    # Dérivées partielles exactes
    dfdx_sym = sp.diff(f_symf_, x)
    dfdy_sym = sp.diff(f_symf_, y)
    # Fonctions lambda numériques
    f   = sp.lambdify((x, y), f_symf_,    'numpy')
    dxf = sp.lambdify((x, y), dfdx_sym, 'numpy')
    dyf = sp.lambdify((x, y), dfdy_sym, 'numpy')
    return f,dxf,dyf