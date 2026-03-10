import numpy as np
from mes_packages import *


# --------------------------------------------------
# Hexagone extérieur
# --------------------------------------------------

def regular_hexagon(R=1.0):
    pts = []
    for k in range(6):
        theta = 2*np.pi*k/6
        pts.append((R*np.cos(theta), R*np.sin(theta)))
    return pts


# --------------------------------------------------
# Etoile à 5 branches
# --------------------------------------------------

def star_polygon(R_outer=0.4, R_inner=0.18, n=5):
    pts = []
    for k in range(2*n):
        theta = k*np.pi/n
        r = R_outer if k % 2 == 0 else R_inner
        pts.append((r*np.cos(theta), r*np.sin(theta)))
    return pts


# --------------------------------------------------
# Construction
# --------------------------------------------------

outer = regular_hexagon(1.0)
inner = star_polygon()
mesh = create_mesh_polygon_with_hole(outer, inner, mesh_size=0.05)
plot_mesh(mesh,secondes=2)
plot_mesh_with_bc(mesh,secondes=0)