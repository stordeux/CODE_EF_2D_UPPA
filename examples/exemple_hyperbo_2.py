import numpy as np
from mes_packages import *
from mes_packages.methode_hyperbolique import build_exemple_1_F0, build_exemple_1_F1, give_F_format,assemble_surface

ordre =4


# Fin du programme de test de give_F_format
mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.02)
plot_mesh(mesh,secondes=2)

ZERO = lambda x, y: 0.0

Lambda = 1
kappa = 2*np.pi/Lambda

UN = lambda x, y: 1 
h = lambda x, y: 1.j * kappa

# Construction du vecteur nodal pour le problème acoustique normalisé
func_vec = exemple_fonction_vectorielle(theta=np.pi/12, kappa=kappa)
vec_nodal = build_vecteur_nodal_hyperbolique(mesh, ordre, func_vec=func_vec, methode="DG")


F0=np.array([
        [h, ZERO, ZERO],
        [ZERO, h, ZERO],
        [ZERO, ZERO, h]
    ], dtype=object)

Fx = np.array([
        [ZERO, UN, ZERO],
        [UN, ZERO, ZERO],
        [ZERO, ZERO, ZERO]
    ], dtype=object)
Fy = np.array([
        [ZERO, ZERO,UN],
        [ZERO, ZERO, ZERO],
        [UN, ZERO, ZERO]
    ], dtype=object)


# Assemblage des matrices volumiques pour le problème hyperbolique
MASSE_H = assemble_hyperbo(mesh, ordre, opu="u", opv="v", F=F0, kind="volume", methode="DG", domaine="all")
F1D1 = assemble_hyperbo(mesh, ordre, opu="dxu", opv="v", F=Fx, kind="volume", methode="DG", domaine="all")
F2D2 = assemble_hyperbo(mesh, ordre, opu="dyu", opv="v", F=Fy, kind="volume", methode="DG", domaine="all")
F1D1T = assemble_hyperbo(mesh, ordre, opu="u", opv="dxv", F=Fx, kind="volume", methode="DG", domaine="all")
F2D2T = assemble_hyperbo(mesh, ordre, opu="u", opv="dyv", F=Fy, kind="volume", methode="DG", domaine="all")
# Assemblage des matrices volumiques pour le problème hyperbolique
Bextx = assemble_hyperbo(mesh, ordre, opu="nxu", opv="v", F=Fx, kind="frontiere",  methode="DG", domaine="all")
Bexty = assemble_hyperbo(mesh, ordre, opu="nyu", opv="v", F=Fy, kind="frontiere",  methode="DG", domaine="all")
Bintx = assemble_hyperbo(mesh, ordre, opu="nxuT", opv="v", F=Fx, kind="squelette_element",  methode="DG")
Binty = assemble_hyperbo(mesh, ordre, opu="nyuT", opv="v", F=Fy, kind="squelette_element",  methode="DG")
BintxV = assemble_hyperbo(mesh, ordre, opu="nxuV", opv="v", F=Fx, kind="squelette_element",  methode="DG")
BintyV = assemble_hyperbo(mesh, ordre, opu="nyuV", opv="v", F=Fy, kind="squelette_element",  methode="DG")

MAT_vol = COOMatrix(MASSE_H.shape[0], MASSE_H.shape[1], MASSE_H.nnz + F1D1.nnz + F2D2.nnz)
MAT_vol -= MASSE_H
MAT_vol += F1D1
MAT_vol += F2D2
# Vérification de la matrice volumique 
print("Norme de la matrice volumique appliquée au vecteur nodal : ", np.linalg.norm(MAT_vol@vec_nodal))
nnz =Bextx.nnz + Bexty.nnz + Bintx.nnz + Binty.nnz+MASSE_H.nnz+F1D1.nnz+F2D2.nnz+F1D1T.nnz+F2D2T.nnz
IPP = COOMatrix(Bextx.shape[0], Bextx.shape[1], nnz)  # Estimation du nombre d'entrées non nulles
IPP-=Bextx
IPP-=Bexty
IPP-=Bintx
IPP-=Binty
IPP+=F1D1
IPP+=F2D2
IPP+=F1D1T
IPP+=F2D2T
# Test si IPP est nul
print("IPP", IPP.is_zero())


func_F_p = np.array([
    [UN,   ZERO, ZERO],
    [ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO]
], dtype=object)
func_F_vx = np.array([
    [ZERO, ZERO, ZERO],
    [ZERO, UN,   ZERO],
    [ZERO, ZERO, ZERO]
], dtype=object)
func_F_vy = np.array([
    [ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO],
    [ZERO, ZERO, UN]
], dtype=object)
func_F_cross = np.array([
    [ZERO, ZERO, ZERO],
    [ZERO, ZERO, UN  ],
    [ZERO, UN,   ZERO]
], dtype=object)
A_p_TT = assemble_hyperbo(
    mesh, ordre,
    opu="uT", opv="vT",
    F=func_F_p,
    kind="squelette_element",
    methode="DG",
    domaine="all")
A_p_TV = assemble_hyperbo(
    mesh, ordre,
    opu="uT", opv="vV",
    F=func_F_p,
    kind="squelette_element",
    methode="DG",
    domaine="all")
A_xx_TT = assemble_hyperbo(
    mesh, ordre,
    opu="nxuT", opv="nxvT",
    F=func_F_vx,
    kind="squelette_element",
    methode="DG",
    domaine="all")
A_xx_TV = assemble_hyperbo(
    mesh, ordre,
    opu="nxuT", opv="nxvV",
    F=func_F_vx,
    kind="squelette_element",
    methode="DG",
    domaine="all")
A_yy_TT = assemble_hyperbo(
    mesh, ordre,
    opu="nyuT", opv="nyvT",
    F=func_F_vy,
    kind="squelette_element",
    methode="DG",
    domaine="all")
A_yy_TV = assemble_hyperbo(
    mesh, ordre,
    opu="nyuT", opv="nyvV",
    F=func_F_vy,
    kind="squelette_element",
    methode="DG",
    domaine="all")
A_yx_TT = assemble_hyperbo(
    mesh, ordre,
    opu="nyuT", opv="nxvT",
    F=func_F_cross,
    kind="squelette_element",
    methode="DG",
    domaine="all")
A_yx_TV = assemble_hyperbo(
    mesh, ordre,
    opu="nyuT", opv="nxvV",
    F=func_F_cross,
    kind="squelette_element",
    methode="DG",
    domaine="all")
nnz = (A_p_TT.nnz + A_p_TV.nnz+ A_xx_TT.nnz + A_xx_TV.nnz
    + A_yy_TT.nnz + A_yy_TV.nnz+ A_yx_TT.nnz + A_yx_TV.nnz)
nx = A_xx_TT.shape[0]
ny = A_xx_TT.shape[1]

PENALISATION = COOMatrix(nx,ny, nnz)
PENALISATION += A_p_TT
PENALISATION += A_p_TV
PENALISATION += A_xx_TT
PENALISATION += A_xx_TV
PENALISATION += A_yy_TT
PENALISATION += A_yy_TV
PENALISATION += A_yx_TT
PENALISATION += A_yx_TV
nx = A_xx_TT.shape[0]
ny = A_xx_TT.shape[1]
nnz = Bintx.nnz + Binty.nnz + BintxV.nnz + BintyV.nnz
CONTINUITE_RECIPRO = COOMatrix(nx,ny, nnz)
CONTINUITE_RECIPRO -= Bintx
CONTINUITE_RECIPRO -= Binty
CONTINUITE_RECIPRO -= BintxV
CONTINUITE_RECIPRO -= BintyV
print("PENALISATION", PENALISATION.is_positive())



func_F_1 = np.array([
    [ZERO, UN,   ZERO],
    [UN,   ZERO, ZERO  ],
    [ZERO, ZERO, ZERO]
], dtype=object)
func_F_2 = np.array([
    [ZERO, ZERO, UN],
    [ZERO, ZERO, ZERO  ],
    [UN,   ZERO, ZERO]
], dtype=object)

A_px_TT = assemble_hyperbo(
    mesh, ordre,
    opu="nxuT", opv="vT",
    F=func_F_1,
    kind="squelette_element",
    methode="DG",
    domaine="all")
A_px_TV = assemble_hyperbo(
    mesh, ordre,
    opu="nxuV", opv="vT",
    F=func_F_1,
    kind="squelette_element",
    methode="DG",
    domaine="all")
A_py_TT = assemble_hyperbo(
    mesh, ordre,
    opu="nyuT", opv="vT",
    F=func_F_2,
    kind="squelette_element",
    methode="DG",
    domaine="all")  
A_py_TV = assemble_hyperbo(
    mesh, ordre,
    opu="nyuV", opv="vT",
    F=func_F_2,
    kind="squelette_element",
    methode="DG",
    domaine="all")

nnz = A_px_TT.nnz + A_px_TV.nnz + A_py_TT.nnz + A_py_TV.nnz
Saut_reciprocite = COOMatrix(nx, ny, nnz)
Saut_reciprocite -= A_px_TT
Saut_reciprocite -= A_px_TV
Saut_reciprocite -= A_py_TT
Saut_reciprocite -= A_py_TV
print("Saut de réciprocité", Saut_reciprocite.is_zero())

nnz = (A_p_TT.nnz + A_p_TV.nnz+ A_xx_TT.nnz + A_xx_TV.nnz
    + A_yy_TT.nnz + A_yy_TV.nnz+ A_yx_TT.nnz + A_yx_TV.nnz
    + A_px_TT.nnz + A_px_TV.nnz + A_py_TT.nnz + A_py_TV.nnz)
MATRICE_INTERIEURE = COOMatrix(nx, ny, nnz)
MATRICE_INTERIEURE += A_p_TT
MATRICE_INTERIEURE -= A_p_TV
MATRICE_INTERIEURE += A_xx_TT
MATRICE_INTERIEURE -= A_xx_TV
MATRICE_INTERIEURE += A_yy_TT
MATRICE_INTERIEURE -= A_yy_TV
MATRICE_INTERIEURE += A_yx_TT
MATRICE_INTERIEURE += A_yx_TV
MATRICE_INTERIEURE += A_px_TT
MATRICE_INTERIEURE += A_px_TV
MATRICE_INTERIEURE += A_py_TT
MATRICE_INTERIEURE += A_py_TV


print("Verification de la continuité",np.linalg.norm(MATRICE_INTERIEURE@vec_nodal))
