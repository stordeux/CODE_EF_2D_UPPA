import numpy as np
from mes_packages import (
    create_mesh_circle_in_square,
    verifier_et_corriger_orientation,
    check_triangle_areas,
    plot_mesh,
    build_neighborhood_structure_with_bc,
)
from mes_packages.mesh import build_neighborhood_structure


def test_mesh():
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    points = mesh.points[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])

    nb_corriges = verifier_et_corriger_orientation(mesh)
    areas, bad = check_triangle_areas(points, triangles, tol=1e-12)

    assert points.shape[1] == 2
    assert triangles.shape[1] == 3
    assert isinstance(nb_corriges, (int, np.integer))
    assert len(bad) == 0


def test_build_neighborhood_structure():
    mesh = create_mesh_circle_in_square(radius=0.1, square_size=0.3, mesh_size=0.025)
    triangles = np.asarray(mesh.cells_dict["triangle"])
    neighbors, neighbor_faces, edges_to_triangles = build_neighborhood_structure(triangles)

    n_tri = len(triangles)
    assert neighbors.shape == (n_tri, 3)
    assert neighbor_faces.shape == (n_tri, 3)

    # Les faces voisines doivent etre -1 (bord) ou 0..2
    assert np.all((neighbor_faces == -1) | ((neighbor_faces >= 0) & (neighbor_faces <= 2)))

    # Voisinage reciproque
    for iT in range(n_tri):
        for local_edge in range(3):
            jT = neighbors[iT, local_edge]
            if jT == -1:
                assert neighbor_faces[iT, local_edge] == -1
                continue

            assert iT in neighbors[jT]
            opp = neighbor_faces[iT, local_edge]
            assert neighbors[jT, opp] == iT

    # Dictionnaire d aretes coherent
    for edge, tri_list in edges_to_triangles.items():
        assert len(edge) == 2
        assert len(tri_list) in (1, 2)
        for tri_idx, local_edge_idx in tri_list:
            assert 0 <= tri_idx < n_tri
            assert 0 <= local_edge_idx < 3



def test_compare_topology_and_bc():
    """
    Vérifie que l'ajout des BC ne modifie PAS la topologie du maillage.

    On doit avoir :
        - mêmes voisins internes
        - transformation exacte des frontières : -1 -> code_BC < 0
        - neighbor_faces inchangé
        - edges_to_triangles inchangé
    """

    mesh = create_mesh_circle_in_square(0.1, 0.3, 0.05)
    triangles = mesh.cells_dict["triangle"]

    # --- Version purement topologique ---
    neigh_topo, neigh_faces_topo, edges_topo = \
        build_neighborhood_structure(triangles)

    # --- Version enrichie BC ---
    neigh_bc, neigh_faces_bc, edges_bc, reference_BC, bc_name = \
        build_neighborhood_structure_with_bc(mesh)

    NT = len(triangles)

    diff_internal = 0
    converted_boundary = 0

    for iT in range(NT):
        for iF in range(3):

            topo_val = neigh_topo[iT, iF]
            bc_val   = neigh_bc[iT, iF]

            if topo_val >= 0:
                # -----------------------------
                # Face interne : DOIT être identique
                # -----------------------------
                assert bc_val == topo_val, (
                    f"Face interne modifiée ! "
                    f"T{iT} F{iF}: topo={topo_val}, bc={bc_val}"
                )
                diff_internal += 1

            else:
                # -----------------------------
                # Face frontière : doit devenir une BC
                # -----------------------------
                assert bc_val < 0, (
                    f"Face frontière devenue interne ! "
                    f"T{iT} F{iF}: {bc_val}"
                )

                assert bc_val != -1, (
                    f"Frontière non typée détectée (encore -1) "
                    f"T{iT} F{iF}"
                )

                converted_boundary += 1

    # --- neighbor_faces doit être strictement identique ---
    assert np.array_equal(neigh_faces_topo, neigh_faces_bc), \
        "neighbor_faces a été modifié par l'injection BC"

    # --- la connectivité duale ne doit PAS changer ---
    assert edges_topo == edges_bc, \
        "edges_to_triangles a été modifié"

    # --- Sanity check : on doit bien avoir trouvé des frontières ---
    assert converted_boundary > 0, "Aucune frontière détectée !"

    # --- (optionnel) vérifier qu'on a au moins deux types de BC ---
    bc_codes = {v for v in neigh_bc.flatten() if v < 0}
    assert len(bc_codes) >= 1, "Aucun code BC trouvé"

