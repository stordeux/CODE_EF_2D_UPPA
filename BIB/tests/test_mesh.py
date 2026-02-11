import numpy as np
from mes_packages import (
    create_mesh_circle_in_square,
    verifier_et_corriger_orientation,
    check_triangle_areas,
    plot_mesh,
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