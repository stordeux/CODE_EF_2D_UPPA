"""
Exemple 1: Équation de Laplace sur un domaine carré
Example 1: Laplace equation on a square domain

Problème: -Δu = 0 dans Ω = [0,1] x [0,1]
Problem: -Δu = 0 in Ω = [0,1] x [0,1]

Conditions aux limites / Boundary conditions:
- u = 0 sur les bords gauche, droit et bas / u = 0 on left, right and bottom edges
- u = 1 sur le bord haut / u = 1 on top edge
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.mesh import Mesh
from src.solver import FESolver


def main():
    print("=" * 60)
    print("Exemple 1: Équation de Laplace")
    print("Example 1: Laplace equation")
    print("=" * 60)
    
    # Paramètres du maillage / Mesh parameters
    lx, ly = 1.0, 1.0  # Dimensions du domaine / Domain dimensions
    nx, ny = 10, 10     # Nombre d'éléments / Number of elements
    
    # Création du maillage / Create mesh
    print(f"\nCréation d'un maillage {nx}x{ny}...")
    print(f"Creating a {nx}x{ny} mesh...")
    mesh = Mesh()
    mesh.generate_rectangular_mesh(lx, ly, nx, ny)
    print(f"Maillage créé: {mesh.n_nodes} noeuds, {mesh.n_elements} éléments")
    print(f"Mesh created: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    
    # Création du solveur / Create solver
    print("\nAssemblage du système...")
    print("Assembling system...")
    solver = FESolver(mesh)
    solver.assemble_system(k=1.0)
    
    # Application des conditions aux limites / Apply boundary conditions
    print("Application des conditions aux limites...")
    print("Applying boundary conditions...")
    
    # Trouver les noeuds sur les bords / Find boundary nodes
    bottom_nodes = [i for i, node in enumerate(mesh.nodes) if abs(node.y) < 1e-10]
    top_nodes = [i for i, node in enumerate(mesh.nodes) if abs(node.y - ly) < 1e-10]
    left_nodes = [i for i, node in enumerate(mesh.nodes) if abs(node.x) < 1e-10]
    right_nodes = [i for i, node in enumerate(mesh.nodes) if abs(node.x - lx) < 1e-10]
    
    # u = 0 sur gauche, droite et bas / u = 0 on left, right and bottom
    bc_nodes = list(set(bottom_nodes + left_nodes + right_nodes))
    bc_values = [0.0] * len(bc_nodes)
    solver.apply_dirichlet_bc(bc_nodes, bc_values)
    
    # u = 1 sur le haut / u = 1 on top
    solver.apply_dirichlet_bc(top_nodes, [1.0] * len(top_nodes))
    
    # Résolution / Solve
    print("\nRésolution du système...")
    print("Solving system...")
    solution = solver.solve()
    print(f"Solution calculée: min={solution.min():.4f}, max={solution.max():.4f}")
    print(f"Solution computed: min={solution.min():.4f}, max={solution.max():.4f}")
    
    # Export VTK
    output_file = "laplace_solution.vtk"
    print(f"\nExport de la solution vers {output_file}...")
    print(f"Exporting solution to {output_file}...")
    solver.export_to_vtk(output_file)
    
    # Visualisation / Visualization
    print("\nCréation de la visualisation...")
    print("Creating visualization...")
    coords = mesh.get_coordinates()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Maillage / Mesh
    ax1.set_title("Maillage / Mesh")
    ax1.triplot(coords[:, 0], coords[:, 1], triangles=[[e.nodes[0].id, e.nodes[1].id, e.nodes[2].id] for e in mesh.elements], 'b-', linewidth=0.5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect('equal')
    ax1.grid(True)
    
    # Solution
    ax2.set_title("Solution u(x,y)")
    triangles = [[e.nodes[0].id, e.nodes[1].id, e.nodes[2].id] for e in mesh.elements]
    contour = ax2.tricontourf(coords[:, 0], coords[:, 1], triangles, solution, levels=20, cmap='jet')
    plt.colorbar(contour, ax=ax2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig("laplace_solution.png", dpi=150)
    print("Visualisation sauvegardée: laplace_solution.png")
    print("Visualization saved: laplace_solution.png")
    
    print("\n" + "=" * 60)
    print("Terminé! / Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
