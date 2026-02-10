"""
Exemple 2: Équation de Poisson avec terme source
Example 2: Poisson equation with source term

Problème: -Δu = f dans Ω = [0,1] x [0,1]
Problem: -Δu = f in Ω = [0,1] x [0,1]

où f(x,y) = 2π² sin(πx) sin(πy)
where f(x,y) = 2π² sin(πx) sin(πy)

Solution analytique: u(x,y) = sin(πx) sin(πy)
Analytical solution: u(x,y) = sin(πx) sin(πy)

Conditions aux limites: u = 0 sur tout le bord
Boundary conditions: u = 0 on all boundaries
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.mesh import Mesh
from src.solver import FESolver


def source_function(x, y):
    """Terme source / Source term"""
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def analytical_solution(x, y):
    """Solution analytique / Analytical solution"""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def main():
    print("=" * 60)
    print("Exemple 2: Équation de Poisson")
    print("Example 2: Poisson equation")
    print("=" * 60)
    
    # Paramètres du maillage / Mesh parameters
    lx, ly = 1.0, 1.0
    nx, ny = 20, 20  # Maillage plus fin / Finer mesh
    
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
    
    # Application du terme source / Apply source term
    print("Application du terme source...")
    print("Applying source term...")
    solver.apply_source_term(source_function)
    
    # Application des conditions aux limites (u = 0 sur tout le bord)
    # Apply boundary conditions (u = 0 on all boundaries)
    print("Application des conditions aux limites...")
    print("Applying boundary conditions...")
    
    # Trouver tous les noeuds sur le bord / Find all boundary nodes
    boundary_nodes = []
    for i, node in enumerate(mesh.nodes):
        if (abs(node.x) < 1e-10 or abs(node.x - lx) < 1e-10 or 
            abs(node.y) < 1e-10 or abs(node.y - ly) < 1e-10):
            boundary_nodes.append(i)
    
    solver.apply_dirichlet_bc(boundary_nodes, [0.0] * len(boundary_nodes))
    
    # Résolution / Solve
    print("\nRésolution du système...")
    print("Solving system...")
    solution = solver.solve()
    
    # Calcul de la solution analytique / Compute analytical solution
    coords = mesh.get_coordinates()
    analytical = np.array([analytical_solution(node.x, node.y) for node in mesh.nodes])
    
    # Calcul de l'erreur / Compute error
    error = np.abs(solution - analytical)
    relative_error = np.linalg.norm(error) / np.linalg.norm(analytical)
    
    print(f"\nRésultats / Results:")
    print(f"  Solution numérique: min={solution.min():.4f}, max={solution.max():.4f}")
    print(f"  Numerical solution: min={solution.min():.4f}, max={solution.max():.4f}")
    print(f"  Solution analytique: min={analytical.min():.4f}, max={analytical.max():.4f}")
    print(f"  Analytical solution: min={analytical.min():.4f}, max={analytical.max():.4f}")
    print(f"  Erreur relative L2: {relative_error:.6f}")
    print(f"  Relative L2 error: {relative_error:.6f}")
    
    # Export VTK
    output_file = "poisson_solution.vtk"
    print(f"\nExport de la solution vers {output_file}...")
    print(f"Exporting solution to {output_file}...")
    solver.export_to_vtk(output_file)
    
    # Visualisation / Visualization
    print("\nCréation de la visualisation...")
    print("Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    triangles = [[e.nodes[0].id, e.nodes[1].id, e.nodes[2].id] for e in mesh.elements]
    
    # Solution numérique / Numerical solution
    ax = axes[0, 0]
    ax.set_title("Solution numérique / Numerical solution")
    contour = ax.tricontourf(coords[:, 0], coords[:, 1], triangles, solution, levels=20, cmap='jet')
    plt.colorbar(contour, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    
    # Solution analytique / Analytical solution
    ax = axes[0, 1]
    ax.set_title("Solution analytique / Analytical solution")
    contour = ax.tricontourf(coords[:, 0], coords[:, 1], triangles, analytical, levels=20, cmap='jet')
    plt.colorbar(contour, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    
    # Erreur / Error
    ax = axes[1, 0]
    ax.set_title(f"Erreur absolue / Absolute error (L2={relative_error:.2e})")
    contour = ax.tricontourf(coords[:, 0], coords[:, 1], triangles, error, levels=20, cmap='hot')
    plt.colorbar(contour, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    
    # Comparaison sur une ligne / Comparison on a line
    ax = axes[1, 1]
    ax.set_title("Comparaison y=0.5 / Comparison at y=0.5")
    mid_nodes = [i for i, node in enumerate(mesh.nodes) if abs(node.y - 0.5) < 0.05]
    mid_nodes = sorted(mid_nodes, key=lambda i: mesh.nodes[i].x)
    x_vals = [mesh.nodes[i].x for i in mid_nodes]
    y_num = [solution[i] for i in mid_nodes]
    y_ana = [analytical[i] for i in mid_nodes]
    ax.plot(x_vals, y_num, 'bo-', label='Numérique / Numerical', markersize=4)
    ax.plot(x_vals, y_ana, 'r-', label='Analytique / Analytical', linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("poisson_solution.png", dpi=150)
    print("Visualisation sauvegardée: poisson_solution.png")
    print("Visualization saved: poisson_solution.png")
    
    print("\n" + "=" * 60)
    print("Terminé! / Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
