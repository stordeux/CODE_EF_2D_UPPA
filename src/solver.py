"""
Module du solveur pour éléments finis
Finite element solver module

Contient la classe principale pour résoudre des problèmes par éléments finis
Contains the main class to solve finite element problems
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import List, Tuple, Callable


class FESolver:
    """
    Solveur pour la méthode des éléments finis
    Finite element method solver
    """
    
    def __init__(self, mesh):
        """
        Initialise le solveur
        Initialize the solver
        
        Args:
            mesh: Maillage / Mesh
        """
        self.mesh = mesh
        self.n_dofs = mesh.n_nodes
        self.K = None  # Matrice de rigidité globale / Global stiffness matrix
        self.F = None  # Vecteur force global / Global force vector
        self.U = None  # Vecteur solution / Solution vector
    
    def assemble_system(self, element_type='triangle', k: float = 1.0):
        """
        Assemble le système global
        Assemble the global system
        
        Args:
            element_type: Type d'élément / Element type
            k: Coefficient de conductivité / Conductivity coefficient
        """
        from .element import TriangleElement
        
        # Initialisation des matrices / Initialize matrices
        self.K = lil_matrix((self.n_dofs, self.n_dofs))
        self.F = np.zeros(self.n_dofs)
        
        # Assemblage élément par élément / Element by element assembly
        for element in self.mesh.elements:
            # Créer l'élément fini / Create finite element
            fe_element = TriangleElement(element.nodes)
            
            # Calculer la matrice de rigidité élémentaire / Compute element stiffness matrix
            K_e = fe_element.compute_stiffness_matrix(k)
            
            # Numéros globaux des DDL / Global DOF numbers
            dofs = [node.id for node in element.nodes]
            
            # Assemblage dans la matrice globale / Assembly into global matrix
            for i, dof_i in enumerate(dofs):
                for j, dof_j in enumerate(dofs):
                    self.K[dof_i, dof_j] += K_e[i, j]
        
        # Conversion en format CSR pour la résolution / Convert to CSR format for solving
        self.K = self.K.tocsr()
    
    def apply_dirichlet_bc(self, node_indices: List[int], values: List[float]):
        """
        Applique des conditions aux limites de Dirichlet
        Apply Dirichlet boundary conditions
        
        Args:
            node_indices: Indices des noeuds / Node indices
            values: Valeurs imposées / Imposed values
        """
        for node_idx, value in zip(node_indices, values):
            # Méthode de pénalisation / Penalty method
            penalty = 1e10
            self.K[node_idx, node_idx] += penalty
            self.F[node_idx] += penalty * value
    
    def apply_neumann_bc(self, node_indices: List[int], flux_values: List[float]):
        """
        Applique des conditions aux limites de Neumann
        Apply Neumann boundary conditions
        
        Args:
            node_indices: Indices des noeuds / Node indices
            flux_values: Valeurs de flux / Flux values
        """
        for node_idx, flux in zip(node_indices, flux_values):
            self.F[node_idx] += flux
    
    def apply_source_term(self, source_function: Callable[[float, float], float]):
        """
        Applique un terme source
        Apply a source term
        
        Args:
            source_function: Fonction source f(x, y) / Source function f(x, y)
        """
        for element in self.mesh.elements:
            from .element import TriangleElement
            fe_element = TriangleElement(element.nodes)
            area = fe_element.compute_area()
            
            # Intégration au centre de l'élément (approximation)
            # Integration at element center (approximation)
            coords = fe_element.get_coordinates()
            x_center = np.mean(coords[:, 0])
            y_center = np.mean(coords[:, 1])
            
            f_value = source_function(x_center, y_center)
            
            # Contribution au vecteur force / Contribution to force vector
            dofs = [node.id for node in element.nodes]
            for dof in dofs:
                self.F[dof] += f_value * area / 3.0
    
    def solve(self):
        """
        Résout le système linéaire
        Solve the linear system
        """
        if self.K is None:
            raise RuntimeError("Le système n'a pas été assemblé / System has not been assembled")
        
        # Résolution / Solve
        self.U = spsolve(self.K, self.F)
        
        return self.U
    
    def get_solution_at_nodes(self) -> np.ndarray:
        """
        Retourne la solution aux noeuds
        Returns solution at nodes
        
        Returns:
            Array avec la solution / Array with solution
        """
        return self.U
    
    def export_to_vtk(self, filename: str):
        """
        Exporte la solution au format VTK
        Export solution to VTK format
        
        Args:
            filename: Nom du fichier / Filename
        """
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("2D FEM Solution\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Points
            f.write(f"POINTS {self.mesh.n_nodes} float\n")
            for node in self.mesh.nodes:
                f.write(f"{node.x} {node.y} 0.0\n")
            
            # Cellules (éléments)
            n_cells = self.mesh.n_elements
            f.write(f"\nCELLS {n_cells} {n_cells * 4}\n")
            for element in self.mesh.elements:
                node_ids = [node.id for node in element.nodes]
                f.write(f"3 {node_ids[0]} {node_ids[1]} {node_ids[2]}\n")
            
            # Types de cellules (5 = triangle)
            f.write(f"\nCELL_TYPES {n_cells}\n")
            for _ in range(n_cells):
                f.write("5\n")
            
            # Données
            if self.U is not None:
                f.write(f"\nPOINT_DATA {self.mesh.n_nodes}\n")
                f.write("SCALARS solution float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for value in self.U:
                    f.write(f"{value}\n")
