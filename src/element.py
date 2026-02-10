"""
Module des éléments finis
Finite elements module

Contient les classes pour différents types d'éléments finis
Contains classes for different types of finite elements
"""

import numpy as np
from typing import Tuple


class TriangleElement:
    """
    Élément triangulaire à 3 noeuds (P1)
    3-node triangular element (P1)
    """
    
    def __init__(self, nodes):
        """
        Initialise un élément triangulaire
        Initialize a triangular element
        
        Args:
            nodes: Liste des 3 noeuds / List of 3 nodes
        """
        self.nodes = nodes
        self.n_nodes = 3
        self.n_dofs = 3  # 1 DDL par noeud / 1 DOF per node
    
    def get_coordinates(self) -> np.ndarray:
        """
        Retourne les coordonnées des noeuds
        Returns node coordinates
        
        Returns:
            Array (3, 2) avec les coordonnées / Array (3, 2) with coordinates
        """
        coords = np.zeros((3, 2))
        for i, node in enumerate(self.nodes):
            coords[i, 0] = node.x
            coords[i, 1] = node.y
        return coords
    
    def compute_area(self) -> float:
        """
        Calcule l'aire de l'élément
        Compute element area
        
        Returns:
            Aire de l'élément / Element area
        """
        coords = self.get_coordinates()
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Formule de l'aire d'un triangle / Triangle area formula
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        return area
    
    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """
        Calcule les fonctions de forme en coordonnées naturelles
        Compute shape functions in natural coordinates
        
        Args:
            xi: Première coordonnée naturelle / First natural coordinate
            eta: Seconde coordonnée naturelle / Second natural coordinate
            
        Returns:
            Valeurs des fonctions de forme / Shape function values
        """
        N = np.array([1 - xi - eta, xi, eta])
        return N
    
    def shape_derivatives(self) -> np.ndarray:
        """
        Calcule les dérivées des fonctions de forme
        Compute shape function derivatives
        
        Returns:
            Matrice des dérivées (2, 3) / Derivative matrix (2, 3)
        """
        # Dérivées par rapport à xi et eta / Derivatives with respect to xi and eta
        dN = np.array([
            [-1, 1, 0],   # dN/dxi
            [-1, 0, 1]    # dN/deta
        ])
        return dN
    
    def compute_B_matrix(self) -> np.ndarray:
        """
        Calcule la matrice B (dérivées des fonctions de forme dans l'espace physique)
        Compute B matrix (shape function derivatives in physical space)
        
        Returns:
            Matrice B (2, 3) / B matrix (2, 3)
        """
        coords = self.get_coordinates()
        dN = self.shape_derivatives()
        
        # Calcul de la matrice Jacobienne / Compute Jacobian matrix
        J = dN @ coords
        
        # Inverse de la Jacobienne / Inverse of Jacobian
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        
        # B = invJ * dN
        B = invJ @ dN
        
        return B
    
    def compute_stiffness_matrix(self, k: float = 1.0) -> np.ndarray:
        """
        Calcule la matrice de rigidité élémentaire (équation de Laplace)
        Compute element stiffness matrix (Laplace equation)
        
        Args:
            k: Coefficient de conductivité / Conductivity coefficient
            
        Returns:
            Matrice de rigidité (3, 3) / Stiffness matrix (3, 3)
        """
        B = self.compute_B_matrix()
        area = self.compute_area()
        
        # K_e = area * B^T * B * k
        K_e = area * k * (B.T @ B)
        
        return K_e


class QuadElement:
    """
    Élément quadrilatéral à 4 noeuds (Q1)
    4-node quadrilateral element (Q1)
    """
    
    def __init__(self, nodes):
        """
        Initialise un élément quadrilatéral
        Initialize a quadrilateral element
        
        Args:
            nodes: Liste des 4 noeuds / List of 4 nodes
        """
        self.nodes = nodes
        self.n_nodes = 4
        self.n_dofs = 4  # 1 DDL par noeud / 1 DOF per node
    
    def get_coordinates(self) -> np.ndarray:
        """
        Retourne les coordonnées des noeuds
        Returns node coordinates
        
        Returns:
            Array (4, 2) avec les coordonnées / Array (4, 2) with coordinates
        """
        coords = np.zeros((4, 2))
        for i, node in enumerate(self.nodes):
            coords[i, 0] = node.x
            coords[i, 1] = node.y
        return coords
    
    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """
        Calcule les fonctions de forme en coordonnées naturelles [-1, 1]
        Compute shape functions in natural coordinates [-1, 1]
        
        Args:
            xi: Première coordonnée naturelle / First natural coordinate
            eta: Seconde coordonnée naturelle / Second natural coordinate
            
        Returns:
            Valeurs des fonctions de forme / Shape function values
        """
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta)
        ])
        return N
    
    def shape_derivatives(self, xi: float, eta: float) -> np.ndarray:
        """
        Calcule les dérivées des fonctions de forme
        Compute shape function derivatives
        
        Args:
            xi: Première coordonnée naturelle / First natural coordinate
            eta: Seconde coordonnée naturelle / Second natural coordinate
            
        Returns:
            Matrice des dérivées (2, 4) / Derivative matrix (2, 4)
        """
        dN = 0.25 * np.array([
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],  # dN/dxi
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]       # dN/deta
        ])
        return dN
