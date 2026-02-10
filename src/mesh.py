"""
Module de maillage pour éléments finis 2D
Mesh module for 2D finite elements

Contient les classes pour gérer les maillages, noeuds et éléments
Contains classes to manage meshes, nodes and elements
"""

import numpy as np
from typing import List, Tuple


class Node:
    """
    Classe représentant un noeud du maillage
    Class representing a mesh node
    """
    
    def __init__(self, node_id: int, x: float, y: float):
        """
        Initialise un noeud
        Initialize a node
        
        Args:
            node_id: Identifiant du noeud / Node identifier
            x: Coordonnée x / x coordinate
            y: Coordonnée y / y coordinate
        """
        self.id = node_id
        self.x = x
        self.y = y
        self.dofs = []  # Degrés de liberté / Degrees of freedom
    
    def __repr__(self):
        return f"Node({self.id}, x={self.x:.3f}, y={self.y:.3f})"


class Element:
    """
    Classe de base pour un élément fini
    Base class for a finite element
    """
    
    def __init__(self, element_id: int, nodes: List[Node]):
        """
        Initialise un élément
        Initialize an element
        
        Args:
            element_id: Identifiant de l'élément / Element identifier
            nodes: Liste des noeuds de l'élément / List of element nodes
        """
        self.id = element_id
        self.nodes = nodes
        self.n_nodes = len(nodes)
    
    def __repr__(self):
        node_ids = [n.id for n in self.nodes]
        return f"Element({self.id}, nodes={node_ids})"


class Mesh:
    """
    Classe représentant un maillage 2D
    Class representing a 2D mesh
    """
    
    def __init__(self):
        """Initialise un maillage vide / Initialize an empty mesh"""
        self.nodes = []
        self.elements = []
        self.n_nodes = 0
        self.n_elements = 0
    
    def add_node(self, x: float, y: float) -> Node:
        """
        Ajoute un noeud au maillage
        Add a node to the mesh
        
        Args:
            x: Coordonnée x / x coordinate
            y: Coordonnée y / y coordinate
            
        Returns:
            Le noeud créé / The created node
        """
        node = Node(self.n_nodes, x, y)
        self.nodes.append(node)
        self.n_nodes += 1
        return node
    
    def add_element(self, node_indices: List[int]) -> Element:
        """
        Ajoute un élément au maillage
        Add an element to the mesh
        
        Args:
            node_indices: Indices des noeuds de l'élément / Node indices for the element
            
        Returns:
            L'élément créé / The created element
        """
        nodes = [self.nodes[i] for i in node_indices]
        element = Element(self.n_elements, nodes)
        self.elements.append(element)
        self.n_elements += 1
        return element
    
    def generate_rectangular_mesh(self, lx: float, ly: float, nx: int, ny: int):
        """
        Génère un maillage rectangulaire
        Generate a rectangular mesh
        
        Args:
            lx: Longueur en x / Length in x
            ly: Longueur en y / Length in y
            nx: Nombre d'éléments en x / Number of elements in x
            ny: Nombre d'éléments en y / Number of elements in y
        """
        # Création des noeuds / Create nodes
        dx = lx / nx
        dy = ly / ny
        
        for j in range(ny + 1):
            for i in range(nx + 1):
                self.add_node(i * dx, j * dy)
        
        # Création des éléments (triangles) / Create elements (triangles)
        for j in range(ny):
            for i in range(nx):
                n0 = j * (nx + 1) + i
                n1 = n0 + 1
                n2 = n0 + (nx + 1)
                n3 = n2 + 1
                
                # Deux triangles par rectangle / Two triangles per rectangle
                self.add_element([n0, n1, n2])
                self.add_element([n1, n3, n2])
    
    def get_coordinates(self) -> np.ndarray:
        """
        Retourne les coordonnées de tous les noeuds
        Returns coordinates of all nodes
        
        Returns:
            Array numpy (n_nodes, 2) / Numpy array (n_nodes, 2)
        """
        coords = np.zeros((self.n_nodes, 2))
        for i, node in enumerate(self.nodes):
            coords[i, 0] = node.x
            coords[i, 1] = node.y
        return coords
    
    def __repr__(self):
        return f"Mesh(nodes={self.n_nodes}, elements={self.n_elements})"
