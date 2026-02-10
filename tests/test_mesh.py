"""
Tests unitaires pour le module mesh
Unit tests for the mesh module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from src.mesh import Mesh, Node, Element


class TestNode(unittest.TestCase):
    """Tests pour la classe Node"""
    
    def test_node_creation(self):
        """Test création d'un noeud"""
        node = Node(0, 1.0, 2.0)
        self.assertEqual(node.id, 0)
        self.assertEqual(node.x, 1.0)
        self.assertEqual(node.y, 2.0)
    
    def test_node_repr(self):
        """Test représentation string"""
        node = Node(5, 3.14, 2.71)
        self.assertIn("Node", str(node))
        self.assertIn("5", str(node))


class TestElement(unittest.TestCase):
    """Tests pour la classe Element"""
    
    def test_element_creation(self):
        """Test création d'un élément"""
        nodes = [Node(0, 0, 0), Node(1, 1, 0), Node(2, 0, 1)]
        element = Element(0, nodes)
        self.assertEqual(element.id, 0)
        self.assertEqual(element.n_nodes, 3)
        self.assertEqual(len(element.nodes), 3)


class TestMesh(unittest.TestCase):
    """Tests pour la classe Mesh"""
    
    def test_mesh_creation(self):
        """Test création d'un maillage vide"""
        mesh = Mesh()
        self.assertEqual(mesh.n_nodes, 0)
        self.assertEqual(mesh.n_elements, 0)
    
    def test_add_node(self):
        """Test ajout de noeuds"""
        mesh = Mesh()
        node1 = mesh.add_node(0.0, 0.0)
        node2 = mesh.add_node(1.0, 0.0)
        
        self.assertEqual(mesh.n_nodes, 2)
        self.assertEqual(node1.id, 0)
        self.assertEqual(node2.id, 1)
    
    def test_add_element(self):
        """Test ajout d'éléments"""
        mesh = Mesh()
        mesh.add_node(0.0, 0.0)
        mesh.add_node(1.0, 0.0)
        mesh.add_node(0.0, 1.0)
        
        element = mesh.add_element([0, 1, 2])
        
        self.assertEqual(mesh.n_elements, 1)
        self.assertEqual(element.id, 0)
        self.assertEqual(element.n_nodes, 3)
    
    def test_rectangular_mesh(self):
        """Test génération d'un maillage rectangulaire"""
        mesh = Mesh()
        mesh.generate_rectangular_mesh(1.0, 1.0, 2, 2)
        
        # Vérifier le nombre de noeuds: (nx+1)*(ny+1) = 3*3 = 9
        self.assertEqual(mesh.n_nodes, 9)
        
        # Vérifier le nombre d'éléments: 2*nx*ny = 2*2*2 = 8
        self.assertEqual(mesh.n_elements, 8)
    
    def test_get_coordinates(self):
        """Test récupération des coordonnées"""
        mesh = Mesh()
        mesh.add_node(0.0, 0.0)
        mesh.add_node(1.0, 2.0)
        mesh.add_node(3.0, 4.0)
        
        coords = mesh.get_coordinates()
        
        self.assertEqual(coords.shape, (3, 2))
        np.testing.assert_array_equal(coords[0], [0.0, 0.0])
        np.testing.assert_array_equal(coords[1], [1.0, 2.0])
        np.testing.assert_array_equal(coords[2], [3.0, 4.0])


if __name__ == '__main__':
    unittest.main()
