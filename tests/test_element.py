"""
Tests unitaires pour le module element
Unit tests for the element module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from src.mesh import Node
from src.element import TriangleElement


class TestTriangleElement(unittest.TestCase):
    """Tests pour la classe TriangleElement"""
    
    def setUp(self):
        """Création d'un triangle de référence"""
        self.nodes = [
            Node(0, 0.0, 0.0),
            Node(1, 1.0, 0.0),
            Node(2, 0.0, 1.0)
        ]
        self.element = TriangleElement(self.nodes)
    
    def test_element_creation(self):
        """Test création d'un élément triangulaire"""
        self.assertEqual(self.element.n_nodes, 3)
        self.assertEqual(self.element.n_dofs, 3)
    
    def test_get_coordinates(self):
        """Test récupération des coordonnées"""
        coords = self.element.get_coordinates()
        self.assertEqual(coords.shape, (3, 2))
        np.testing.assert_array_equal(coords[0], [0.0, 0.0])
        np.testing.assert_array_equal(coords[1], [1.0, 0.0])
        np.testing.assert_array_equal(coords[2], [0.0, 1.0])
    
    def test_compute_area(self):
        """Test calcul de l'aire"""
        area = self.element.compute_area()
        expected_area = 0.5  # Triangle rectangle de côté 1
        self.assertAlmostEqual(area, expected_area, places=10)
    
    def test_shape_functions(self):
        """Test fonctions de forme"""
        # Au noeud 1: (xi=1, eta=0)
        N = self.element.shape_functions(1.0, 0.0)
        np.testing.assert_array_almost_equal(N, [0.0, 1.0, 0.0])
        
        # Au centre: (xi=1/3, eta=1/3)
        N = self.element.shape_functions(1/3, 1/3)
        np.testing.assert_array_almost_equal(N, [1/3, 1/3, 1/3])
        
        # Somme des fonctions de forme = 1
        xi, eta = 0.2, 0.3
        N = self.element.shape_functions(xi, eta)
        self.assertAlmostEqual(np.sum(N), 1.0, places=10)
    
    def test_shape_derivatives(self):
        """Test dérivées des fonctions de forme"""
        dN = self.element.shape_derivatives()
        self.assertEqual(dN.shape, (2, 3))
        
        # Les dérivées sont constantes pour un triangle P1
        expected_dN = np.array([[-1, 1, 0], [-1, 0, 1]])
        np.testing.assert_array_equal(dN, expected_dN)
    
    def test_compute_stiffness_matrix(self):
        """Test calcul de la matrice de rigidité"""
        K = self.element.compute_stiffness_matrix(k=1.0)
        
        # Vérifications de base
        self.assertEqual(K.shape, (3, 3))
        
        # La matrice doit être symétrique
        np.testing.assert_array_almost_equal(K, K.T)
        
        # La somme de chaque ligne doit être proche de 0 (conservation)
        row_sums = np.sum(K, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.zeros(3), decimal=10)


if __name__ == '__main__':
    unittest.main()
