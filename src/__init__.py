"""
Module principal pour le code d'éléments finis 2D
Main module for 2D finite element code

Ce module contient les classes de base pour la méthode des éléments finis en 2D
This module contains the base classes for the finite element method in 2D
"""

__version__ = "0.1.0"
__author__ = "stordeux"

from .mesh import Mesh, Node, Element
from .element import TriangleElement, QuadElement
from .solver import FESolver

__all__ = ['Mesh', 'Node', 'Element', 'TriangleElement', 'QuadElement', 'FESolver']
