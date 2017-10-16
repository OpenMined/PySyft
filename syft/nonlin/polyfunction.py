# coding=utf-8
"""
    Module polyfunction implements polynomial primitives for activation functions.
    Note:The Documentation in this file follows the NumPy Doc. Style;
         Hence, it is mandatory that future docs added here
         strictly follow the same, to maintain readability and consistency
         of the codebase.
    NumPy Documentation Style-
        http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
"""

import numpy as np

class PolyFunction:
    """
    Represents a polynomial and its derivative.
    """
    def __init__(self, coefs, derivative_coefs = None):
        """
        Builds a polynomial function with the given coefficients.
        Parameters
        ----------
        coefs : numpy array
            Polynomial coefficients in descending order.
        
        derivative_coefs : numpy array, optional
            Polynomial derivative coefficients in descending order.
            If None, derivative is calculated from the polynomial coefficients.
        """
        coefs = np.asarray(coefs)
        self.degree = coefs.size - 1
        
        self.coefs = coefs
        self.function = np.poly1d(self.coefs)
        
        if derivative_coefs is None:
            derivative_coefs = np.polyder(coefs)
            
        self.derivative_coefs = np.asarray(derivative_coefs)
        self.derivative = np.poly1d(self.derivative_coefs)
        
    def __call__(self, x):
        """
        Evaluates the polynomial in the given points.
        Parameters
        ----------
        x : numpy array
            The points to evaluate the polynomial at.
        Returns
        -------
        numpy array:
            Polynomial values in x.
        """
        return self.function(x)

    @staticmethod
    def fit_function(func, degree, precision, nodes):
        """
        Performs polynomial approximation of the given function.
        Parameters
        ----------
        func : function
            The function to be approximated.
        
        degree : int
            Polynomial degree used for interpolation.
            
        precision : int
            Numerical precision of coefficients.
           
        nodes : numpy array
            The interpolation points.
            
        Returns
        -------
        numpy array:
            Coefficients of the interpolated polynomial.
        """
        # interpolate polynomial of given max degree
        coefs = np.polyfit(nodes, func(nodes), degree)

        # reduce precision of interpolated coefficients
        coefs = np.asarray([int(x * 10**precision) / 10**precision for x in coefs])
        
        return coefs
        
    @classmethod
    def from_approximation(cls, f_real, f_derivative=None, degree=10, precision=10, min_range=-5, max_range=5, num=100, distribution = 'uniform'):
        """
        Builds a PolyFunction by approximating a given function.
        Parameters
        ----------
        f_real : function
            The function to be interpolated.
        
        f_derivative : function, optional
            The function derivative to be interpolated. 
            If None, it will be calculated from the interpolating polynomial.
            
        degree : int, optional
            Polynomial degree used for interpolation.
           
        precision : int, optional
            Numerical precision of coefficients.
            
        min_range : int, optional
            Lower bound of the interpolation nodes.
            
        max_range : int, optional
            Upper bound of the interpolation nodes.
            
        num : int, optional
            Number of interpolation nodes.
            
        distribution : string:function, optional
            Node distribution to be used.
            uniform, cheby and function accepting min/max range and number of nodes.
            
        Returns
        -------
        PolyFunction:
            Fitted polynomial function with derivative.
        """
        if distribution == 'uniform':
            nodes = np.linspace(min_range, max_range, num)
        elif distribution == 'cheby':
            nodes = np.cos( (np.pi/(2*num))*np.arange(1,2*num,2) ) * (max_range-min_range)/2
            nodes = nodes + (max_range+min_range)/2
        elif callable(distribution):
            nodes = distribution(min_range, max_range, num)
        else:
            raise ValueError('Invalid distribution type.')
        
        coefs = cls.fit_function(f_real, degree, precision, nodes)
        
        derivative_coefs = None
        if f_derivative is not None:
            derivative_coefs = cls.fit_function(f_derivative, degree, precision, nodes)
        
        poly = cls(coefs, derivative_coefs)
        poly.is_approx = True
        poly.precision = precision
        poly.min_range = min_range
        poly.max_range = max_range
        poly.distribution = distribution
        poly.nodes = nodes
        
        return poly
