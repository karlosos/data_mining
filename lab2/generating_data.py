import numpy as np
import matplotlib.pyplot as plt
from numpy import sin


def generate_hiperplane(n_dimension):
    """ Generate random hiperplane that goes trough middle point """
    middle_point = np.array([0.5 for _ in range(n_dimension-1)])
    coeffs = np.random.rand(n_dimension-1)
    coeff_0 = 0.5 - np.sum(coeffs * middle_point)
    return coeff_0, coeffs
    

def generate_cube(n_dimension, n_samples, noise=False):
    x = np.random.rand(n_samples, n_dimension)
    return x


def generate_data(n_dimension=2, n_samples=100, noise=False):
    samples = generate_cube(n_dimension, n_samples, noise)
    b, coeffs = generate_hiperplane(n_dimension)

    labels = np.sum(coeffs * samples[:, 0:-1], axis=1) + b > samples[:, -1]
    return samples, labels, (b, coeffs)

def generate_checkerboard(n_size, n_samples, noise=False):
    """Generate checkerboard of size n_size x n_size"""
    x = np.random.uniform(-(n_size//2)*np.pi, (n_size//2)*np.pi, size=(n_samples, 2))
    mask = np.logical_or(np.logical_and(sin(x[:, 0]) > 0.0, sin(x[:, 1]) > 0.0),
                         np.logical_and(sin(x[:, 0]) < 0.0, sin(x[:, 1]) < 0.0))
    # normalization
    x = (x-np.min(x))/(np.max(x)-np.min(x))
    y = np.eye(2)[1*mask]
    y = y[:, 0]  # the second column is inverse, we don't need that
    
    return x, y
