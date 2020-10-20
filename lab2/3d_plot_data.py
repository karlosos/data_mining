import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_hiperplane(n_dimension):
    """ Generate random hiperplane that goes trough middle point """
    middle_point = np.array([0.5 for _ in range(n_dimension-1)])
    coeffs = np.random.rand(n_dimension-1)
    coeff_0 = 0.5 - np.sum(coeffs * middle_point)
    return coeff_0, coeffs
    
def generate_cube(n_dimension, n_samples, noise=False):
    x = np.random.rand(n_samples, n_dimension)
    return x

n_dimension = 3
samples = generate_cube(n_dimension, 2000)
b, coeffs = generate_hiperplane(n_dimension)

labels = np.sum(coeffs * samples[:, 0:-1], axis=1) + b > samples[:, -1]
colors = ['red' if l == 0 else 'green' for l in labels]

xs = np.linspace(0, 1, 100)
zs = np.linspace(0, 1, 100)

X, Z = np.meshgrid(xs, zs)
Y = coeffs[0] * X + coeffs[1] * Z + b

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Z, Y, color='blue')
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], color=colors)
plt.show()