import numpy as np
import scipy as sp

# Reshape Operation

x = np.arange(12)
print(x)
print(x.shape)

x = x.reshape(1, 12)
print(x)
print(x.shape)

x = x.reshape(3, 4)
print(x)
print("================NEW===============")

# transpose implementation

A = np.arange(12).reshape(3, 4)
print(A)

print(A.T)
print("================NEW===============")

# Matrix Multiplication Implementation

A = np.arange(6).reshape(3, 2)
B = np.arange(6).reshape(2, 3)
print(A)
print(B)

print(np.matmul(A, B))
print("================NEW===============")

# Matrix Operations

A = np.arange(6).reshape(3, 2)
print(A)
print(A*A)
print(A+A)
print("================NEW===============")

# Matrix Inverse Implementation

A = np.arange(4).reshape(2, 2)
print(A)
print(np.linalg.inv(A))
print("================NEW===============")

# Eigenvalues and Eigenvectors

import matplotlib.pyplot as plt
from scipy.linalg import eig

A = [[1, 2],
     [2, 1]]
evals, evecs = eig(A)
evecs = evecs[:, 0 ], evecs[:, 1]

fig, ax = plt.subplots()
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position('zero')
    ax.grid(alpha=0.4)
    xmin, xmax = -3, 3
    ymin, ymax = -3, 3
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

for v in evecs:
    ax.annotate("", xy=v, xytext=(0, 0),
    arrowprops=dict(facecolor='blue',
    shrink = 0,
    alpha = 0.6,
    width = 0.5))

x = np.linspace(xmin, xmax, 3)
for v in evecs:
    y = v[1]/v[0]
    ax.plot(x, y*x, 'r-', lw =0.4)
    plt.show()

# Run the Code and the output will be the eigenvectors and eigenvalues of the matrix A, along with a plot showing the eigenvectors in the 2D space.

#get determinant of a matrix

E = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

print(np.linalg.det(E))

# Image compression

from pylab import *

img = plt.imread('lena.jpg')[:]
if img.ndim == 3:
    img = img[:, :, :3].mean(axis=2)

plt.savefig('./lena_gray')
plt.gray()
plt.figure(1)
plt.imshow(img)
plt.show()

m, n = img.shape
print(np.shape(img))
U,sigma,V = np.linalg.svd(img)
print(np.shape(sigma))
sigma = np.resize(sigma, (m, 1)) * np.eye(m, n)
k = 10
img1 = np.dot(U[:, 0:k], np.dot(sigma[0:k, 0:k], V[0:k, :]))
plt.figure(2)
plt.imshow(img1)
plt.show()

# Run the Code to see the Original image and the Compressed Image.