# -*- coding: utf-8 -*-
"""
Created on Tue May 30 23:37:11 2017

@author: Arindam
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

# simple sine and cosine plot
t = np.arange(-4, 4, 0.01)
s1 = np.sin(t)
s2 = np.cos(t)
plt.plot(t, s1, 'bo')
plt.plot(t, s2, 'r+')
plt.xlabel('time')
plt.ylabel('Value')
plt.title('Simple Sine Cosine Graph')
plt.grid(True)
plt.savefig("demo1_1.png")
plt.show()

# plotting iris data points in 2D scatterplot
iris = datasets.load_iris()
x = iris.data[:, :2]  # we only take the first two features.
y = iris.target
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.PiYG)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris data Sepal width vs Sepal length')
plt.grid(True)
plt.savefig("demo1_2.png")
plt.show()

# plotting iris data points in 3D scatterplot
x = iris.data[:, :3]  # we only take the first three features.
y = iris.target
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, marker='o')
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_zlabel('Petal length')
plt.grid(True)
plt.savefig("demo1_3.png")
plt.show()
