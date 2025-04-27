import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# Generate some sample data
np.random.seed(0)
# Red points in Quadrant 1
X_red = np.random.randn(10, 2) * 0.4 + [2, 2]
# Green points in Quadrant 3
X_green = np.random.randn(10, 2) * 0.4 + [-2, -2]

X = np.vstack((X_red, X_green))
y = np.array([0]*10 + [1]*10)  # 0: red, 1: green

# Fit the SVM model
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Create a mesh to plot the decision boundary
xx, yy = np.meshgrid(np.linspace(-4, 4, 500), np.linspace(-4, 4, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot 1: just the points
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_red[:, 0], X_red[:, 1], c='red', label='Class 0 (Red)')
plt.scatter(X_green[:, 0], X_green[:, 1], c='green', label='Class 1 (Green)')
plt.title('Red and Green Points')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)

# Plot 2: with decision boundary
plt.subplot(1, 2, 2)
plt.scatter(X_red[:, 0], X_red[:, 1], c='red')
plt.scatter(X_green[:, 0], X_green[:, 1], c='green')
plt.contour(xx, yy, Z, levels=[0], colors='blue', linewidths=2)
plt.title('SVM Decision Boundary')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.tight_layout()
plt.savefig('../output/visualizations/SVM_example.png')
plt.close()

# Plot 3: electrode visualizations

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../assets/BCIsensor_xy.csv", header=None)
x = df.iloc[:, 0]
y = df.iloc[:, 1]
plt.plot(x, y)
plt.scatter(x, y, color="blue", s=20, zorder=3)
for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(i + 1), fontsize=8, ha='right', va='bottom')
plt.axis("equal")
plt.axis("off")
plt.savefig('../output/visualizations/electrodes.png')
plt.close()