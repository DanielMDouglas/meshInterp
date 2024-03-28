import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from meshInterp import field

from tqdm import tqdm

nodeFile = 'nodes.csv'
valFile = 'B_field.csv'

B_field = field(nodeFile, valFile)

# you can interpolate a field by passing in an (n x m) array
# where n is the number of points in the query

# input can be either a numpy array of points, or a normal list
pointA = np.array([[0, 0, 1]])
pointB = [[0, 0, 0]]

print ("field value at point", pointA, B_field(pointA))
print ("field value at point", pointB, B_field(pointB))
print ("evaluating at both points with a single array", B_field(np.concatenate([pointA, pointB])))

# let's make a grid of points to query and plot the resulting surface
# NpointsX, NpointsY = 400, 800 # this is quite high resolution, lower values might be better
NpointsX, NpointsY = 100, 200 # more manageable for testing
xSpace = np.linspace(-8.5, 0, NpointsX)

ySpace = np.linspace(-5, 12, NpointsY)

z = 0

fieldValues = np.empty((NpointsX, NpointsY))
print ("scanning a grid to generate a field map for the slice z =", z)
for i, x_i in tqdm(enumerate(xSpace), total = NpointsX):
    for j, y_j in enumerate(ySpace):
        queryPoint = [[x_i, y_j, z]]
        fieldValues[i, j] = B_field(queryPoint)

fig = plt.figure()
plt.pcolormesh(xSpace, ySpace, fieldValues.T,
               vmin = 0,
               vmax = 0.005,
               cmap = 'jet')
plt.xlabel(r'x (Horizontal Direction)')
plt.ylabel(r'y (Beam Direction)')

plt.colorbar()

plt.show()
# plt.savefig('slice.png')
