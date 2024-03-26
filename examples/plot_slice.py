import numpy as np
import matplotlib.pyplot as plt

from meshInterp import field

nodeFile = 'nodes.csv'
valFile = 'B_field.csv'

B_field = field(nodeFile, valFile)

# you can interpolate a field by passing in an (n x m) array
# where n is the number of points in the query

# input can be either a numpy array of points, or a normal list
print (B_field(np.array([[0, 0, 1]])))
print (B_field([[0, 0, 0]]))
print (B_field([[0, 0, 0],
                [0, 0, 1]]))

# let's make a grid of points to query and plot the resulting surface
NpointsX, NpointsY = 400, 800 # this is quite high resolution, lower values might be better
# NpointsX, NpointsY = 100, 200 # more manageable for testing
xSpace = np.linspace(-8.5, 0, NpointsX)

ySpace = np.linspace(-5, 12, NpointsY)

z = 0

fieldValues = np.empty((NpointsX, NpointsY))
for i, x_i in enumerate(xSpace):
    for j, y_j in enumerate(ySpace):
        queryPoint = [[x_i, y_j, z]]
        fieldValues[i, j] = B_field(queryPoint)

fig = plt.figure()
plt.imshow(fieldValues,
           vmin = 0,
           vmax = 0.005,
           cmap = 'jet')

plt.colorbar()

plt.show()
