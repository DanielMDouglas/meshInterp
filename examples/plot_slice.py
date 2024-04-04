import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from meshInterp import Field

nodeFile = 'nodes.csv'
valFile = 'B_field.csv'

B_field = Field(nodeFile, valFile)

# you can interpolate a field by passing in an (n x m) array
# where n is the number of points in the query

# input can be either a numpy array of points, or a normal list
pointA = np.array([[0, 0, 1]])
pointB = [[0, 0, 0]]

print ("field value at point", pointA, B_field(pointA))
print ("field value at point", pointB, B_field(pointB))
print ("evaluating at both points with a single array", B_field(np.concatenate([pointA, pointB])))

# let's make a grid of points to query and plot the resulting surface
NpointsX, NpointsY = 400, 800 # this is quite high resolution, lower values might be better
# NpointsX, NpointsY = 100, 200 # more manageable for testing
xSpace = np.linspace(-8.5, 0, NpointsX)

ySpace = np.linspace(-5, 12, NpointsY)

zSpace = np.array([0])

domain = np.array(np.meshgrid(xSpace, ySpace, zSpace)).reshape(3, NpointsX*NpointsY).T

print ("scanning a grid to generate a field map for the slice z = 0")
fieldValues = B_field(domain).reshape(NpointsY, NpointsX)

fig = plt.figure()
plt.pcolormesh(xSpace, ySpace, fieldValues,
               vmin = 0,
               vmax = 0.005,
               cmap = 'jet')
plt.xlabel(r'x (Horizontal Direction)')
plt.ylabel(r'y (Beam Direction)')

plt.colorbar()

plt.show()
# plt.savefig('slice.png')
