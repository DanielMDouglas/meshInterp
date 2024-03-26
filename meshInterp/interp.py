import numpy as np

def barycentric_interp(nodePos, nodeVal, qp):
    """
    Returns the interpolated value of a field at point qp
    using the barycentric method
    """

    # get the barycentric coordinates of the query point
    rot = np.matrix([np.concatenate([[1], nodeLoc]) for nodeLoc in nodePos]).T
    interpVec = np.concatenate([[[1]], qp], axis = 1).flatten()

    baryCoords = np.array(np.dot(rot.I, interpVec)).flatten()
                    
    # return the interpolated value
    result = np.dot(baryCoords, nodeVal)

    return result

def inv_dist_interp(nodePos, nodeVal, qp):
    """
    Returns the interpolated value of a field at point qp
    using the inverse distance weighting method
    """
    
    # get the distance between each node point
    # and the query point
    d = np.sqrt(np.sum(np.power(qp - nodePos, 2), axis = 1))
    w = 1./d

    return np.dot(w, nodeVal)/np.sum(w)

def mean_interp(nodePos, nodeVal, qp):
    """
    The dumbest interpolation you can imagine.
    Just give the mean of the input neighbors
    """
    return np.mean(nodeVal)

def closest_point_interp(nodePos, nodeVal, qp):
    """
    The second dumbest interpolation you can imagine.
    Return the value of the closest input neighbor
    """
    d = np.sqrt(np.sum(np.power(qp - nodePos, 2), axis = 1))
    return nodeVal[d == np.min(d)][0]

def constant_interp(nodePos, nodeVal, qp, C = -0.1):
    """
    okay, THIS is the dumbest interpolation possible
    Just return a constant value
    """
    return C
