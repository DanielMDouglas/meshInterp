import numpy as np

def is_within_range(ref_vals, interp_val):
    """
    Sanity check function to ensure that the interpolated
    value lies somewhere within the range spanned by the
    input values
    """

    if interp_val > np.max(ref_vals):
        return False
    if interp_val < np.min(ref_vals):
        return False
    return True

def barycentric_interp(node_pos, node_val, qp):
    """
    Returns the interpolated value of a field at point qp
    using the barycentric method
    """

    # get the barycentric coordinates of the query point
    rot = np.matrix(np.concatenate([np.ones((4, 1)), node_pos], axis = -1)).T
    interp_vec = np.concatenate([[1], qp]).flatten()

    bary_coords = np.array(np.dot(rot.I, interp_vec)).flatten()

    # return the interpolated value
    result = np.dot(bary_coords, node_val)

    return result

def inv_dist_interp(node_pos, node_val, qp):
    """
    Returns the interpolated value of a field at point qp
    using the inverse distance weighting method
    """

    # get the distance between each node point
    # and the query point
    d = np.sqrt(np.sum(np.power(qp - node_pos, 2), axis = 1))
    w = 1./d

    return np.dot(w, node_val)/np.sum(w)

def mean_interp(node_pos, node_val, qp):
    """
    The dumbest interpolation you can imagine.
    Just give the mean of the input neighbors
    """
    return np.mean(node_val)

def closest_point_interp(node_pos, node_val, qp):
    """
    The second dumbest interpolation you can imagine.
    Return the value of the closest input neighbor
    """
    d = np.sqrt(np.sum(np.power(qp - node_pos, 2), axis = 1))
    return node_val[d == np.min(d)][0]

def constant_interp(node_pos, node_val, qp, c = -0.1):
    """
    okay, THIS is the dumbest interpolation possible
    Just return a constant value
    """
    return c
