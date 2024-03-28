"""
This module defines a scalar field object with a __call__ method
"""

import numpy as np
from scipy.spatial import KDTree, Delaunay, QhullError

from . import interp

class Field:
    """
    Scalar Field defined over a subset of R^N

    Fields are defined in unstructured grids and
    an appropriate interpolation method is defined
    based on the input format
    
    Currently, this object can be initialized from a set of
    CSV files containing node locations and node values
    More reader methods to come in the future!
    """
    def __init__(self, *infiles, input_format = 'CSV', out_of_bounds_value = 0):
        self.infiles = infiles
        self.out_of_bounds_value = out_of_bounds_value

        if input_format == 'CSV':
            # load node array
            self.nodes = np.loadtxt(self.infiles[0],
                                    usecols = [0, 1, 2, 3],
                                    skiprows = 1,
                                    delimiter = ',',
                                    )
            # sort by the node index to make sure
            # the nodes are aligned with the values
            self.nodes = self.nodes[self.nodes[:,0].argsort()]

            # load value array
            # assumes one (scalar) value per node for now
            self.vals = np.loadtxt(self.infiles[1],
                                   usecols = [0, 1],
                                   skiprows = 1,
                                   delimiter = ',',
                                   )
            # sort by the node index
            self.vals = self.vals[self.vals[:,0].argsort()]

            # make sure that the indices match up!
            # this should be true because we sorted the input arrays
            assert np.all(self.nodes[:,0] == self.vals[:,0])

            # build a KD tree for rough node finding
            self.kdtree = KDTree(self.nodes[:,1:])

            def interp_field(query_point, n_local_neighbors = 10):
                # get a somewhat large number of neighbors
                query = self.kdtree.query(query_point, n_local_neighbors)
                # get the position and fields for these neighbors
                neighbor_points = self.nodes[query[1]][0]
                neighbor_vals = self.vals[query[1]][0]

                # find the delaunay graph in this neighborhood
                # this is to enforce interpolation from nodes which
                # "enclose" the query point
                try:
                    tess = Delaunay(neighbor_points[:,1:])

                except QhullError:
                    return self.out_of_bounds_value

                simpl_index = tess.find_simplex(query_point)
                node_indices = tess.simplices[simpl_index]

                # the 4 mesh nodes which form a simplex around the query
                simplex_nodes = neighbor_points[node_indices][0,:,1:]
                simplex_vals = neighbor_vals[node_indices][0,:,1:]

                try:
                    bary_result = interp.barycentric_interp(simplex_nodes,
                                                            simplex_vals,
                                                            query_point)

                    # if the matrix inversion fails, just use the inverse distance interpolation
                    if bary_result > np.max(simplex_vals) or bary_result < np.min(simplex_vals):
                        return interp.inv_dist_interp(simplex_nodes,
                                                      simplex_vals,
                                                      query_point)

                    return bary_result

                except np.linalg.LinAlgError: # is the matrix singular?
                    return interp.inv_dist_interp(simplex_nodes, simplex_vals, query_point)

            self.interp_field = interp_field

    def __call__(self, query_point, **kwargs):
        # input must be an array of shape (n x m)
        # or (1 x m).  If n > 1, just interpolate serially
        qp = np.array(query_point)
        assert len(qp.shape) == 2 and qp.shape[1] == 3

        if qp.shape[0] > 1:
            result = []
            for this_query in qp:
                result.append(self.interp_field([this_query], **kwargs))

            return np.array(result)

        return self.interp_field(qp)

    def get_nodes(self):
        """
        Getter method for node positions
        """
        return self.nodes

    def get_vals(self):
        """
        Getter method for node values
        """
        return self.vals
