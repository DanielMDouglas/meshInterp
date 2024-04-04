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

            # now that the arrays are aligned, let's ditch the index
            self.nodes = self.nodes[:,1:]
            self.vals = self.vals[:,1:]

            self.field_dim = self.vals.shape[1]
            
            # build a KD tree for rough node finding
            self.kdtree = KDTree(self.nodes)

            def interp_field(query_array, n_local_neighbors = 8):
                try:
                    assert type(query_array) == np.ndarray
                except AssertionError:
                    query_array = np.array(query_array)
                
                n_query_points = query_array.shape[0]

                result = np.empty((self.field_dim,
                                   n_query_points))
                
                # get a somewhat large number of neighbors
                neighbor_query = self.kdtree.query(query_array, n_local_neighbors)

                # get the position and fields for these neighbors
                neighbor_points = np.swapaxes(self.nodes[neighbor_query[1][:]], 0, 1)
                neighbor_vals = np.swapaxes(self.vals[neighbor_query[1][:]], 0, -1)
                
                # a simple inverse-distance weighting interpolation
                # this is very simple to write in a parallel way
                # as opposed to a more sophisticated barycentric interpolation
                d = np.sqrt(np.sum(np.power(query_array - neighbor_points, 2), axis = -1))
                w = np.power(d, -1)

                interp_val = np.sum(w*neighbor_vals, axis = 1)/np.sum(w, axis = 0)

                return interp_val
                
            self.interp_field = interp_field

    def __call__(self, query_point, **kwargs):
        
        return self.interp_field(query_point)

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
