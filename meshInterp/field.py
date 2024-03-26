import numpy as np
import scipy as sp

from .interp import *

class field:
    def __init__(self, *infiles, format = 'CSV', OOB_value = 0):
        self.infiles = infiles
        self.OOB_value = OOB_value
        
        if format == 'CSV':
            # load node array
            self.nodes = np.loadtxt(self.infiles[0],
                                    usecols = [0, 1, 2, 3],
                                    skiprows = 1,
                                    delimiter = ',',
                                    )
            # sort by the node index to make sure
            # the nodes are aligned with the values 
            self.nodes = self.nodes[self.nodes[:,0].argsort()]
            print ("nodes", self.nodes.shape)
            print ("ext",
                   np.min(self.nodes[:,0]),
                   np.max(self.nodes[:,0]),
                   np.min(self.nodes[:,1]),
                   np.max(self.nodes[:,1]),
                   np.min(self.nodes[:,2]),
                   np.max(self.nodes[:,2]),
                   np.min(self.nodes[:,3]),
                   np.max(self.nodes[:,3]),
                   )
            # load value array
            # assumes one (scalar) value per node for now
            self.vals = np.loadtxt(self.infiles[1],
                                   usecols = [0, 1],
                                   skiprows = 1,
                                   delimiter = ',',
                                   )
            # sort by the node index
            self.vals = self.vals[self.vals[:,0].argsort()]

            # build a KD tree for rough node finding
            self.KDTree = sp.spatial.KDTree(self.nodes[:,1:])

            def interp(queryPoint, N_local_neighbors = 10):
                # get a somewhat large number of neighbors
                query = self.KDTree.query(queryPoint, N_local_neighbors)

                # get the position and fields for these neighbors
                kNNpoints = self.nodes[query[1]][0]
                kNNvals = self.vals[query[1]][0]
                # make sure that the indices match up!
                # this should be true because we sorted the input arrays
                assert (np.all(kNNpoints[:,0] == kNNvals[:,0]))

                # find the delaunay graph in this neighborhood
                # this is to enforce interpolation from nodes which
                # "enclose" the query point
                try:
                    tess = sp.spatial.Delaunay(kNNpoints[:,1:])
                except sp.spatial._qhull.QhullError:
                    return self.OOB_value
                    
                simplIndex = tess.find_simplex(queryPoint)
                nodeIndices = tess.simplices[simplIndex]

                # the 4 mesh nodes which form a simplex around the query 
                simplexNodes = kNNpoints[nodeIndices][0,:,1:]
                simplexVals = kNNvals[nodeIndices][0,:,1:]

                try:
                    bary_result = barycentric_interp(simplexNodes, simplexVals, queryPoint)

                    # if the matrix inversion fails, just use the inverse distance interpolation
                    if bary_result > np.max(simplexVals) or bary_result < np.min(simplexVals):
                        return inv_dist_interp(simplexNodes, simplexVals, queryPoint)

                    return bary_result

                except np.linalg.LinAlgError: # is the matrix singular?
                    return inv_dist_interp(simplexNodes, simplexVals, queryPoint)

            self.interp = interp
            
    def __call__(self, queryPoint, **kwargs):
        # input must be an array of shape (n x m)
        # or (1 x m).  If n > 1, just interpolate serially
        qp = np.array(queryPoint)
        assert len(qp.shape) == 2 and qp.shape[1] == 3

        if qp.shape[0] > 1:
            result = []
            for thisQuery in qp:
                result.append(self.interp([thisQuery], **kwargs))
            
            return np.array(result)

        else:
            return self.interp(qp)
