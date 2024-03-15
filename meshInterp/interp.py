import numpy as np
import scipy as sp

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
            # load value array
            # assumes one (scalar) value per node for now
            self.vals = np.loadtxt(self.infiles[1],
                                   usecols = [0, 1],
                                   skiprows = 1,
                                   delimiter = ',',
                                   )

            # build a KD tree for rough node finding
            self.KDTree = sp.spatial.KDTree(self.nodes[:,1:])

            def interp(queryPoint, N_local_neighbors = 10):
                # get a somewhat large number of neighbors
                query = self.KDTree.query(queryPoint, N_local_neighbors)

                # get the position and fields for these neighbors
                kNNpoints = self.nodes[query[1]]
                kNNvals = self.vals[query[1]]

                # find the delaunay graph in this neighborhood
                # this is to enforce interpolation from nodes which
                # "enclose" the query point
                try:
                    tess = sp.spatial.Delaunay(kNNpoints[0,:,1:])
                except sp.spatial._qhull.QhullError:
                    return self.OOB_value
                    
                simplIndex = tess.find_simplex(queryPoint)
                nodeIndices = tess.simplices[simplIndex]

                # the 4 mesh nodes which form a simplex around the query 
                simplexNodes = kNNpoints[0,nodeIndices][0,:,1:]
                simplexVals = kNNvals[0,nodeIndices][0,:,1:]

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
