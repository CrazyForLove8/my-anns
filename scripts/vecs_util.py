import os
import numpy
from sklearn.model_selection import train_test_split


def ivecs_read(fname):
    a = numpy.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    with open(fname, 'rb') as f:
        vectors = []
        while True:
            len_prefix = numpy.fromfile(f, dtype=numpy.int32, count=1)
            if len_prefix.size == 0:
                break
            d = len_prefix[0]
            vector = numpy.fromfile(f, dtype=numpy.float32, count=d)
            vectors.append(vector)
    return numpy.array(vectors)


def write_fvecs(X, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'wb') as f:
        for vec in X:
            numpy.array([len(vec)], dtype=numpy.int32).tofile(f)
            vec.astype(numpy.float32).tofile(f)
            
def write_ivecs(X : numpy.array, output_path : str):
    with open(output_path, 'wb') as f:
        for vec in X:
            numpy.array([len(vec)], dtype=numpy.int32).tofile(f)
            vec.astype(numpy.int32).tofile(f)
        
def calculate_gt(X_train, X_test, n_neighbors, metric='euclidean'):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric=metric).fit(X_train)
    distances, indices = nbrs.kneighbors(X_test)
    return indices, distances