from vecs_util import *
import numpy as np

def fbin_read(fname):
    with open(fname, 'rb') as f:
        vectors = []
        size_prefix = numpy.fromfile(f, dtype=numpy.int32, count=1)
        len_prefix = numpy.fromfile(f, dtype=numpy.int32, count=1)
        print("Reading " + str(size_prefix[0]) + " vectors of length " + str(len_prefix[0]))
        for i in range(size_prefix[0]):
            if i % 10000 == 0:
                print("Reading vector " + str(i) + " of " + str(size_prefix[0]))
            vector = numpy.fromfile(f, dtype=numpy.float32, count=len_prefix[0])
            vectors.append(vector)
    return numpy.array(vectors)

def load_memmap_file_original(filename):
    data = np.memmap(filename, dtype=np.uint8, mode='r')
    header = data[:8].view(dtype=np.int32)
    count, dim = header[0], header[1]
    rest_data = data[8:].view(dtype=np.float32)
    array = rest_data.reshape(count, dim)
    return dim, count, array

if __name__ == "__main__":
    dim, count, array = load_memmap_file_original("/root/mount/tmp/msmar_query.bin")
    print("Loaded " + str(count) + " vectors of dimension " + str(dim))