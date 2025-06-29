from vecs_util import *

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

if __name__ == "__main__":
    base = fbin_read("/root/mount/dataset/internet_search/internet_search_train.fbin")
    print("Base shape: " + str(base.shape))