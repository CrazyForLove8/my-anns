import os
import numpy
from sklearn.model_selection import train_test_split
from vecs_util import *

d = 100
fn = "/root/datasets/glove/twitter/glove.twitter.27B.%dd.txt" % d
base = "/root/datasets/glove/twitter/"

with open(fn) as z:
    print("preparing")
    X = []
    for line in z:
        values = line.strip().split()[1:]
        if len(values) == d:
            v = [float(x) for x in values]
            X.append(numpy.array(v))
    X = numpy.array(X)
    X_train, X_test = train_test_split(X, test_size=10000, random_state=1)
    
    train = numpy.array(X_train)
    test = numpy.array(X_test)
    print("saving")
    
    write_fvecs(numpy.array(X_train), base + "glove_base_%d.fvecs" % d)
    write_fvecs(numpy.array(X_test), base + "glove_query_%d.fvecs" % d)

    print("computing")
    indices, distances = calculate_gt(train, test, 100, metric='cosine')
    write_ivecs(indices, base + "glove_groudtruth_%d.ivecs" % d)