import numpy as np
from pyrsistent import b
from vecs_util import *

base_path = "/root/datasets/sift/100k/sift_base.fvecs"
query_path = "/root/datasets/sift/100k/sift_query.fvecs"
gt_path = "/root/datasets/sift/100k/sift_groundtruth.ivecs"

print("Reading base and query")
base = fvecs_read(base_path)
query = fvecs_read(query_path)

print("base:" + str(base.shape))
print("query:" + str(query.shape))

print("Calculating groundtruth")
indices, distances = calculate_gt(base, query, 100, metric='euclidean')
print("Calculated groundtruth")
print(indices.shape)
print(indices)
print("Writing groundtruth")
write_ivecs(indices, gt_path)

g = ivecs_read(gt_path)
print("groundtruth: " + str(g.shape))