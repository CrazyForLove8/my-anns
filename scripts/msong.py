import numpy as np
from vecs_util import *

base = fvecs_read("/root/datasets/msong/msong_base.fvecs")
query = fvecs_read("/root/datasets/msong/msong_query.fvecs")

print(base.shape)
print(query.shape)

# indices, distances = calculate_gt(base, query, 100, metric='euclidean')
# print(indices.shape)
# print(indices)
# write_ivecs(indices, "/root/datasets/msong/msong_groundtruth.ivecs")

# g = ivecs_read("/root/datasets/msong/msong_groundtruth.ivecs")
# print(g.shape)