import numpy as np
from pyrsistent import b
from vecs_util import *

base_path = "/root/datasets/sift/1m/sift_base.fvecs"

base = fvecs_read(base_path)

# split it into 200k and 500k

write_fvecs(base[:200000], "/root/datasets/sift/200k/sift_base.fvecs")

write_fvecs(base[:500000], "/root/datasets/sift/500k/sift_base.fvecs")

# compute gt for each of them

for i in [200, 500]:
    base_path = "/root/datasets/sift/" + str(i) + "k/sift_base.fvecs"
    query_path = "/root/datasets/sift/1m/sift_query.fvecs"
    gt_path = "/root/datasets/sift/" + str(i) + "k/sift_groundtruth.ivecs"
    
    base = fvecs_read(base_path)
    query = fvecs_read(query_path)
    
    indices, distances = calculate_gt(base, query, 100)
    
    write_ivecs(indices, gt_path)
    
    print("Done for " + str(i) + "k")
    




# write_fvecs(base[:2000000], "/root/datasets/sift/2m/sift_base.fvecs")

# write_fvecs(base[:3000000], "/root/datasets/sift/3m/sift_base.fvecs")

# write_fvecs(base[:4000000], "/root/datasets/sift/4m/sift_base.fvecs")

# write_fvecs(base[:5000000], "/root/datasets/sift/5m/sift_base.fvecs")

# # compute gt for each of them

# for i in range(2, 6):
#     base_path = "/root/datasets/sift/" + str(i) + "m/sift_base.fvecs"
#     query_path = "/root/datasets/sift/1m/sift_query.fvecs"
#     gt_path = "/root/datasets/sift/" + str(i) + "m/sift_groundtruth.ivecs"
    
#     base = fvecs_read(base_path)
#     query = fvecs_read(query_path)
    
#     indices, distances = calculate_gt(base, query, 100)
    
#     write_ivecs(indices, gt_path)
    
#     print("Done for " + str(i) + "m")