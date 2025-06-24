from vecs_util import *

base2 = fvecs_read("/root/datasets/crawl/crawl_base.fvecs")
query2 = fvecs_read("/root/datasets/crawl/crawl_query.fvecs")

print("base:" + str(base2.shape))
print("query:" + str(query2.shape))

indices, distances = calculate_gt(base2, query2, 100, metric='cosine')

print("Calculated groundtruth")
print(indices.shape)
print(indices)

write_ivecs(indices, "/root/datasets/crawl/crawl_groundtruth.ivecs")

g = ivecs_read("/root/datasets/crawl/crawl_groundtruth.ivecs")
print("groundtruth: " + str(g.shape))