from vecs_util import *

base_path = "/root/mount/dataset/siftsmall/siftsmall_base.fvecs"
query_path = "/root/mount/dataset/siftsmall/siftsmall_query.fvecs"
gt_path = "/root/mount/dataset/siftsmall/siftsmall_gt.ivecs"

# check_fvecs_file(base_path)

# print("Reading base and query")
# base = fvecs_read_single_dim_header(base_path)
# query = fvecs_read_single_dim_header(query_path)
#
# print("base:" + str(base.shape))
#
# indices, distances = calculate_gt(base, query, 100, metric='euclidean')
#
# print("indices shape: " + str(indices.shape))
# print("distances shape: " + str(distances.shape))
# print("Saving indices and distances")
# write_ivecs(indices, gt_path)

if __name__ == "__main__":
    print("Reading base and query")
    base = fvecs_read(base_path)
    query = fvecs_read(query_path)

    print("base:" + str(base.shape))

    indices, distances = calculate_gt(base, query, 100, metric='euclidean')

    print("indices shape: " + str(indices.shape))
    print("distances shape: " + str(distances.shape))
    print("Saving indices and distances")
    write_ivecs(indices, gt_path)