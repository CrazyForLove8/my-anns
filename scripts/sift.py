from vecs_util import *

base_path = "/root/mount/dataset/sift10m/sift10m_base.fvecs"
query_path = "/root/mount/dataset/sift10m/sift10m_query.fvecs"
gt_path = "/root/mount/dataset/sift10m/sift10m_gt.ivecs"

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

    # truncate the first 5M vectors from the base
    base = base[5000000:]
    print("base after truncation:" + str(base.shape))
    print("query:" + str(query.shape))

    # save the truncated base
    write_fvecs(base, "/root/mount/dataset/sift10m/sift5m_base.fvecs")

    indices, distances = calculate_gt(base, query, 100, metric='euclidean')

    print("indices shape: " + str(indices.shape))
    print("distances shape: " + str(distances.shape))
    print("Saving indices and distances")
    write_ivecs(indices, "/root/mount/dataset/sift10m/sift5m_gt.ivecs")