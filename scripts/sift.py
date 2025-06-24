import numpy as np
from pyrsistent import b
from vecs_util import *

base_path = "/root/datasets/sift/1m/sift_base.fvecs"

print("Reading base and query")
base = fvecs_read(base_path)

print("base:" + str(base.shape))

print("Divide into 2 parts")
base1 = base[:500000]
base2 = base[500000:]

print("save")
write_fvecs(base1, "/root/datasets/sift/1m/2/sift_base1.fvecs")
write_fvecs(base2, "/root/datasets/sift/1m/2/sift_base2.fvecs")