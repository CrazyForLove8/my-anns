import numpy as np
from pyrsistent import b
from vecs_util import *

base_path = "/root/datasets/deep/1m/deep_base.fvecs"

print("Reading base and query")
base = fvecs_read(base_path)

print("base:" + str(base.shape))

print("Divide into 2 parts")
base1 = base[:500000]
base2 = base[500000:]

print("save")
write_fvecs(base1, "/root/datasets/deep/1m/2/deep_base1.fvecs")
write_fvecs(base2, "/root/datasets/deep/1m/2/deep_base2.fvecs")

print("Divide into 4 parts")
base1 = base[:250000]
base2 = base[250000:500000]
base3 = base[500000:750000]
base4 = base[750000:]

print("save")
write_fvecs(base1, "/root/datasets/deep/1m/4/deep_base1.fvecs")
write_fvecs(base2, "/root/datasets/deep/1m/4/deep_base2.fvecs")
write_fvecs(base3, "/root/datasets/deep/1m/4/deep_base3.fvecs")
write_fvecs(base4, "/root/datasets/deep/1m/4/deep_base4.fvecs")

print("Divide into 8 parts")
base1 = base[:125000]
base2 = base[125000:250000]
base3 = base[250000:375000]
base4 = base[375000:500000]
base5 = base[500000:625000]
base6 = base[625000:750000]
base7 = base[750000:875000]
base8 = base[875000:]

print("save")
write_fvecs(base1, "/root/datasets/deep/1m/8/deep_base1.fvecs")
write_fvecs(base2, "/root/datasets/deep/1m/8/deep_base2.fvecs")
write_fvecs(base3, "/root/datasets/deep/1m/8/deep_base3.fvecs")
write_fvecs(base4, "/root/datasets/deep/1m/8/deep_base4.fvecs")
write_fvecs(base5, "/root/datasets/deep/1m/8/deep_base5.fvecs")
write_fvecs(base6, "/root/datasets/deep/1m/8/deep_base6.fvecs")
write_fvecs(base7, "/root/datasets/deep/1m/8/deep_base7.fvecs")
write_fvecs(base8, "/root/datasets/deep/1m/8/deep_base8.fvecs")