# Research on Approximate Nearest Neighbors Search

## Contents

- [Topic](#topic)
- [Implementation](#implementation)
    - [Algorithms](#algorithms)
        - [Diversification-based HNSW](#diversification-based-hnsw)
        - [Sphere-based HNSW](#sphere-based-hnsw)
    - [Dataset preparation](#dataset-preparation)
    - [Running the code](#running-the-code)
        - [Environment](#environment)
        - [Build](#build)
        - [Tests](#tests)
- [Contributions](#contributions)
    - [Formatting the code](#formatting-the-code)
- [References](#references)

## Topic

Diversified Top-k Retrieval for Approximate Nearest Neighbors Search.

## Implementation

### Algorithms

#### Diversification-based HNSW

This method makes use of search optimizations. In this algorithm, HNSW's search process is modified with naive
diversification method and MMR method.

#### Sphere-based HNSW

This method makes use of index techniques. We construct the index by grouping points into spheres. The center of the
sphere is used as the embedding in the index, and the radius of the sphere is used as the threshold. When constructing
the index, we actually build the index based on all the centers of the spheres, rather than all the points.

### Dataset preparation

- [SIFT10k](http://corpus-texmex.irisa.fr/): 10k SIFT features.

Download sift10k by running the following command:

```bash
python3 scripts/sift_download.py
```

And preprocess the dataset by running the following command:

```bash
python3 scripts/sift_preprocess.py
```

After running the above command, we will get a perturbed SIFT10k dataset with around 35k points, with each point in the
original dataset copied 0~5 times randomly.

### Running the code

#### Environment

- WSL2
- Ubuntu 22.04
- CMake 3.22.1
- GCC 11.4.0

#### Build

```bash
mkdir build
cd build
cmake ..
make
```

#### Tests

We provide three tests for the algorithms.

`tests/test_algorithm.cpp` tests the performance of the original HNSW algorithm on the perturbed SIFT10k dataset.

`tests/test_dhnsw.cpp` tests the performance of the diversification-based HNSW algorithm on the perturbed SIFT10k
dataset.

`tests/test_shnsw.cpp` tests the performance of the sphere-based HNSW algorithm on the perturbed SIFT10k dataset.

To run the tests, execute the following commands:

```bash
./tests/test_algorithm
./tests/test_dhnsw
./tests/test_shnsw
```

## Contributions

### Formatting the code

We use `clang-format` to format the code. To format the code, execute the following command:

```bash
sh scripts/format-cpp.sh
```

## References

- [HNSW](https://ieeexplore.ieee.org/abstract/document/8594636)
- [VSAG](https://github.com/antgroup/vsag)
