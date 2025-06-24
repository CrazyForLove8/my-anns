import numpy as np
from pyrsistent import b
from vecs_util import *

base_path = "/root/datasets/sift/10k/sift_base.fvecs"

base = fvecs_read(base_path)

def add_duplicate_randomly(data, min_copies=0, max_copies=5, epsilon=0.01):
    """
    Augment a dataset by creating a random number of copies of each point with small random offsets.

    :param data: Original dataset as a NumPy array of shape (n, d).
    :param min_copies: Minimum number of copies for each point.
    :param max_copies: Maximum number of copies for each point.
    :param epsilon: Maximum offset for random perturbations.
    :return: Augmented dataset as a NumPy array.
    """
    n, d = data.shape
    augmented_data = [data]  # Start with the original dataset

    for point in data:
        # Randomly determine the number of copies for this point
        num_copies = np.random.randint(min_copies, max_copies + 1)
        # Generate random offsets for the copies
        offsets = np.random.uniform(-epsilon, epsilon, size=(num_copies, d))
        # Apply offsets to create perturbed points
        perturbed_points = point + offsets
        # Add the perturbed points to the list
        augmented_data.append(perturbed_points)

    # Concatenate all parts of the data
    augmented_data = np.vstack(augmented_data)
    return augmented_data

def add_duplicate(data, num_copies=5, epsilon=0.01, index=None):
    """
    Augment a dataset by creating a fixed number of copies for one defined point with small random offsets.

    :param data: Original dataset as a NumPy array of shape (n, d).
    :param num_copies: Number of copies for each point.
    :param epsilon: Maximum offset for random perturbations.
    :param index: Index of the point to duplicate.
    :return: Augmented dataset as a NumPy array.
    """
    n, d = data.shape
    if index is None:
        index = np.random.randint(n)
    point = data[index]    
    # Generate random offsets for the copies
    offsets = np.random.uniform(-epsilon, epsilon, size=(num_copies, d))
    # Apply offsets to create perturbed points
    perturbed_points = point + offsets
    # Concatenate the original and perturbed points
    augmented_data = np.vstack([data, perturbed_points])
    return augmented_data

# base = add_duplicate_randomly(base, min_copies=0, max_copies=10, epsilon=0.01)
rand_a, rand_b, rand_c = np.random.randint(0, 10000, 3)
base = add_duplicate(base, num_copies=10000, epsilon=0.01, index=rand_a)
base = add_duplicate(base, num_copies=10000, epsilon=0.01, index=rand_b)
base = add_duplicate(base, num_copies=10000, epsilon=0.01, index=rand_c)
np.random.shuffle(base)
print("base:" + str(base.shape))

write_fvecs(base, "/root/datasets/sift_perturbed/10k/sift_base_1_3_001.fvecs")