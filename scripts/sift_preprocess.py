import numpy as np
import os

def fvecs_read(fname):
    with open(fname, 'rb') as f:
        vectors = []
        while True:
            len_prefix = np.fromfile(f, dtype=np.int32, count=1)
            if len_prefix.size == 0:
                break
            d = len_prefix[0]
            vector = np.fromfile(f, dtype=np.float32, count=d)
            vectors.append(vector)
    return np.array(vectors)

def write_fvecs(X, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'wb') as f:
        for vec in X:
            np.array([len(vec)], dtype=np.int32).tofile(f)
            vec.astype(np.float32).tofile(f)

def preprocess(data, min_copies=0, max_copies=5, epsilon=0.01):
    """
    Preprocess the data by augmenting it with nearby points.
    :param data: The data to preprocess.
    :param min_copies: The minimum number of copies to make for each point.
    :param max_copies: The maximum number of copies to make for each point.
    :param epsilon: The maximum perturbation to apply to each copy.
    :return: The preprocessed data.
    """
    _, d = data.shape
    augmented_data = [data]

    for point in data:
        num_copies = np.random.randint(min_copies, max_copies + 1)
        offsets = np.random.uniform(-epsilon, epsilon, size=(num_copies, d))
        perturbed_points = point + offsets
        augmented_data.append(perturbed_points)
    augmented_data = np.vstack(augmented_data)
    return augmented_data

base_path = "datasets/siftsmall/siftsmall_base.fvecs"

base = fvecs_read(base_path)

base = preprocess(base)
print("base:" + str(base.shape))

write_fvecs(base, "datasets/siftsmall/siftsmall_base_preprocessed.fvecs")