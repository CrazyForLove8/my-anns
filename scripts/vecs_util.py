import os
import numpy
from sklearn.model_selection import train_test_split


def ivecs_read(fname):
    a = numpy.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    with open(fname, 'rb') as f:
        vectors = []
        while True:
            len_prefix = numpy.fromfile(f, dtype=numpy.int32, count=1)
            if len_prefix.size == 0:
                break
            d = len_prefix[0]
            vector = numpy.fromfile(f, dtype=numpy.float32, count=d)
            vectors.append(vector)
    return numpy.array(vectors)

def fvecs_read_new(fname):
    with open(fname, 'rb') as f:
        vectors = []
        dim = -1
        while True:
            if dim == -1 :
                len_prefix = numpy.fromfile(f, dtype=numpy.int32, count=1)
                if len_prefix.size == 0:
                    break
                dim = len_prefix[0]
            vector = numpy.fromfile(f, dtype=numpy.float32, count=dim)
            vectors.append(vector)
    return numpy.array(vectors)

def write_fvecs(X, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'wb') as f:
        for vec in X:
            numpy.array([len(vec)], dtype=numpy.int32).tofile(f)
            vec.astype(numpy.float32).tofile(f)
            
def write_ivecs(X : numpy.array, output_path : str):
    with open(output_path, 'wb') as f:
        for vec in X:
            numpy.array([len(vec)], dtype=numpy.int32).tofile(f)
            vec.astype(numpy.int32).tofile(f)
        
def calculate_gt(X_train, X_test, n_neighbors, metric='euclidean'):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric=metric).fit(X_train)
    distances, indices = nbrs.kneighbors(X_test)
    return indices, distances

def fvecs_read_single_dim_header(fname):
    with open(fname, 'rb') as f:
        # 1. Read the single dimension prefix at the very beginning
        dim_prefix = numpy.fromfile(f, dtype=numpy.int32, count=1)

        if dim_prefix.size == 0:
            raise ValueError(f"File '{fname}' is empty or too short to read dimension.")

        dim = dim_prefix[0]
        if dim <= 0:
            raise ValueError(f"Invalid dimension ({dim}) read from file '{fname}'. Dimension must be positive.")

        print(f"Detected global dimension: {dim}")

        # 2. Calculate the number of vectors
        # Get current position in file (after reading dimension)
        current_pos = f.tell()
        # Get total file size
        file_size = os.fstat(f.fileno()).st_size

        # Remaining bytes after reading the dimension header
        remaining_bytes = file_size - current_pos

        # Each vector has 'dim' float32 elements, and each float32 is 4 bytes
        bytes_per_vector = dim * 4

        if bytes_per_vector == 0: # Avoid division by zero if dim somehow becomes 0
            raise ValueError(f"Calculated bytes per vector is zero (dimension={dim}). Cannot proceed.")

        num_vectors = remaining_bytes // bytes_per_vector
        remainder_bytes = remaining_bytes % bytes_per_vector

        print(f"Total file size: {file_size} bytes")
        print(f"Bytes after dimension header: {remaining_bytes} bytes")
        print(f"Bytes per vector: {bytes_per_vector} bytes")
        print(f"Expected number of vectors: {num_vectors}")

        if remainder_bytes != 0:
            print(f"WARNING: File size is not a perfect multiple of vector size after header. Remainder bytes: {remainder_bytes}.")
            print("This indicates a truncated or malformed file. Reading as many full vectors as possible.")

        # 3. Read all vector data at once
        # Calculate total count of float32 elements to read
        total_float_elements = num_vectors * dim
        all_vectors_flat = numpy.fromfile(f, dtype=numpy.float32, count=total_float_elements)

        # 4. Reshape the flat array into a 2D array (num_vectors x dim)
        if all_vectors_flat.size != total_float_elements:
            print(f"WARNING: Actual elements read ({all_vectors_flat.size}) do not match expected ({total_float_elements}).")
            print("This could be due to file truncation. Reshaping with actual read count.")
            # Adjust num_vectors based on actual elements read if truncation occurred
            num_vectors_actual = all_vectors_flat.size // dim
            if all_vectors_flat.size % dim != 0:
                print(f"ERROR: Final read data is not a multiple of dimension. Cannot reshape properly.")
                raise ValueError("Data read is inconsistent with expected dimension for all vectors.")
            return all_vectors_flat.reshape((num_vectors_actual, dim))
        else:
            return all_vectors_flat.reshape((num_vectors, dim))
def check_fvecs_file(fname):
    print(f"--- Checking file: {fname} ---")
    try:
        with open(fname, 'rb') as f:
            vector_count = 0
            expected_dim = -1

            while True:
                # Read the dimension prefix
                len_prefix = numpy.fromfile(f, dtype=numpy.int32, count=1)

                if len_prefix.size == 0:
                    # End of file reached
                    print("End of file reached.")
                    break

                current_dim = len_prefix[0]

                # Check for valid dimension (e.g., positive)
                if current_dim <= 0:
                    print(f"  ERROR: Vector {vector_count+1} has an invalid dimension (<=0): {current_dim}. Likely file corruption.")
                    break # Stop processing, file is malformed

                # Read the vector data
                # We'll calculate expected bytes to read
                expected_bytes_for_vector = current_dim * 4
                # Read raw bytes to check if enough data is available
                vector_bytes = f.read(expected_bytes_for_vector)

                if len(vector_bytes) != expected_bytes_for_vector:
                    print(f"  ERROR: Vector {vector_count+1} (dim={current_dim}) is truncated!")
                    print(f"  Expected {expected_bytes_for_vector} bytes, but read only {len(vector_bytes)} bytes.")
                    break # File truncated or corrupted

                # If we read enough bytes, convert them to numpy array
                vector = numpy.frombuffer(vector_bytes, dtype=numpy.float32)


                if expected_dim == -1:
                    expected_dim = current_dim
                    print(f"  First vector dimension detected: {expected_dim}")
                elif current_dim != expected_dim:
                    print(f"  ERROR: Vector {vector_count+1} has inconsistent dimension!")
                    print(f"  Expected: {expected_dim}, Found: {current_dim}")
                    break # Stop processing due to inconsistent dimensions

                vector_count += 1
                if vector_count % 10000 == 0: # Print progress every 1000 vectors
                    print(f"  Processed {vector_count} vectors...")

            print(f"--- Finished checking. Total valid vectors found: {vector_count} ---")
            if expected_dim != -1:
                print(f"  All vectors (if valid) have dimension: {expected_dim}")

    except Exception as e:
        print(f"An error occurred during file check: {e}")
