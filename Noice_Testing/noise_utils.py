import numpy as np


def add_gaussian_noise(A, sigma):
    """Add elementwise Gaussian noise to a matrix or sparse matrix."""
    # convert to dense for simplicity
    A_dense = A.toarray() if hasattr(A, "toarray") else np.array(A)
    noise = np.random.normal(scale=sigma, size=A_dense.shape)
    return A_dense + noise
