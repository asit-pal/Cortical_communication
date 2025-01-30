import numpy as np

def Squash(A):
    """
    Reshape a 3D array into a 2D array by combining the last two dimensions
    """
    dim_A = A.shape
    return np.reshape(A, [dim_A[0], dim_A[1] * dim_A[2]]) 