import numpy as np
from scipy.linalg import cholesky

def logdet(A):
    """
    Compute log(det(A)) where A is positive-definite.
    This is faster and more stable than using log(det(A)).
    
    Parameters:
    -----------
    A : ndarray, shape (n, n)
        Positive-definite matrix
        
    Returns:
    --------
    y : float
        Natural logarithm of the determinant of A
        
    Notes:
    ------
    Original MATLAB implementation by Tom Minka
    (c) Microsoft Corporation. All rights reserved.
    """
    
    # Compute Cholesky decomposition
    U = cholesky(A)
    
    # Sum the log of diagonal elements and multiply by 2
    # Equivalent to MATLAB's y = 2*sum(log(diag(U)))
    y = 2 * np.sum(np.log(np.diag(U)))
    
    return y 