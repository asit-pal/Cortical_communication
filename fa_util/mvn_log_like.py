import numpy as np
from scipy.linalg import solve
from .logdet import logdet

def mvn_log_like(X, m, S):
    """
    Compute log-likelihood of data under multivariate Gaussian distribution
    
    Parameters:
    -----------
    X : ndarray, shape (N, p)
        Data matrix where N is number of samples, p is dimensionality
    m : ndarray, shape (p,)
        Mean vector
    S : ndarray, shape (p, p)
        Covariance matrix
        
    Returns:
    --------
    log_like : float
        Log likelihood of the data
    """
    
    # Get dimensions of data
    N, p = X.shape
    
    # Create matrix of repeated means
    # Equivalent to MATLAB's M = m(ones(N, 1), :)
    M = np.tile(m, (N, 1))
    
    # Center the data
    X = X - M
    
    # Compute log determinant of covariance matrix
    # Using custom logdet function for positive-definite matrices
    logdet_S = logdet(S)
    
    # Solve system of equations S\X' efficiently
    # Equivalent to MATLAB's S\X'
    S_inv_X = solve(S, X.T)
    
    # Compute log likelihood
    # sum(sum()) in MATLAB becomes np.sum() with axis=None
    log_like = -0.5 * (N * p * np.log(2*np.pi) + 
                       N * logdet_S + 
                       np.sum(X * S_inv_X.T))
    
    return log_like 