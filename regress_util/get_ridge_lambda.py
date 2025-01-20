import numpy as np

def get_ridge_lambda(d_max_shrink, X, scale=True):
    """
    Compute appropriate range for Ridge regression regularization parameter
    
    Parameters:
    -----------
    d_max_shrink : array-like
        Shrinkage factors to compute lambda values
    X : ndarray, shape (N, p)
        Data matrix where N is samples, p is source dimensionality
    scale : bool, default=True
        Whether to use variance scaling (z-scoring)
        
    Returns:
    --------
    lambda_ : ndarray
        Set of regularization parameters for Ridge regression
    dof : ndarray
        Effective degrees of freedom for each lambda value
        
    Notes:
    ------
    See Elements of Statistical Learning, by Hastie, Tibshirani and Friedman
    for more information.
    """
    
    # Calculate mean and standard deviation
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0, ddof=1)  # ddof=1 for sample standard deviation
    
    # Find indices where standard deviation is near zero
    idxs = np.where(np.abs(s) < np.sqrt(np.finfo(float).eps))[0]
    
    # Remove columns with zero variance
    if len(idxs) > 0:
        X = np.delete(X, idxs, axis=1)
        m = np.delete(m, idxs)
        s = np.delete(s, idxs)
    
    # Get dimensions
    n, K = X.shape
    
    # Create matrices of repeated means and standard deviations
    M = np.tile(m, (n, 1))
    S = np.tile(s, (n, 1))
    
    # Center and optionally scale the data
    if scale:
        Z = (X - M) / S
    else:
        Z = X - M
    
    # Compute eigenvalues and maximum eigenvalue
    d = np.linalg.eigvals(Z.T @ Z)
    d_max = np.max(d)
    
    # Compute lambda values
    lambda_ = d_max * (1 - d_max_shrink) / d_max_shrink
    
    # Compute degrees of freedom for each lambda
    num_lambdas = len(lambda_)
    D = np.tile(d, (num_lambdas, 1)).T
    Lambda = np.tile(lambda_, (K, 1))
    dof = np.sum(D / (D + Lambda), axis=0)
    
    return lambda_, dof 