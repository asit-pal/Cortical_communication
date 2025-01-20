import numpy as np
from scipy.linalg import cholesky
from .FactorAnalysis import factor_analysis
from .mvn_log_like import mvn_log_like

def factor_analysis_test_log_like(X_train, X_test, q, method='FA'):
    """
    Compute log-likelihood of test data under Factor Analysis models
    
    Applies factor analysis models with latent dimensionalities given by q to X_train,
    and computes the log-likelihood of the data in X_test under these models.
    
    Parameters:
    -----------
    X_train : ndarray, shape (N_train, p)
        Training data matrix
    X_test : ndarray, shape (N_test, p)
        Test data matrix
    q : array-like
        Vector of latent dimensionalities to test
    method : str, default='FA'
        Method to use ('FA' or 'PPCA')
        
    Returns:
    --------
    log_like : ndarray
        Log likelihood of test data under the models fit to training data
    """
    
    # Calculate mean and covariance of training data
    # Equivalent to MATLAB's mean() and cov(X, 1)
    m = np.mean(X_train, axis=0)
    S = np.cov(X_train, rowvar=False, bias=True)
    
    # Initialize log likelihood array
    num_dims = len(q)
    log_like = np.zeros(num_dims)
    
    # Loop through each dimensionality
    for i in range(num_dims):
        if q[i] == 0:
            # For q=0, use diagonal covariance matrix
            Psi = np.diag(np.diag(S))
            
            # Check if matrix is positive definite using Cholesky
            try:
                cholesky(Psi)
                log_like[i] = mvn_log_like(X_test, m, Psi)
            except np.linalg.LinAlgError:
                log_like[i] = np.nan
                
        else:
            # Fit Factor Analysis model
            L, psi = factor_analysis(S, q[i], method=method)
            
            # Check for near-zero values in psi
            idxs = np.where(np.abs(psi) < np.sqrt(np.finfo(float).eps))[0]
            if len(idxs) > 0:
                log_like[i] = np.nan
                continue
            
            # Compute full covariance matrix
            Psi = np.diag(psi)
            C = L @ L.T + Psi
            
            # Check if matrix is positive definite using Cholesky
            try:
                cholesky(C)
                log_like[i] = mvn_log_like(X_test, m, C)
            except np.linalg.LinAlgError:
                log_like[i] = np.nan
    
    return log_like 