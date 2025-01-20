import numpy as np
from sklearn.model_selection import KFold
from .FactorAnalysis import factor_analysis
from .CrossValFa import cross_val_fa
from .FactorAnalysisModelSelect import factor_analysis_model_select

def extract_fa_latents(X, q=None):
    """
    Extract Factor Analysis latent variables
    
    Parameters:
    -----------
    X : ndarray, shape (N, p)
        Data matrix where N is number of samples, p is data dimensionality
    q : int or array-like, optional
        Latent dimensionality or array of dimensionalities to test
        If None, will test q = 0:(p-1)
        
    Returns:
    --------
    Z : ndarray, shape (N, q)
        Latent variables
    U : ndarray, shape (p, q)
        Factor analysis dominant dimensions
    Q : ndarray, shape (p, q)
        Factor analysis "decoding" matrix
    q_opt : int
        Optimal dimensionality found via cross-validation
    """
    
    # Constants (equivalent to MATLAB's C_CV_NUM_FOLDS and C_CV_OPTIONS)
    CV_NUM_FOLDS = 10
    
    # Get data dimensions
    n, p = X.shape
    
    # If q not provided, test all possible dimensions
    if q is None:
        q = np.arange(p)  # equivalent to MATLAB's 0:p-1
    
    # If multiple q values provided, do cross-validation to find optimal q
    if np.size(q) > 1:
        # Cross validate FA model for each dimensionality
        cv_loss = cross_val_fa(X, q, CV_NUM_FOLDS)
        # Select optimal dimensionality
        q_opt = factor_analysis_model_select(cv_loss, q)
    else:
        q_opt = q
    
    # Compute covariance matrix
    Sigma = np.cov(X, rowvar=False)  # rowvar=False because samples are in rows
    
    # Fit Factor Analysis model - now correctly unpacking three return values
    L, psi, _ = factor_analysis(Sigma, q_opt)
    
    # Create diagonal matrix from psi
    Psi = np.diag(psi)
    
    # Compute total covariance
    C = L @ L.T + Psi
    
    # SVD of loading matrix L
    U, S, V = np.linalg.svd(L, full_matrices=False)
    
    # Compute decoding matrix Q
    # Equivalent to MATLAB's Q = C\L*V*S'
    Q = np.linalg.solve(C, L @ V.T @ S)
    
    # Center the data
    m = np.mean(X, axis=0)
    M = np.tile(m, (n, 1))
    
    # Compute latent variables
    Z = (X - M) @ Q
    
    return Z, U, Q, q_opt 