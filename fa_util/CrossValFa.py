import numpy as np
from sklearn.model_selection import KFold
from .FactorAnalysis import factor_analysis
from .FactorAnalysisTestLogLike import factor_analysis_test_log_like
from .FactorAnalysisModelSelect import factor_analysis_model_select

def cross_val_fa(X, q, cv_num_folds=10):
    """
    Cross-validate Factor Analysis model
    
    Performs cross-validation for the Factor Analysis model. Computes the 
    cross-validated log-likelihood for data X and latent state dimensionalities q,
    selects the latent dimensionality q_max for which the cross-validated 
    log-likelihood is highest and returns the cumulative shared variance explained.
    
    Parameters:
    -----------
    X : ndarray, shape (N, p)
        Data matrix where N is number of samples, p is data dimensionality
    q : array-like
        Vector of latent dimensionalities to test
    cv_num_folds : int, default=10
        Number of folds for cross-validation
        
    Returns:
    --------
    cv_loss : ndarray
        Cumulative shared variance explained by latent dimensions for FA model
        with highest cross-validated log-likelihood
    cv_log_like : ndarray
        Cross-validated log-likelihood for each fold and dimension
    """
    
    # Sort q in ascending order
    q = np.sort(q)
    
    # Initialize cross-validation
    kf = KFold(n_splits=cv_num_folds, shuffle=True)
    
    # Initialize array to store log-likelihood for each fold and dimension
    cv_log_like = np.zeros((cv_num_folds, len(q)))
    
    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        
        # Calculate log-likelihood for this fold
        cv_log_like[fold_idx] = factor_analysis_test_log_like(X_train, X_test, q)
    
    # Calculate covariance matrix with bias=True (equivalent to MATLAB's cov(X, 1))
    S = np.cov(X, rowvar=False, bias=True)
    
    # Find q that maximizes mean log-likelihood
    mean_log_like = np.nanmean(cv_log_like, axis=0)
    q_max_idx = np.argmax(mean_log_like)
    q_max = q[q_max_idx]
    
    if q_max == 0:
        cv_loss = np.nan
    else:
        # Get loading matrix L from factor analysis
        L = factor_analysis(S, q_max)[0]  # Only need first return value
        
        # Calculate eigenvalues of L*L^T in descending order
        d = np.sort(np.linalg.eigvals(L @ L.T))[::-1]
        
        # Calculate cumulative shared variance explained
        cv_loss = (1 - np.cumsum(d)/np.sum(d))
        
        # Handle case where q starts from 0
        if q[0] == 0:
            cv_loss = np.concatenate(([1], cv_loss))
            cv_loss = cv_loss[q + 1]
        else:
            cv_loss = cv_loss[q]
    
    return cv_loss, cv_log_like 