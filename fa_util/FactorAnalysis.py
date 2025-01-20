import numpy as np
from scipy.linalg import cholesky, eigh
from scipy.stats.mstats import gmean

def factor_analysis(S, q, method='FA'):
    """
    Apply Factor Analysis to sample covariance matrix
    
    Parameters:
    -----------
    S : ndarray, shape (p, p)
        Sample covariance matrix
    q : int
        Number of factors (latent dimensionality)
    method : str, default='FA'
        Method to use ('FA' or 'PPCA')
        
    Returns:
    --------
    L : ndarray, shape (p, q)
        Factor loadings
    psi : ndarray, shape (p,)
        Diagonal of uniqueness matrix
    log_like : float
        Log likelihood at final EM iteration
    """
    
    # Constants
    TOL = 1e-8           # Stopping criterion for EM
    MAX_ITER = int(1e8)  # Maximum number of EM iterations
    MIN_FRAC_VAR = 0.01  # Minimum fraction of variance
    
    # Exclude dimensions with zero variance
    s = np.diag(S)
    idxs = np.where(np.abs(s) < np.sqrt(np.finfo(float).eps))[0]
    
    if len(idxs) > 0:
        aux_p = S.shape[0]
        aux_idxs = np.arange(aux_p)
        aux_idxs = np.delete(aux_idxs, idxs)
        S = np.delete(np.delete(S, idxs, axis=0), idxs, axis=1)
    
    # Initialize parameters
    p = S.shape[0]
    if np.linalg.matrix_rank(S) == p:
        # If S is full rank, use Cholesky decomposition
        scale = np.exp(2 * np.sum(np.log(np.diag(cholesky(S)))) / p)
    else:
        # If S is rank deficient, use eigendecomposition
        r = np.linalg.matrix_rank(S)
        d = np.sort(np.linalg.eigvals(S))[::-1]  # Sort in descending order
        scale = gmean(d[:r])
    
    # Initialize L and psi
    L = np.random.randn(p, q) * np.sqrt(scale/q)
    psi = np.diag(S).copy()
    
    # Set variance floor
    var_floor = MIN_FRAC_VAR * psi
    
    # Initialize EM variables
    I = np.eye(q)
    c = -p/2 * np.log(2*np.pi)
    log_like = 0
    
    # EM iterations
    for i in range(MAX_ITER):
        # E-step
        # Use Woodbury identity for faster inverse computation
        inv_psi = np.diag(1/psi)
        inv_psi_times_L = inv_psi @ L
        inv_C = inv_psi - inv_psi_times_L @ np.linalg.solve(
            I + L.T @ inv_psi_times_L, L.T @ inv_psi)
        
        # Compute expected values
        V = inv_C @ L
        S_times_V = S @ V
        EZZ = I - V.T @ L + V.T @ S_times_V
        
        # Compute log-likelihood
        prev_log_like = log_like
        ldm = np.sum(np.log(np.diag(cholesky(inv_C))))
        log_like = c + ldm - 0.5 * np.sum(inv_C * S)
        
        # Check convergence
        if i <= 2:
            base_log_like = log_like
        elif (log_like - base_log_like) < (1 + TOL) * (prev_log_like - base_log_like):
            break
        
        # M-step
        L = S_times_V @ np.linalg.solve(EZZ, np.eye(q))
        psi = np.diag(S) - np.sum(S_times_V * L, axis=1)
        
        if method.upper() == 'PPCA':
            psi = np.mean(psi) * np.ones(p)
        else:  # FA
            psi = np.maximum(var_floor, psi)
    
    # Restore original dimensions if needed
    if len(idxs) > 0:
        aux_L = L.copy()
        L = np.zeros((aux_p, q))
        L[aux_idxs] = aux_L
        
        aux_psi = psi.copy()
        psi = np.zeros(aux_p)
        psi[aux_idxs] = aux_psi
    
    return L, psi, log_like 