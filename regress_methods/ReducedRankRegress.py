import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def reduced_rank_regress(Y, X, dim=None, ridge_init=False, scale=False):
    """
    Reduced Rank Regression implementation.
    
    Parameters:
    -----------
    Y : ndarray, shape (N, K)
        Target data matrix where N is number of samples, K is target dimensionality.
    X : ndarray, shape (N, p)
        Source data matrix where p is source dimensionality.
    dim : int or array-like, optional
        Number(s) of predictive dimensions to test.
    ridge_init : bool, default=False
        Whether to use Ridge-Reduced Rank Regression.
    scale : bool, default=False
        Whether to use variance scaling (z-scoring).
        
    Returns:
    --------
    B : ndarray, shape (p+1, K*numDims)
        Extended mapping matrix including intercept.
    B_ : ndarray, shape (p, K)
        Predictive dimensions ordered by target variance explained.
    V : ndarray, shape (K, K)
        Eigenvectors of the optimal linear predictor.
    """
    
    # Scale the data if required
    if scale:
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        X = scaler_X.fit_transform(X)
        Y = scaler_Y.fit_transform(Y)
    
    # Exclude features (neurons) with near-zero variance
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0, ddof=1)
    idxs = np.where(np.abs(s) < np.sqrt(np.finfo(float).eps))[0]
    
    if len(idxs) > 0:
        aux_p = X.shape[1]
        aux_idxs = np.delete(np.arange(aux_p), idxs)
        X = np.delete(X, idxs, axis=1)
        m = np.delete(m, idxs)
    
    n, K = Y.shape
    p = X.shape[1]
    
    # Fit the regression model without centering
    B_full = np.linalg.lstsq(X, Y, rcond=None)[0]  # Shape: (p, K)
    
    # Compute predicted values
    Y_hat = X @ B_full  # Shape: (N, K)
    
    # PCA on predicted values
    pca = PCA(n_components=min(Y_hat.shape))
    pca.fit(Y_hat)
    V = pca.components_.T  # Shape: (K, K)
    
    # Compute B_ (Predictive dimensions)
    B_ = B_full @ V  # Shape: (p, K)
    
    # Restore B_ to original size if features were excluded
    if len(idxs) > 0:
        aux_B_ = B_.copy()
        B_ = np.zeros((aux_p, K))
        B_[aux_idxs, :] = aux_B_
    
    # Handle cases where dim is not provided
    if dim is None:
        # Add intercept term
        y_mean = np.mean(Y, axis=0)
        intercept = y_mean
        B_aug = np.vstack([intercept, B_full])  # Shape: (p+1, K)
        # Restore original feature size if necessary
        if len(idxs) > 0:
            B_final = np.zeros((aux_p + 1, K))
            B_final[0, :] = B_aug[0, :]
            B_final[1:, aux_idxs] = B_aug[1:, :]
        else:
            B_final = B_aug
        return B_final, B_, V
    
    # Ensure dim is array-like
    if isinstance(dim, (int, np.integer)):
        dim = [int(dim)]
    elif isinstance(dim, (list, tuple, np.ndarray)):
        dim = list(dim)
    else:
        raise TypeError(f"Unsupported type for dim: {type(dim)}. Must be int or array-like.")
    
    # Initialize B for multiple dimensions
    numDims = len(dim)
    B_list = []
    
    for d in dim:
        if d == 0:
            B_d = np.zeros((p, K))
        else:
            B_d = B_full @ V[:, :d] @ V[:, :d].T
        B_list.append(B_d)
    
    # Concatenate B matrices for multiple dimensions
    B_concat = np.hstack(B_list)  # Shape: (p, K*numDims)
    
    # Add intercept term
    y_mean = np.mean(Y, axis=0)
    intercept = y_mean  # Since X is not centered
    B_aug = np.vstack([intercept, B_concat])  # Shape: (p+1, K*numDims)
    
    # Restore B_aug to original size if features were excluded
    if len(idxs) > 0:
        B_final = np.zeros((aux_p + 1, K*numDims))
        B_final[0, :] = B_aug[0, :]  # Intercept
        B_final[1:, aux_idxs] = B_aug[1:, :]  # Only the included features
    else:
        B_final = B_aug
    
    return B_final, B_, V 