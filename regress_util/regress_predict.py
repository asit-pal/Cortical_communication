import numpy as np
from .mean_squared_error import mean_squared_error
from .normalized_squared_error import normalized_squared_error

def regress_predict(Y, X, B, loss_measure='NSE'):
    """
    Predict target data and compute prediction error.
    
    Parameters:
    -----------
    Y : ndarray, shape (N, K)
        Target data matrix where N is samples, K is target dimensionality.
    X : ndarray, shape (N, p)
        Source data matrix where p is source dimensionality.
    B : ndarray, shape (p+1, K*numDims)
        Mapping matrix (includes intercept). If multiple dimensions are tested,
        B is extended horizontally.
    loss_measure : str, default='NSE'
        Loss measure to use:
        - 'NSE': Normalized Squared Error (default)
        - 'MSE': Mean Squared Error
        - 'MVNSE': Multivariate Normalized Squared Error
    
    Returns:
    --------
    loss : float or ndarray
        Loss incurred when predicting Y using X and mapping B.
        If multiple dimensions are used, returns an array of losses.
    Y_hat : ndarray, shape (N, K*numDims)
        Predicted target data.
    """
    
    # Always add intercept term, matching MATLAB behavior
    intercept = np.ones((X.shape[0], 1))
    X_aug = np.hstack([intercept, X])  # Shape: (N, p+1)
    
    # Validate dimensions
    if B.shape[0] != X_aug.shape[1]:
        raise ValueError(f"Mismatch in dimensions: X_aug has {X_aug.shape[1]} columns, but B has {B.shape[0]} rows.")
    
    # Compute predictions
    Y_hat = X_aug @ B  # Shape: (N, K*numDims)
    
    # Compute loss based on specified measure
    if loss_measure.upper() == 'MSE':
        loss = mean_squared_error(Y, Y_hat)
    elif loss_measure.upper() == 'NSE':
        loss = normalized_squared_error(Y, Y_hat)
    elif loss_measure.upper() == 'MVNSE':
        from .mv_normalized_squared_error import mv_normalized_squared_error
        loss = mv_normalized_squared_error(Y, Y_hat)
    else:
        raise ValueError(f"Unknown loss measure: {loss_measure}")
    
    return loss, Y_hat 