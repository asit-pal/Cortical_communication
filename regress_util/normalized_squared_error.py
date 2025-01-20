import numpy as np

def normalized_squared_error(Y, Y_hat):
    """
    Compute Normalized Squared Error between Y and Y_hat.
    
    Normalized Squared Error (NSE) is defined as the squared norm of the error
    divided by the squared norm of the true values.
    
    Parameters:
    -----------
    Y : ndarray, shape (N, K)
        True target values.
    Y_hat : ndarray, shape (N, K)
        Predicted target values.
    
    Returns:
    --------
    nse : float
        Normalized Squared Error.
    """
    error = Y - Y_hat
    nse = np.sum(error ** 2) / np.sum(Y ** 2)
    return nse 