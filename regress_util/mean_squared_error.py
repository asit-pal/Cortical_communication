import numpy as np

def mean_squared_error(Y, Y_hat):
    """
    Compute Mean Squared Error between Y and Y_hat.
    
    Parameters:
    -----------
    Y : ndarray, shape (N, K)
        True target values.
    Y_hat : ndarray, shape (N, K)
        Predicted target values.
    
    Returns:
    --------
    mse : float
        Mean Squared Error.
    """
    mse = np.mean((Y - Y_hat) ** 2)
    return mse 