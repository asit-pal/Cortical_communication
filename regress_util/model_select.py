import numpy as np

def model_select(cv_loss, num_dims):
    """
    Select the optimal model based on cross-validation loss.
    
    Parameters:
    -----------
    cv_loss : ndarray, shape (2, num_dims)
        Cross-validation loss where:
        - cv_loss[0] contains mean loss across folds.
        - cv_loss[1] contains standard error of the mean.
    num_dims : ndarray
        Array of dimension values tested.
    
    Returns:
    --------
    opt_dim : int
        Optimal number of dimensions with the lowest mean loss.
    """
    mean_loss = cv_loss[0]
    opt_index = np.argmin(mean_loss)
    opt_dim = num_dims[opt_index]
    return opt_dim 