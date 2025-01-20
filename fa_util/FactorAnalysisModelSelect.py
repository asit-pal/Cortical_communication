import numpy as np

def factor_analysis_model_select(cv_loss, q):
    """
    Select optimal dimensionality for factor analysis model
    
    Selects the optimal dimensionality based on cumulative shared variance explained.
    The optimal dimensionality is selected as the minimum number of latent dimensions
    necessary to account for 95% of the shared variance, as defined by the factor 
    analysis model for which the cross-validated log-likelihood was highest.
    
    Parameters:
    -----------
    cv_loss : ndarray, shape (numDims,)
        Cumulative shared variance explained by latent dimensions for the FA model
        with highest cross-validated log-likelihood
    q : ndarray, shape (numDims,)
        Vector of latent dimensionalities corresponding to each entry in cv_loss
        
    Returns:
    --------
    q_opt : int
        Optimal dimensionality for the factor analysis model
    """
    
    # Threshold for variance explained (95%)
    # Use VAR_THRESHOLD = 1 - eps to select the latent dimensionality 
    # for which the cross-validated log-likelihood was highest
    VAR_THRESHOLD = 0.95
    
    # Handle case where cv_loss contains NaN values
    if np.isnan(cv_loss).any():
        q_opt = 0
    else:
        # Find first index where 1-cv_loss > VAR_THRESHOLD
        # Equivalent to MATLAB's find(1-cv_loss > VAR_THRESHOLD, 1)
        indices = np.where(1 - cv_loss > VAR_THRESHOLD)[0]
        if len(indices) > 0:
            q_opt = q[indices[0]]
        else:
            # If no dimensions meet the threshold, use the last one
            q_opt = q[-1]
    
    return q_opt 