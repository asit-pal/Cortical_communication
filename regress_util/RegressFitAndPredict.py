from .regress_predict import regress_predict

def regress_fit_and_predict(regress_fun, Y_train, X_train, Y_test, X_test, dim, **kwargs):
    """
    Fit regression model and predict test data.
    
    Parameters:
    -----------
    regress_fun : function
        The regression function to use (e.g., reduced_rank_regress).
    Y_train : ndarray, shape (N_train, K)
        Training target data.
    X_train : ndarray, shape (N_train, p)
        Training source data.
    Y_test : ndarray, shape (N_test, K)
        Testing target data.
    X_test : ndarray, shape (N_test, p)
        Testing source data.
    dim : int or array-like
        Number(s) of predictive dimensions to test.
    **kwargs : dict
        Additional arguments to pass to the regression function.
    
    Returns:
    --------
    loss : float or ndarray
        Loss incurred when predicting Y_test using X_test and the fitted model.
    """
    # Extract 'LossMeasure' from kwargs, default to 'NSE' if not provided
    loss_measure = kwargs.pop('LossMeasure', 'NSE')
    
    # Get the extended mapping matrix including intercept
    B, _, _ = regress_fun(Y_train, X_train, dim=dim, **kwargs)
    
    # Predict and compute loss
    loss, _ = regress_predict(Y_test, X_test, B, loss_measure=loss_measure)
    
    return loss 