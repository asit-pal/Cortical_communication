import scipy.io as sio
import numpy as np

def _check_keys(d):
    """Recursively convert mat_struct to nested dict."""
    for key in d:
        if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key])
        elif isinstance(d[key], np.ndarray):
            if d[key].dtype == 'object':
                d[key] = [_todict(e) if isinstance(e, sio.matlab.mio5_params.mat_struct) else e for e in d[key]]
    return d

def _todict(matobj):
    """Recursively convert mat_struct to nested dict."""
    d = {}
    for fieldname in matobj._fieldnames:
        elem = getattr(matobj, fieldname)
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            d[fieldname] = _todict(elem)
        elif isinstance(elem, np.ndarray):
            if elem.dtype == 'object':
                d[fieldname] = [_todict(e) if isinstance(e, sio.matlab.mio5_params.mat_struct) else e for e in elem]
            else:
                d[fieldname] = elem
        else:
            d[fieldname] = elem
    return d

def loadmat(filename):
    """
    Load a MATLAB .mat file and convert mat_structs to nested dictionaries.
    
    Parameters:
    -----------
    filename : str
        Path to the .mat file.
    
    Returns:
    --------
    dict
        Dictionary containing the loaded data with mat_structs converted to dicts.
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    data = _check_keys(data)
    return data
