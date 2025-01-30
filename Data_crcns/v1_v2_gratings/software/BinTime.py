import numpy as np
from .SET_CONSTS import TIME_RES

def BinTime(spike_rasters, bin_width):
    """
    Bin spike rasters into time windows
    
    Parameters:
    -----------
    spike_rasters : numpy.ndarray
        Array of shape (num_units, trial_length)
    bin_width : int
        Width of time bins in ms
    
    Returns:
    --------
    binned_spikes : numpy.ndarray
        Array of shape (num_units, num_bins)
    """
    if len(spike_rasters.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {spike_rasters.shape}")
        
    num_units, trial_length = spike_rasters.shape
    num_bins = trial_length // bin_width
    
    # Reshape the array to group time points into bins
    reshaped = spike_rasters[:, :num_bins*bin_width].reshape(num_units, num_bins, bin_width)
    
    # Sum within each bin
    binned_spikes = reshaped.sum(axis=2)
    
    return binned_spikes 