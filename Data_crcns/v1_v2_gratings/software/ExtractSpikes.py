import numpy as np
from .SET_CONSTS import DRIVEN_TRIAL_LENGTH, BLANK_TRIAL_LENGTH
from .BinTime import BinTime

def ExtractSpikes(neural_data, bin_width=100, trial_period='Blank'):
    """
    Extract spikes from neural data with specified binning
    
    Parameters:
    -----------
    neural_data : dict
        Dictionary containing spikeRasters and stim
    bin_width : int
        Width of time bins in ms (default: 100ms)
    trial_period : str or list
        Can be 'Driven', 'Blank', 'Full' or a list specifying time range
    
    Returns:
    --------
    spikes : list
        List of numpy arrays containing binned spikes for each area
        Shape: (num_neurons, num_time_bins, num_trials)
    stim : numpy array
        Stimulus values
    """
    # Modified to handle list structure of spikeRasters
    num_trials = len(neural_data['spikeRasters']) // 2
    
    # Determine number of populations from the first trial
    num_pops = len(neural_data['spikeRasters'][0])
    
    # Get number of units for each population
    num_units = []
    for pop_idx in range(num_pops):
        num_units.append(neural_data['spikeRasters'][0][pop_idx].shape[0])
    
    stim = np.zeros(num_trials)
    
    if isinstance(trial_period, str):
        if trial_period.upper() == 'DRIVEN':
            trial_length = DRIVEN_TRIAL_LENGTH
            binned_trial_length = int(trial_length // bin_width)
            
            spikes = []
            for pop_idx in range(num_pops):
                spikes.append(np.zeros((num_units[pop_idx], binned_trial_length, num_trials)))
            
            for trial_idx in range(0, 2*num_trials, 2):
                for pop_idx in range(num_pops):
                    # Convert sparse matrix to dense and bin in 100ms windows
                    spikes[pop_idx][:,:,trial_idx//2] = BinTime(
                        neural_data['spikeRasters'][trial_idx][pop_idx].toarray(), 
                        bin_width
                    )
                stim[trial_idx//2] = neural_data['stim'][trial_idx]
                
        elif trial_period.upper() == 'BLANK':
            trial_length = BLANK_TRIAL_LENGTH
            binned_trial_length = int(trial_length // bin_width)
            
            spikes = []
            for pop_idx in range(num_pops):
                spikes.append(np.zeros((num_units[pop_idx], binned_trial_length, num_trials)))
            
            for trial_idx in range(0, 2*num_trials, 2):
                for pop_idx in range(num_pops):
                    # Convert sparse matrix to dense and bin in 100ms windows
                    spikes[pop_idx][:,:,trial_idx//2] = BinTime(
                        neural_data['spikeRasters'][trial_idx+1][pop_idx].toarray(), 
                        bin_width
                    )
                    
        else:  # 'FULL'
            trial_length = DRIVEN_TRIAL_LENGTH + BLANK_TRIAL_LENGTH
            binned_trial_length = int(trial_length // bin_width)
            
            spikes = []
            for pop_idx in range(num_pops):
                spikes.append(np.zeros((num_units[pop_idx], binned_trial_length, num_trials)))
            
            for trial_idx in range(0, 2*num_trials, 2):
                for pop_idx in range(num_pops):
                    # Combine driven and blank periods
                    combined = np.hstack([
                        neural_data['spikeRasters'][trial_idx][pop_idx].toarray(),
                        neural_data['spikeRasters'][trial_idx+1][pop_idx].toarray()
                    ])
                    # Bin in 100ms windows
                    spikes[pop_idx][:,:,trial_idx//2] = BinTime(combined, bin_width)
                stim[trial_idx//2] = neural_data['stim'][trial_idx]
    
    else:  # trial_period is a time range
        trial_length = trial_period[1] - trial_period[0] + 1
        binned_trial_length = int(trial_length // bin_width)
        
        spikes = []
        for pop_idx in range(num_pops):
            spikes.append(np.zeros((num_units[pop_idx], binned_trial_length, num_trials)))
        
        for trial_idx in range(0, 2*num_trials, 2):
            for pop_idx in range(num_pops):
                combined = np.hstack([
                    neural_data['spikeRasters'][trial_idx][pop_idx].toarray(),
                    neural_data['spikeRasters'][trial_idx+1][pop_idx].toarray()
                ])
                # Extract specified time range and bin in 100ms windows
                spikes[pop_idx][:,:,trial_idx//2] = BinTime(
                    combined[:, trial_period[0]-1:trial_period[1]], 
                    bin_width
                )
            stim[trial_idx//2] = neural_data['stim'][trial_idx]
    
    return spikes, stim 