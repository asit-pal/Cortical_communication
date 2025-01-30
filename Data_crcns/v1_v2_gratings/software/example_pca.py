import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from SET_CONSTS import DRIVEN_TRIAL_LENGTH, V1
from ExtractSpikes import ExtractSpikes
import os
import logging

# Set up logging with more detail
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load and pre-process neural data
mat_file = '/home/ap6603/communication-subspace/Data_crcns/v1-v2_gratings/mat_neural_data/107l003p143.mat'
neural_data = sio.loadmat(mat_file)['neuralData']

# Print the available fields in neural_data for debugging
logging.info(f"Available fields in neural_data: {neural_data.dtype.names}")

# Get the spike rasters for V1 (first column, all rows)
spike_rasters_v1 = neural_data['spikeRasters'][0, 0]  # Get all trials for V1
stim_values = neural_data['stim'].flatten()  # Get stimulus values
num_trials = len(stim_values)

logging.info(f"Number of trials: {num_trials}")
logging.info(f"Stimulus values shape: {stim_values.shape}")

# Get number of units (neurons) in V1
num_units_v1 = neural_data['unitCodes'][0, 0].shape[0]
logging.info(f"Number of units in V1: {num_units_v1}")

try:
    # Extract spikes with appropriate binning
    BIN_WIDTH = 1  # 1ms bins
    spikes, stim = ExtractSpikes(neural_data, BIN_WIDTH, trial_period='Driven')
    
    # Add debugging for spikes
    logging.info(f"Number of population arrays: {len(spikes)}")
    for i, spike_array in enumerate(spikes):
        logging.info(f"Shape of spikes[{i}]: {spike_array.shape}")
    
    # Use V1 population (index 0)
    pop_idx = 0
    
    if len(spikes) <= pop_idx:
        raise ValueError(f"Population index {pop_idx} is out of range. Only {len(spikes)} populations available.")
    
    if spikes[pop_idx].size == 0:
        raise ValueError(f"No spike data found for population index {pop_idx}")
    
    # Log shapes at each step
    logging.info(f"Shape of spikes[pop_idx] before transpose: {spikes[pop_idx].shape}")
    
    X = np.transpose(spikes[pop_idx], (2, 0, 1))
    logging.info(f"Shape after transpose: {X.shape}")
    
    X = X.reshape(X.shape[0], -1)
    logging.info(f"Shape after reshape: {X.shape}")

    # Perform PCA
    pca = PCA(n_components=3)
    Z = pca.fit_transform(X)

    # Load color map
    color_map = sio.loadmat('COLOR_MAP.mat')['COLOR_MAP']
    color_map = color_map[::8]

    # Get unique stimulus IDs (excluding blank screens - stimulus ID 0)
    stim_ids = np.unique(stim[stim > 0])
    num_stim = len(stim_ids)

    # Create the plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each stimulus condition
    for stim_idx, stim_id in enumerate(stim_ids):
        mask = stim == stim_id
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            Z[mask, 2],
            c=[color_map[stim_idx]],
            s=100,
            label=f'{(stim_id-1)*22.5}°'  # Convert stim_id to degrees
        )

    ax.set_axis_off()
    ax.set_box_aspect([1,1,1])
    ax.legend()

    # Create Figures directory if it doesn't exist
    os.makedirs('Figures', exist_ok=True)

    # Save the figure
    plt.savefig('Figures/pca_plot_driven.png')
    print('Plot saved as pca_plot_driven.png')

except Exception as e:
    logging.error(f"Error: {e}")
    logging.error(f"Error occurred at line {e.__traceback__.tb_lineno}")
    logging.error(f"Neural data shape: {neural_data['spikeRasters'].shape}")
    raise

# The 'trial_period' option can be set to 'Driven' (trials for which an
# oriented grating was presented), 'Spontaneous' (trials for which no
# grating was presented), 'Full' (combine each driven trial with the
# subsequent spontaneous trial), or a list specifying time range in ms 
# (range must be within 1-2780, where first 1280 ms is driven activity,
# subsequent 1500ms is spontaneous activity) 