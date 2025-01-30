import numpy as np
import scipy.io as sio
import scipy.sparse as sp  # Import scipy.sparse
import logging
import os
from loadmat_custom import loadmat  # Import the helper function
import matplotlib.pyplot as plt  # For visualization
from Data_crcns.v1_v2_gratings.software.ExtractSpikes import ExtractSpikes

# Set up logging with less verbose output
logging.basicConfig(level=logging.INFO, format='%(message)s')

def process_area_data(spike_rasters, blank_trials, area_idx):
    """Process spike data for a specific brain area."""
    all_spike_counts = []
    
    for trial_idx in blank_trials:
        spikes = spike_rasters[trial_idx][area_idx]
        if sp.isspmatrix(spikes) and spikes.shape[0] > 0:
            spike_counts = spikes.getnnz(axis=1)  # Count spikes per neuron
            all_spike_counts.append(spike_counts)
    
    return np.array(all_spike_counts)

def save_area_results(spike_counts, area_idx, output_dir, all_centered_rates=None):
    """Save results and create visualization for an area."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to firing rates (Hz) - assuming 100ms bins
    firing_rates = spike_counts * 10
    
    # Subtract mean for each neuron
    mean_rates = np.mean(firing_rates, axis=0, keepdims=True)
    firing_rates_centered = firing_rates - mean_rates
    
    # Save mean-centered firing rates
    output_file = os.path.join(output_dir, f'centered_firing_rates_area_{area_idx}.npy')
    np.save(output_file, firing_rates_centered)
    
    # Store centered rates for combined plot
    if all_centered_rates is not None:
        all_centered_rates[area_idx] = firing_rates_centered
    
    # Calculate statistics
    total_spikes = spike_counts.sum()
    mean_spikes = total_spikes / spike_counts.shape[1]
    num_neurons = spike_counts.shape[1]
    
    # Log results
    logging.info(f"\nArea {area_idx} Statistics:")
    logging.info(f"Shape: {spike_counts.shape} (trials × neurons)")
    logging.info(f"Total spikes: {total_spikes}")
    logging.info(f"Mean spikes per neuron: {mean_spikes:.2f}")
    logging.info(f"Number of neurons: {num_neurons}")

def plot_combined_distributions(all_centered_rates, output_dir):
    """Create combined distribution plot for V1 and V2."""
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red']
    labels = ['V1', 'V2']
    
    for idx in range(2):
        if all_centered_rates[idx] is not None:
            plt.hist(all_centered_rates[idx].flatten(), 
                    bins=100, 
                    color=colors[idx], 
                    alpha=0.5,
                    label=labels[idx],
                    density=True)
    
    plt.title('Distribution of Mean-centered Firing Rates - V1 and V2')
    plt.xlabel('Firing Rate - Mean (Hz)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'combined_centered_firing_rates_distribution.png'))
    plt.close()

def main():
    try:
        # Load the .mat file
        mat_file = '/home/ap6603/communication-subspace/Data_crcns/v1_v2_gratings/mat_neural_data/106r001p26.mat'
        neural_data = loadmat(mat_file)['neuralData']
        
        # Extract spikes with 100ms bins for blank periods
        spikes, stim = ExtractSpikes(neural_data, bin_width=100, trial_period='BLANK')
        
        # Store centered rates for both areas
        all_centered_rates = [None, None]
        
        # Process each area (V1 and V2)
        for area_idx, area_spikes in enumerate(spikes):
            # area_spikes shape: (num_neurons, num_time_bins, num_trials)
            logging.info(f"\nProcessing Area {area_idx}")
            
            # Reshape to (num_trials * num_time_bins, num_neurons)
            num_neurons, num_bins, num_trials = area_spikes.shape
            reshaped_spikes = area_spikes.transpose(0, 2, 1).reshape(num_neurons, -1).T
            
            # Save results
            save_area_results(reshaped_spikes, area_idx, 'Spontaneous_Spike_Data', all_centered_rates)
        
        # Create combined distribution plot
        plot_combined_distributions(all_centered_rates, 'Spontaneous_Spike_Data')
            
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()