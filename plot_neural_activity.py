import numpy as np
import matplotlib.pyplot as plt
import os

def plot_neural_distributions(data_dir='Spontaneous_Spike_Data'):
    """Plot distributions of mean-centered firing rates for V1 and V2."""
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Load and plot data for both areas
    areas = ['V1', 'V2']
    colors = ['blue', 'red']
    
    for area_idx, (ax, area) in enumerate(zip([ax1, ax2], areas)):
        file_path = os.path.join(data_dir, f'centered_firing_rates_area_{area_idx}.npy')
        if os.path.exists(file_path):
            # Load centered firing rates
            firing_rates = np.load(file_path)
            
            # Create histogram
            counts, bins, _ = ax.hist(firing_rates.flatten(), 
                                    bins=100,
                                    color=colors[area_idx],
                                    alpha=0.6,
                                    density=True)
            
            # Add smoothed line (approximating KDE)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.plot(bin_centers, counts, 
                   color='black',
                   linewidth=2)
            
            ax.set_title(f'{area} Neural Activity Distribution (Mean-centered)')
            ax.set_xlabel('Firing Rate - Mean (Hz)')
            ax.set_ylabel('Density')
            
            # Add vertical line at x=0
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            
            # Print some statistics
            print(f"\n{area} Statistics:")
            print(f"Mean: {np.mean(firing_rates):.3f}")
            print(f"Std: {np.std(firing_rates):.3f}")
            print(f"Range: [{np.min(firing_rates):.3f}, {np.max(firing_rates):.3f}]")
    
    plt.tight_layout()
    os.makedirs('Figures', exist_ok=True)  # Create Figures directory if it doesn't exist
    plt.savefig('Figures/firing_rate_distributions.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_neural_distributions() 