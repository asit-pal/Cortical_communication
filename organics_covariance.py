import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.model_selection import KFold

# Import custom functions
from regress_methods.ReducedRankRegress import reduced_rank_regress
from regress_util.RegressFitAndPredict import regress_fit_and_predict
from regress_util.model_select import model_select

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.size": 36,  # Set a consistent font size for all text in the plot
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def verify_covariance(samples, true_cov):
    """
    Verify if sampled data produces the correct covariance.

    Parameters:
        samples (np.ndarray): Generated samples from the multivariate Gaussian.
        true_cov (np.ndarray): The true covariance matrix used to generate the samples.

    Returns:
        tuple: (relative error, estimated covariance matrix)
    """
    sample_cov = np.cov(samples.T)
    error = np.linalg.norm(sample_cov - true_cov) / np.linalg.norm(true_cov)
    return error, sample_cov


def perform_rrr_cv(X, Y, num_dims, n_folds=10, loss_measure='NSE'):
    """
    Perform Reduced Rank Regression with cross-validation.

    Parameters:
        X (np.ndarray): Source population activity (n_samples x p).
        Y (np.ndarray): Target population activity (n_samples x K).
        num_dims (np.ndarray): Array of dimensions to test.
        n_folds (int): Number of cross-validation folds.
        loss_measure (str): Loss measure to use ('NSE' by default).

    Returns:
        tuple: (mean_loss, std_error) across cross-validation folds.
    """
    # Initialize K-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Initialize array for storing cross-validation results
    cv_results = np.zeros((len(num_dims), n_folds))

    # Cross-validation loop for RRR
    for i, dims in enumerate(num_dims):
        for j, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Compute loss using regress_fit_and_predict
            loss = regress_fit_and_predict(
                reduced_rank_regress,
                Y_train, X_train,
                Y_test, X_test,
                dim=dims,
                LossMeasure=loss_measure
            )

            cv_results[i, j] = loss

    # Calculate mean and standard error across folds
    mean_loss = np.mean(cv_results, axis=1)
    std_error = np.std(cv_results, axis=1) / np.sqrt(n_folds)

    return mean_loss, std_error


def plot_rrr_results(results, num_dims_used_for_prediction, gammas, line_labelsize=42, legendsize=42):
    """
    Plot RRR prediction performance versus dimensions with customized aesthetics.

    Parameters:
        results (dict): Nested dictionary containing RRR results.
        num_dims_used_for_prediction (np.ndarray): Array of dimensions tested.
        gammas (list): List of gamma values analyzed.
        line_labelsize (int): Font size for axis labels.
        legendsize (int): Font size for legends.
    """
    # Define color scheme for gammas
    color_map = {
        0.5: '#DC143C',  # Crimson
        1.0: '#00BFFF'   # Deep Sky Blue
    }

    # Define markers for targets
    markers = {
        'Y_V1': 's',  # Square
        'Y_V2': 'o'   # Circle
    }

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot performance for each gamma and target
    for gamma in gammas:
        for target in ['Y_V1', 'Y_V2']:
            mean_loss = results[gamma][target]['mean_loss']
            std_error = results[gamma][target]['std_error']
            perf = 1 - mean_loss  # Convert loss to performance

            # Fill between for error regions
            ax.fill_between(
                num_dims_used_for_prediction,
                perf -   std_error,
                perf +   std_error,
                color='gray',
                alpha=0.1
            )

            # Plot performance points
            ax.plot(
                num_dims_used_for_prediction,
                perf,
                marker=markers[target],
                linestyle='None',
                markersize=12,
                color=color_map[gamma],
                markeredgecolor='black',
                # label=f'Gamma={gamma} | {target}'
            )

    # Create handles for gamma legend
    gamma_patches = [
        mpatches.Patch(color=color_map[gamma], label=rf'$\gamma$={gamma}')
        for gamma in gammas
    ]

    # Create handles for target markers
    target_handles = [
        mlines.Line2D([], [], color='black', marker=markers['Y_V1'], linestyle='None',
                      markersize=12, label=r'$\mathrm{V1-V1}$'),
        mlines.Line2D([], [], color='black', marker=markers['Y_V2'], linestyle='None',
                      markersize=12, label=r'$\mathrm{V1-V2}$')
    ]

    # First legend for gammas
    legend1 = ax.legend(handles=gamma_patches, fontsize=legendsize,
                        loc='upper left', frameon=False,
                        handletextpad=0.1, labelspacing=0.15,
                        handlelength=1.0)
    ax.add_artist(legend1)

    # Second legend for targets
    legend2 = ax.legend(handles=target_handles, fontsize=legendsize,
                        loc='upper right', frameon=False,
                        handletextpad=0.1, labelspacing=0.15,
                        handlelength=1.0)
    ax.add_artist(legend2)

    # Set labels with LaTeX formatting
    ax.set_xlabel(r'$\mathrm{Predictive\;Dimensions}$', fontsize=line_labelsize)
    ax.set_ylabel(r'$\mathrm{Prediction\;Performance}$', fontsize=line_labelsize)
    ax.tick_params(axis='both', which='major', labelsize=line_labelsize)

    # Grid and layout
    ax.grid(True)
    fig.tight_layout()

    # Save the plot
    plt.savefig('Figures/communication_subspace_contrast_1.0_N_72_2.pdf', dpi=400, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to process covariance data, perform RRR analyses, and store results.
    """
    # ======================
    # 1) Loading Sample Data
    # ======================

    try:
        cov_data = np.load('mat_sample/Covariance_data_fb_gain_all.npy', allow_pickle=True).item()
    except FileNotFoundError:
        raise FileNotFoundError("The file 'Covariance_data_fb_gain_all.npy' was not found in 'mat_sample' directory.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading covariance data: {e}")

    gammas = [1.0, 0.5]  # Define the gamma values to analyze
    contrast = 1.0

    # Initialize dictionaries to store results
    results = {
        gamma: {
            'Y_V1': {},
            'Y_V2': {}
        } for gamma in gammas
    }

    # Define range of dimensions to test (1-10)
    num_dims_used_for_prediction = np.arange(1, 11)
    cv_num_folds = 10

    for gamma in gammas:
        print(f"===== Processing gamma = {gamma} =====")

        # Load covariance matrix for current gamma
        try:
            cov_matrix = cov_data[gamma, contrast]
        except KeyError:
            raise KeyError(f"Covariance data for gamma={gamma} and contrast={contrast} not found.")

        # Determine the number of neurons from the covariance matrix
        total_neurons = cov_matrix.shape[0]
        mean = np.zeros(total_neurons)

        # Check if covariance matrix is positive definite
        if not np.all(np.linalg.eigvals(cov_matrix) > 0):
            raise ValueError(f"Covariance matrix for gamma={gamma} is not positive definite.")

        n_samples = 4000

        # Generate samples
        try:
            samples = np.random.multivariate_normal(mean, cov_matrix, size=n_samples)
        except ValueError as ve:
            raise ValueError(f"Error in generating multivariate normal samples for gamma={gamma}: {ve}")

        # Verify covariance reconstruction
        cov_error, estimated_cov = verify_covariance(samples, cov_matrix)
        print(f"Covariance reconstruction error for gamma={gamma}: {cov_error:.4f}")

        # Define neuron indices ranges
        # Adjust these ranges based on your actual neuron organization within the covariance matrix
        # For illustration, assuming:
        # V1: first half of neurons
        # V2: second half of neurons
        N = 36
        V1_indices = np.arange(N, 2*N)          
        V2_indices = np.arange(3*N, 4*N) 

        # Define neuron selection sizes
        num_X = 18
        num_Y_V1 = 18
        num_Y_V2 = 18        
        # random_seed = 42

        # Select neurons for X from V1
        # X_indices = np.random.choice(V1_indices, size=num_X, replace=False)
        # X_indices = np.arange(N, 2*N,2)
        # Y_V1_indices = np.arange(N+1, 2*N,2)
        # Y_V2_indices = np.arange(3*N +1, 4*N,2)
        
        # X_indices = np.array([46, 44, 61, 63, 51, 62, 68, 39, 52, 70, 57, 67, 60, 43, 69])
        # Y_V1_indices = np.array([48, 53, 71, 50, 55, 49, 56, 66, 41, 45, 36, 37, 47, 65, 40])
        # Y_V2_indices = np.array([109, 141, 128, 136, 133, 122, 112, 117, 137, 131, 139, 142, 110, 127, 120])


        # Get remaining V1 neurons and select specific number for Y_V1
        X_indices = np.random.choice(V1_indices, size=num_X, replace=False)
        remaining_V1_indices = np.setdiff1d(V1_indices, X_indices)
        Y_V1_indices = np.random.choice(remaining_V1_indices, size=num_Y_V1, replace=False)

        # Select neurons for Y_V2 from V2
        Y_V2_indices = np.random.choice(V2_indices, size=num_Y_V2, replace=False)

        # Assign the neurons to respective variables
        X = samples[:, X_indices]
        Y_V1 = samples[:, Y_V1_indices]
        Y_V2 = samples[:, Y_V2_indices]

        # Print selected indices for verification
        print(f"Selected neuron indices for gamma={gamma}:")
        print(f"X_indices ({len(X_indices)}): {X_indices}")
        print(f"Y_V1_indices ({len(Y_V1_indices)}): {Y_V1_indices}")
        print(f"Y_V2_indices ({len(Y_V2_indices)}): {Y_V2_indices}\n")

        # =====================================
        # 2) Perform RRR for Y_V1 and Y_V2
        # =====================================

        print(f"Performing RRR for Y_V1 as target with gamma={gamma}...")
        mean_loss_Y_V1, std_error_Y_V1 = perform_rrr_cv(
            X, Y_V1, num_dims_used_for_prediction, n_folds=cv_num_folds, loss_measure='NSE'
        )
        print("Completed RRR for Y_V1.\n")

        print(f"Performing RRR for Y_V2 as target with gamma={gamma}...")
        mean_loss_Y_V2, std_error_Y_V2 = perform_rrr_cv(
            X, Y_V2, num_dims_used_for_prediction, n_folds=cv_num_folds, loss_measure='NSE'
        )
        print("Completed RRR for Y_V2.\n")

        # =====================================
        # 3) Find Optimal Dimensionality
        # =====================================

        # For Y_V1
        cv_loss_rrr_Y_V1 = np.vstack([mean_loss_Y_V1, std_error_Y_V1])
        opt_dim_Y_V1 = model_select(cv_loss_rrr_Y_V1, num_dims_used_for_prediction)
        print(f"Optimal number of predictive dimensions for RRR with Y_V1 (gamma={gamma}): {opt_dim_Y_V1}")

        # For Y_V2
        cv_loss_rrr_Y_V2 = np.vstack([mean_loss_Y_V2, std_error_Y_V2])
        opt_dim_Y_V2 = model_select(cv_loss_rrr_Y_V2, num_dims_used_for_prediction)
        print(f"Optimal number of predictive dimensions for RRR with Y_V2 (gamma={gamma}): {opt_dim_Y_V2}\n")

        # Store the results
        results[gamma]['Y_V1']['mean_loss'] = mean_loss_Y_V1
        results[gamma]['Y_V1']['std_error'] = std_error_Y_V1
        results[gamma]['Y_V2']['mean_loss'] = mean_loss_Y_V2
        results[gamma]['Y_V2']['std_error'] = std_error_Y_V2

    return results, num_dims_used_for_prediction, gammas


if __name__ == "__main__":
    # Execute the main function and retrieve results
    results, num_dims_used_for_prediction, gammas = main()

    # Plot the results with the desired aesthetics
    plot_rrr_results(results, num_dims_used_for_prediction, gammas) 
