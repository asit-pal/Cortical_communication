import numpy as np
import scipy.io
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os

# Import custom functions from their respective modules
from regress_methods.ReducedRankRegress import reduced_rank_regress
from regress_util.RegressFitAndPredict import regress_fit_and_predict
from regress_util.model_select import model_select  # Ensure this is implemented

def perform_rrr_cv(X, Y, num_dims, n_folds=1, loss_measure='NSE'):
    """
    Perform Reduced Rank Regression with cross-validation.

    Parameters:
        X (np.ndarray): Source population activity (n_samples x p).
        Y (np.ndarray): Target population activity (n_samples x K).
        num_dims (np.ndarray): Array of dimensions to test.
        n_folds (int): Number of cross-validation folds.
        loss_measure (str): Loss measure to use ('NSE' by default).

    Returns:
        tuple: (mean_loss, std_dev) across cross-validation folds.
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
    
    # Calculate mean and standard deviation (not standard error) across folds
    mean_loss = np.mean(cv_results, axis=1)
    std_dev = np.std(cv_results, axis=1)  # Removed division by sqrt(n_folds)
    
    return mean_loss, std_dev

def main():
    # ======================
    # 1) Loading and Setup
    # ======================
    
    # # Load the sample data
    # data = scipy.io.loadmat('mat_sample/sample_data.mat')
    
    # data_X = data['X']         # Source population activity (n_samples x p matrix)
    # Y_V2_data = data['Y_V2']    # Target population activity (n_samples x K matrix)
    # Y_V1_data = data['Y_V1']    # Target population activity (n_samples x K matrix)
    
    # Load the centered firing rates
    data_X = np.load('Spontaneous_Spike_Data/centered_firing_rates_area_0.npy')
    Y_V2_data = np.load('Spontaneous_Spike_Data/centered_firing_rates_area_1.npy')
    
    # Initialize parameters
    num_iterations = 100  # Number of iterations for averaging
    V1_indices = np.arange(0, 109)
    V2_indices = np.arange(0, 30)
    num_dims_used_for_prediction = np.arange(1, 11)
    cv_num_folds = 2
    
    s_V1 = 30
    t_V1 = 30
    t_V2 = 30

    # Initialize dictionary to store aggregated results
    aggregated_results = {
        'Y_V1': {
            'mean_loss_runs': [],
            'std_dev_runs': []
        },
        'Y_V2': {
            'mean_loss_runs': [],
            'std_dev_runs': []
        }
    }

    print(f"Starting {num_iterations} iterations of RRR analysis...")
    
    # Perform multiple iterations
    for iteration in range(1, num_iterations + 1):
        print(f"  Iteration {iteration}/{num_iterations}")
        
        # Random selection of indices for each iteration
        X_indices = np.random.choice(V1_indices, size=s_V1, replace=False)
        remaining_V1_indices = np.setdiff1d(V1_indices, X_indices)
        Y_V1_indices = np.random.choice(remaining_V1_indices, size=t_V1, replace=False)
        Y_V2_indices = np.random.choice(V2_indices, size=t_V2, replace=False)
        
        # Extract data for selected indices
        X = data_X[4000:8000, X_indices]
        Y_V1 = data_X[4000:8000, Y_V1_indices]
        Y_V2 = Y_V2_data[4000:8000, Y_V2_indices]

        # =====================================
        # 2) Perform RRR for Y_V1 and Y_V2
        # =====================================
        
        mean_loss_Y_V1, std_dev_Y_V1 = perform_rrr_cv(
            X, Y_V1, num_dims_used_for_prediction, n_folds=cv_num_folds, loss_measure='NSE'
        )
        
        mean_loss_Y_V2, std_dev_Y_V2 = perform_rrr_cv(
            X, Y_V2, num_dims_used_for_prediction, n_folds=cv_num_folds, loss_measure='NSE'
        )

        # Store results for this iteration
        aggregated_results['Y_V1']['mean_loss_runs'].append(mean_loss_Y_V1)
        aggregated_results['Y_V1']['std_dev_runs'].append(std_dev_Y_V1)
        aggregated_results['Y_V2']['mean_loss_runs'].append(mean_loss_Y_V2)
        aggregated_results['Y_V2']['std_dev_runs'].append(std_dev_Y_V2)

    # =====================================
    # 3) Compute Final Averaged Results
    # =====================================
    
    # Calculate mean and standard deviation across all iterations
    final_results = {
        'Y_V1': {
            'mean_loss': np.mean(aggregated_results['Y_V1']['mean_loss_runs'], axis=0),
            'std_dev': np.std(aggregated_results['Y_V1']['mean_loss_runs'], axis=0)  # Removed division by sqrt(num_iterations)
        },
        'Y_V2': {
            'mean_loss': np.mean(aggregated_results['Y_V2']['mean_loss_runs'], axis=0),
            'std_dev': np.std(aggregated_results['Y_V2']['mean_loss_runs'], axis=0)  # Removed division by sqrt(num_iterations)
        }
    }

    # =====================================
    # 4) Find Optimal Dimensionality
    # =====================================
    
    # For Y_V1
    # cv_loss_rrr_Y_V1 = np.vstack([final_results['Y_V1']['mean_loss'], 
    #                              final_results['Y_V1']['std_dev']])
    # opt_dim_Y_V1 = model_select(cv_loss_rrr_Y_V1, num_dims_used_for_prediction)
    # print(f"\nOptimal number of predictive dimensions for RRR with Y_V1: {opt_dim_Y_V1}")
    
    # For Y_V2
    # cv_loss_rrr_Y_V2 = np.vstack([final_results['Y_V2']['mean_loss'], 
    #                              final_results['Y_V2']['std_dev']])
    # opt_dim_Y_V2 = model_select(cv_loss_rrr_Y_V2, num_dims_used_for_prediction)
    # print(f"Optimal number of predictive dimensions for RRR with Y_V2: {opt_dim_Y_V2}")

    # =====================================
    # 5) Plotting RRR Cross-validation Results
    # =====================================
    
    # Plotting with fill_between for standard deviation bands
    plt.figure(figsize=(10, 6))
    
    # Plot RRR results for Y_V1
    plt.fill_between(
        num_dims_used_for_prediction,
        1 - final_results['Y_V1']['mean_loss'] - 0.5*final_results['Y_V1']['std_dev'],
        1 - final_results['Y_V1']['mean_loss'] + 0.5*final_results['Y_V1']['std_dev'],
        color='gray', 
        alpha=0.1
    )
    plt.plot(
        num_dims_used_for_prediction,
        1 - final_results['Y_V1']['mean_loss'],
        marker='o',
        linestyle='--',
        linewidth=2,
        color='blue',
        markeredgecolor='black',
        label='RRR with Y_V1',
        markersize=8
    )
    
    # Plot RRR results for Y_V2
    plt.fill_between(
        num_dims_used_for_prediction,
        1 - final_results['Y_V2']['mean_loss'] - 0.5*final_results['Y_V2']['std_dev'],
        1 - final_results['Y_V2']['mean_loss'] + 0.5*final_results['Y_V2']['std_dev'],
        color='gray', 
        alpha=0.1
    )
    plt.plot(
        num_dims_used_for_prediction,
        1 - final_results['Y_V2']['mean_loss'],
        marker='s',
        linestyle='--',
        linewidth=2,
        color='green',
        markeredgecolor='black',
        label='RRR with Y_V2',
        markersize=8
    )
    
    plt.xlabel('Number of Predictive Dimensions')
    plt.ylabel('Predictive Performance (1 - NSE)')
    plt.title('RRR Results for Y_V1 and Y_V2 as Targets')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.savefig('Figures/Spontaneous/Spontaneous_Spike_Data_30_30_30_4-8k_100_iter.pdf', dpi=400, bbox_inches='tight')
    plt.show()

    # Save the final results dictionary
    np.save('Spontaneous_Spike_Data/Spontaneous_30_30_30_4-8k_100_iter_std_dev.npy', final_results)

    return final_results, num_dims_used_for_prediction

if __name__ == "__main__":
    final_results, num_dims_used_for_prediction = main() 