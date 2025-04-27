# plotting.py
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import griddata
from sklearn.metrics import roc_curve, auc

def ensure_dir(directory):
    """Creates the directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def plot_svm_weights_stem(weights, title="SVM Weights", filename="stem_plot.png", output_dir="output/plots", top_n=6):
    """Generates and saves a stem plot of SVM weights, highlighting top N absolute values."""
    ensure_dir(output_dir)
    save_path = os.path.join(output_dir, filename)

    if weights is None:
        print(f"Info: Skipping stem plot '{title}' (weights data is None). File not saved: {save_path}")
        return
    if not isinstance(weights, np.ndarray) or weights.ndim != 1:
        print(f"Error: Weights must be a 1D numpy array for stem plot '{title}'. Got type {type(weights)}.")
        return
    if len(weights) == 0:
        print(f"Error: Empty weights array provided for stem plot '{title}'.")
        return

    n_channels = len(weights)
    channel_indices = np.arange(1, n_channels + 1)

    # Ensure top_n is valid
    actual_top_n = min(top_n, n_channels)
    if actual_top_n < top_n:
        print(f"Warning: Requested top {top_n} channels, but only {n_channels} available. Plotting top {actual_top_n}.")

    # Get indices of channels with top N absolute weights
    if actual_top_n > 0:
        dominant_indices = np.argsort(np.abs(weights))[-actual_top_n:]
    else:
        dominant_indices = [] # Handle case where top_n=0 or no channels

    plt.figure(figsize=(12, 6))
    markerline, stemlines, baseline = plt.stem(channel_indices, weights, linefmt='grey', markerfmt='o', basefmt='r-', label='_nolegend_')
    plt.setp(markerline, markersize=4, markerfacecolor='grey', markeredgecolor='black')

    # Highlight dominant channels and add text labels
    max_abs_weight = np.max(np.abs(weights)) if n_channels > 0 else 1.0 # Avoid division by zero
    for idx in dominant_indices:
        plt.stem(channel_indices[idx], weights[idx], linefmt='b-', markerfmt='bo', basefmt=' ') # Re-plot dominant stems to be blue
        # Adjust text position slightly based on sign and magnitude
        vertical_offset = 0.05 * max_abs_weight * np.sign(weights[idx]) if weights[idx] != 0 else 0.05 * max_abs_weight
        # Ensure minimum offset if weight is near zero
        if abs(vertical_offset) < 0.01: vertical_offset = 0.01 * np.sign(vertical_offset) if vertical_offset != 0 else 0.01

        plt.text(channel_indices[idx], weights[idx] + vertical_offset, f'{weights[idx]:.2f}',
                 ha='center', va='bottom' if weights[idx] >= 0 else 'top', color='blue', fontsize=9)

    # Add legend entry for dominant channels only if some exist
    if actual_top_n > 0:
        dominant_proxy = plt.Line2D([0], [0], linestyle='none', c='b', marker='o', markersize=5, label=f'Top {actual_top_n} Abs. Magnitude Channels')
        plt.legend(handles=[dominant_proxy])
    # No else needed, legend won't show if handles is empty

    plt.xlabel("Channel Index")
    plt.ylabel("SVM Weight / Approx. Weight")
    plt.title(title)
    plt.xlim(0, max(n_channels, 1) + 1) # Ensure xlim is reasonable even for 0/1 channel
    plt.grid(True, axis='y', linestyle=':')
    plt.tight_layout()

    try:
        plt.savefig(save_path)
        print(f"Stem plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving stem plot to {save_path}: {e}")
    plt.show()
    plt.close()


def plot_weights_on_brain(weights, sensor_x, sensor_y, title="Weight Magnitude", filename="brain_plot.png", output_dir="output/plots", grid_resolution=100):
    """Visualizes and saves SVM weight magnitude interpolated on the brain surface."""
    ensure_dir(output_dir)
    save_path = os.path.join(output_dir, filename)

    # --- Input Validation ---
    if weights is None:
        print(f"Info: Skipping brain plot '{title}' (weights data is None). File not saved: {save_path}")
        return
    if not isinstance(weights, np.ndarray) or weights.ndim != 1:
        print(f"Error: Weights must be a 1D numpy array for brain plot '{title}'. Got type {type(weights)}.")
        return
    if len(weights) != 204:
        print(f"Error: Expected 204 weights for brain plot '{title}'. Got {len(weights)}.")
        return
    if sensor_x is None or sensor_y is None:
        print(f"Error: Sensor locations (sensor_x, sensor_y) are required for brain plot '{title}'.")
        return
    if not isinstance(sensor_x, np.ndarray) or not isinstance(sensor_y, np.ndarray) or \
       sensor_x.shape != (102,) or sensor_y.shape != (102,):
        print(f"Error: Sensor locations must be 1D numpy arrays of shape (102,). Got shapes {sensor_x.shape}, {sensor_y.shape} for plot '{title}'.")
        return

    # --- Calculation ---
    try:
        # Calculate magnitude per electrode location (sqrt(Ex^2 + Ey^2))
        # Assumes weights are ordered [ch1_x, ch1_y, ch2_x, ch2_y, ...]
        electrode_magnitudes = np.sqrt(weights[0::2]**2 + weights[1::2]**2)
    except IndexError:
         print(f"Error: Could not pair weights for magnitude calculation in brain plot '{title}'. Check weight vector length (needs to be even).")
         return
    if len(electrode_magnitudes) != 102:
        print(f"Error: Calculated magnitude array size incorrect ({len(electrode_magnitudes)}). Expected 102 for plot '{title}'.")
        return

    # Create grid for interpolation
    # Add small buffer to ensure all sensors are within grid boundaries
    xi = np.linspace(np.nanmin(sensor_x) - 0.5, np.nanmax(sensor_x) + 0.5, grid_resolution)
    yi = np.linspace(np.nanmin(sensor_y) - 0.5, np.nanmax(sensor_y) + 0.5, grid_resolution)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate magnitudes onto the grid - handle potential NaNs in sensor data
    valid_sensors = ~np.isnan(sensor_x) & ~np.isnan(sensor_y) & ~np.isnan(electrode_magnitudes)
    if not np.any(valid_sensors):
        print(f"Error: No valid sensor locations or magnitudes found for interpolation in plot '{title}'.")
        return

    zi = griddata((sensor_x[valid_sensors], sensor_y[valid_sensors]),
                  electrode_magnitudes[valid_sensors],
                  (xi, yi), method='cubic') # Options: 'linear', 'nearest', 'cubic'

    # --- Plotting ---
    plt.figure(figsize=(7, 6))
    # Use contourf for filled contours. Handle cases where zi might be all NaN.
    if not np.all(np.isnan(zi)):
        contour = plt.contourf(xi, yi, zi, levels=15, cmap=plt.cm.viridis, extend='both') # levels=15 is arbitrary
        plt.colorbar(contour, label='SVM Weight Magnitude (Approx. for Non-Linear)')
    else:
        print(f"Warning: Interpolated data (zi) is all NaN for plot '{title}'. Skipping contour plot.")

    # Optional: Overlay sensor locations
    # plt.scatter(sensor_x[valid_sensors], sensor_y[valid_sensors], c='red', s=10, alpha=0.6, label='Sensor Locations')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off') # Hide axes for a cleaner look
    plt.title(title)
    plt.tight_layout()

    try:
        plt.savefig(save_path)
        print(f"Brain plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving brain plot to {save_path}: {e}")
    plt.show()
    plt.close()

def plot_overall_roc(true_labels, decision_scores, data_label="Data", title="Overall ROC Curve", filename="roc_overall.png", output_dir="output/plots"):
    """Plots and saves a single overall ROC curve."""
    ensure_dir(output_dir)
    save_path = os.path.join(output_dir, filename)

    # --- Input Validation ---
    if true_labels is None or decision_scores is None:
        print(f"Error: Missing true_labels or decision_scores for Overall ROC plot '{title}'. File not saved: {save_path}")
        return
    if not isinstance(true_labels, np.ndarray) or not isinstance(decision_scores, np.ndarray):
         print(f"Error: true_labels and decision_scores must be numpy arrays for Overall ROC plot '{title}'.")
         return
    if true_labels.shape != decision_scores.shape:
         print(f"Error: Mismatched shapes for true_labels ({true_labels.shape}) and decision_scores ({decision_scores.shape}) for Overall ROC plot '{title}'.")
         return
    if len(true_labels) == 0:
        print(f"Error: Empty data provided for Overall ROC plot '{title}'.")
        return
    if len(np.unique(true_labels)) < 2:
        print(f"Warning: Only one class present in true_labels for Overall ROC plot '{title}'. ROC AUC is not defined.")
        # Optionally, plot anyway but without AUC or skip entirely
        # For now, we'll let roc_curve handle it, which might raise an error or return non-standard values.
        # Consider adding a check here to skip plotting if desired.
        # return # Uncomment to skip if only one class

    # --- Calculation and Plotting ---
    try:
        fpr, tpr, thresholds = roc_curve(true_labels, decision_scores)
        roc_auc = auc(fpr, tpr)
    except ValueError as e:
        print(f"Error calculating ROC curve for Overall ROC '{title}': {e}. Check label inputs.")
        return # Stop if roc_curve fails

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC ({data_label}, AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.50)')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':')
    plt.tight_layout()

    try:
        plt.savefig(save_path)
        print(f"Overall ROC plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving Overall ROC plot to {save_path}: {e}")
    plt.show()
    plt.close()


def plot_individual_and_overall_roc(outer_fold_true_labels, outer_fold_scores,
                                    all_true_labels, all_decision_scores,
                                    data_label="Data", title="Individual & Overall ROC", filename="roc_individual.png", output_dir="output/plots"):
    """Plots and saves individual fold ROC curves and the aggregated overall ROC curve."""
    ensure_dir(output_dir)
    save_path = os.path.join(output_dir, filename)

    # --- Input Validation ---
    if not isinstance(outer_fold_true_labels, list) or not isinstance(outer_fold_scores, list) or \
       len(outer_fold_true_labels) != len(outer_fold_scores):
        print(f"Error: Invalid or mismatched per-fold data (must be lists of same length) for Individual ROC plot '{title}'. File not saved: {save_path}")
        return
    if all_true_labels is None or all_decision_scores is None:
         print(f"Error: Missing aggregated data (all_true_labels or all_decision_scores) for Individual ROC plot '{title}'.")
         return
    if not isinstance(all_true_labels, np.ndarray) or not isinstance(all_decision_scores, np.ndarray):
         print(f"Error: Aggregated data (all_true_labels, all_decision_scores) must be numpy arrays for Individual ROC plot '{title}'.")
         return
    if len(all_true_labels) == 0:
         print(f"Warning: Empty aggregated data for Individual ROC plot '{title}'. Overall ROC cannot be plotted.")
         # Proceed to plot individual folds if available

    n_folds = len(outer_fold_true_labels)
    if n_folds == 0 and len(all_true_labels) == 0:
        print(f"Error: No fold data and no aggregated data provided for Individual ROC plot '{title}'. Skipping plot.")
        return


    # --- Plotting ---
    plt.figure(figsize=(9, 7))
    fold_aucs = []

    # Plot individual fold ROCs
    print(f"Plotting individual ROCs for {n_folds} folds...")
    for i in range(n_folds):
        fold_labels = outer_fold_true_labels[i]
        fold_scores = outer_fold_scores[i]

        # Check validity of data for this specific fold
        if fold_labels is not None and fold_scores is not None and \
           isinstance(fold_labels, np.ndarray) and isinstance(fold_scores, np.ndarray) and \
           len(fold_labels) > 0 and len(fold_scores) > 0 and \
           fold_labels.shape == fold_scores.shape and \
           len(np.unique(fold_labels)) > 1: # Need at least two classes
             try:
                 fpr, tpr, _ = roc_curve(fold_labels, fold_scores)
                 fold_auc = auc(fpr, tpr)
                 fold_aucs.append(fold_auc)
                 plt.plot(fpr, tpr, lw=1, alpha=0.4, label=f'Fold {i+1} (AUC = {fold_auc:.2f})')
             except ValueError as e:
                 print(f"Warning: Could not calculate/plot ROC for Fold {i+1}. Check labels. Error: {e}")
             except Exception as e:
                 print(f"Warning: An unexpected error occurred plotting ROC for Fold {i+1}: {e}")
        else:
             # Provide more specific warnings
             if fold_labels is None or fold_scores is None: reason = "data is None"
             elif not isinstance(fold_labels, np.ndarray) or not isinstance(fold_scores, np.ndarray): reason = "data not numpy arrays"
             elif len(fold_labels) == 0: reason = "empty data"
             elif fold_labels.shape != fold_scores.shape: reason = "mismatched shapes"
             elif len(np.unique(fold_labels)) < 2: reason = "only one class present"
             else: reason = "unknown issue"
             print(f"Warning: Skipping ROC for Fold {i+1} due to invalid data ({reason}).")

    # Plot overall aggregated ROC curve (only if data is valid)
    overall_roc_plotted = False
    if len(all_true_labels) > 0 and all_true_labels.shape == all_decision_scores.shape and \
       len(np.unique(all_true_labels)) > 1:
            try:
                fpr_all, tpr_all, _ = roc_curve(all_true_labels, all_decision_scores)
                roc_auc_all = auc(fpr_all, tpr_all)
                plt.plot(fpr_all, tpr_all, color='b', lw=2.5, alpha=0.9,
                         label=f'Overall ROC ({data_label}, AUC = {roc_auc_all:.2f})')
                overall_roc_plotted = True
            except ValueError as e:
                 print(f"Warning: Could not calculate/plot Overall ROC. Check aggregated labels. Error: {e}")
            except Exception as e:
                 print(f"Warning: An unexpected error occurred plotting Overall ROC: {e}")
    elif len(all_true_labels) > 0: # Data exists but is invalid for ROC
        reason = "unknown issue"
        if all_true_labels.shape != all_decision_scores.shape: reason = "mismatched shapes"
        elif len(np.unique(all_true_labels)) < 2: reason = "only one class present"
        print(f"Warning: Skipping Overall ROC due to invalid aggregated data ({reason}).")


    # Plot chance line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.50)')

    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)

    # Add mean fold AUC to legend title if available
    legend_title = None
    if fold_aucs:
        mean_fold_auc = np.mean(fold_aucs)
        std_fold_auc = np.std(fold_aucs)
        legend_title = f"Mean Fold AUC: {mean_fold_auc:.2f} ± {std_fold_auc:.2f}"

    # Only show legend if there's something to show (individual folds or overall)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles=handles, labels=labels, loc="lower right", fontsize='small', title=legend_title)

    plt.grid(True, linestyle=':')
    plt.tight_layout()

    try:
        plt.savefig(save_path)
        print(f"Individual ROC plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving Individual ROC plot to {save_path}: {e}")
    plt.show()
    plt.close()