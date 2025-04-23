import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import griddata
from sklearn.metrics import roc_curve, auc

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_svm_weights_stem(weights, title="SVM Weights", filename="stem_plot.png", output_dir="output/plots", top_n=6):
    ensure_dir(output_dir)
    save_path = os.path.join(output_dir, filename)

    if weights is None:
        print(f"Info: Skipping stem plot '{title}' (weights not available, likely non-linear kernel).")
        return
    n_channels = len(weights)
    channel_indices = np.arange(1, n_channels + 1)
    dominant_indices = np.argsort(np.abs(weights))[-top_n:]

    plt.figure(figsize=(12, 6))
    markerline, stemlines, baseline = plt.stem(channel_indices, weights, linefmt='grey', markerfmt='o', basefmt='r-', label='_nolegend_')
    plt.setp(markerline, markersize=4, markerfacecolor='grey', markeredgecolor='black')
    # plt.setp(stemlines, 'color', 'grey', 'linewidth', 1) # Other stem styling

    for idx in dominant_indices:
        plt.stem(channel_indices[idx], weights[idx], linefmt='b-', markerfmt='bo', basefmt=' ') # Use basefmt=' ' to avoid drawing baseline again
        plt.text(channel_indices[idx], weights[idx] + 0.05 * np.sign(weights[idx]) + 0.01, # Add small offset
                 f'{weights[idx]:.2f}', ha='center', va='bottom', color='blue', fontsize=9)

    dominant_proxy = plt.Line2D([0], [0], linestyle='none', c='b', marker='o', markersize=5, label=f'Top {top_n} Dominant Channels')

    plt.xlabel("Channel Index")
    plt.ylabel("SVM Weight")
    plt.title(title)
    plt.xlim(0, n_channels + 1)
    plt.legend(handles=[dominant_proxy])
    plt.grid(True, axis='y', linestyle=':')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Stem plot saved to {save_path}")
    plt.show()
    plt.close()


def plot_weights_on_brain(weights, sensor_x, sensor_y, title="Weight Magnitude", filename="brain_plot.png", output_dir="output/plots", grid_resolution=100):
    ensure_dir(output_dir)
    save_path = os.path.join(output_dir, filename)

    if weights is None:
        print(f"Info: Skipping brain plot '{title}' (weights not available, likely non-linear kernel).")
        return
    # Calculate magnitude per electrode location (sqrt(Ex^2 + Ey^2))
    electrode_magnitudes = np.sqrt(weights[0::2]**2 + weights[1::2]**2)
    xi = np.linspace(sensor_x.min() - 0.5, sensor_x.max() + 0.5, grid_resolution)
    yi = np.linspace(sensor_y.min() - 0.5, sensor_y.max() + 0.5, grid_resolution)
    xi, yi = np.meshgrid(xi, yi)
    valid_sensors = ~np.isnan(sensor_x) & ~np.isnan(sensor_y) & ~np.isnan(electrode_magnitudes) # Some error handling
    zi = griddata((sensor_x[valid_sensors], sensor_y[valid_sensors]),
                    electrode_magnitudes[valid_sensors],
                    (xi, yi), method='cubic')

    plt.figure(figsize=(7, 6))
    contour = plt.contourf(xi, yi, zi, levels=15, cmap=plt.cm.viridis, extend='both')
    plt.colorbar(contour, label='SVM Weight Magnitude')
    # Maybe overlay sensor stuff
    # plt.scatter(sensor_x, sensor_y, c='red', s=5, alpha=0.5, label='Sensors')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Brain plot saved to {save_path}")
    plt.show()
    plt.close()


def plot_overall_roc(true_labels, decision_scores, data_label="Data", title="Overall ROC Curve", filename="roc_overall.png", output_dir="output/plots"):
    ensure_dir(output_dir)
    save_path = os.path.join(output_dir, filename)
    fpr, tpr, thresholds = roc_curve(true_labels, decision_scores)
    roc_auc = auc(fpr, tpr)
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
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_individual_and_overall_roc(outer_fold_true_labels, outer_fold_scores,
                                    all_true_labels, all_decision_scores,
                                    data_label="Data", title="Individual & Overall ROC", filename="roc_individual.png", output_dir="output/plots"):
    ensure_dir(output_dir)
    save_path = os.path.join(output_dir, filename)
    plt.figure(figsize=(9, 7))
    n_folds = len(outer_fold_true_labels)

    # Plot individual fold ROCs
    print(f"Plotting individual ROCs for {n_folds} folds...")
    for i in range(n_folds):
        fold_labels = outer_fold_true_labels[i]
        fold_scores = outer_fold_scores[i]
        if fold_labels is not None and fold_scores is not None and len(fold_labels) > 0 and len(fold_scores) > 0:
             try:
                 fpr, tpr, _ = roc_curve(fold_labels, fold_scores)
                 fold_auc = auc(fpr, tpr)
                 plt.plot(fpr, tpr, lw=1, alpha=0.4, label=f'Fold {i+1} (AUC = {fold_auc:.2f})')
             except Exception as e:
                 print(f"Warning: Could not calculate/plot ROC for Fold {i+1}. Error: {e}")
        else:
             print(f"Warning: Insufficient data for Fold {i+1} in individual ROC plot.")

    # Plot overall aggregated ROC curve
    fpr_all, tpr_all, _ = roc_curve(all_true_labels, all_decision_scores)
    roc_auc_all = auc(fpr_all, tpr_all)
    plt.plot(fpr_all, tpr_all, color='b', lw=2.5, alpha=0.9, label=f'Overall ROC ({data_label}, AUC = {roc_auc_all:.2f})')


    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')

    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()