# main.py
import os
import json
import numpy as np
from src.data_loader import load_feature_data, load_sensor_locations
from src.analysis import perform_two_level_cv, perform_cross_training
from src.plotting import (plot_svm_weights_stem, plot_weights_on_brain,
                          plot_overall_roc, plot_individual_and_overall_roc,
                          ensure_dir)

print("\n=== Starting BCI Analysis Workflow ===")

# Config stuff
# Base directory for data relative to this script location
BASE_DATA_DIR = "assets"
BASE_OUTPUT_DIR = "output"
PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, "results")

# Ensure output directories exist
ensure_dir(PLOTS_DIR)
ensure_dir(RESULTS_DIR)

OVERT_FILE_1 = os.path.join(BASE_DATA_DIR, 'feaSubEOvert_1.csv')
OVERT_FILE_2 = os.path.join(BASE_DATA_DIR, 'feaSubEOvert_2.csv')
IMG_FILE_1 = os.path.join(BASE_DATA_DIR, 'feaSubEImg_1.csv')
IMG_FILE_2 = os.path.join(BASE_DATA_DIR, 'feaSubEImg_2.csv')
SENSOR_FILE = os.path.join(BASE_DATA_DIR, 'BCIsensor_xy.csv')

# Analysis Flags
RUN_SAME_TRAIN = True
RUN_CROSS_TRAIN = True
RUN_RBF_KERNEL_EXAMPLE = True

def save_results(results, filename, results_dir=RESULTS_DIR):
    ensure_dir(results_dir)
    save_path = os.path.join(results_dir, filename)
    # Convert numpy arrays to lists for JSON compatibility
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, list) and all(isinstance(i, np.ndarray) for i in value):
             serializable_results[key] = [arr.tolist() for arr in value]
        elif isinstance(value, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
             serializable_results[key] = int(value)
        elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
             serializable_results[key] = float(value)
        elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
             serializable_results[key] = value
        else:
             serializable_results[key] = repr(value)

    try:
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"Results summary saved to {save_path}")
    except TypeError as e:
        print(f"Error saving results to JSON: {e}. Some data types might not be serializable.")
    except Exception as e:
        print(f"Error saving results: {e}")


# --- Load Sensor Locations ---
sensor_x, sensor_y = load_sensor_locations(SENSOR_FILE)

# --- Load Feature Data ---
X_overt, y_overt = load_feature_data(OVERT_FILE_1, OVERT_FILE_2)
X_img, y_img = load_feature_data(IMG_FILE_1, IMG_FILE_2)

# === Run Analyses ===

# --- Scenario 1 & 2: Same-Train (Linear Kernel) ---
if RUN_SAME_TRAIN:
    # Overt Linear
    print("\n" + "="*30 + "\n Scenario: Same-Train Overt (Linear Kernel)\n" + "="*30)
    if X_overt is not None:
        overt_results_lin = perform_two_level_cv(X_overt, y_overt, data_label="Overt", kernel='linear')
        if overt_results_lin:
            save_results(overt_results_lin, "results_overt_linear_2level.json")
            plot_svm_weights_stem(overt_results_lin.get('fold1_coefficients'), title="Overt (Linear): SVM Weights (Fold 1)", filename="overt_lin_weights_stem.png", output_dir=PLOTS_DIR)
            plot_weights_on_brain(overt_results_lin.get('fold1_coefficients'), sensor_x, sensor_y, title="Overt (Linear): Weight Magnitude (Fold 1)", filename="overt_lin_weights_brain.png", output_dir=PLOTS_DIR)
            plot_individual_and_overall_roc(overt_results_lin.get('outer_fold_true_labels'), overt_results_lin.get('outer_fold_scores'), overt_results_lin.get('all_true_labels'), overt_results_lin.get('all_decision_scores'), data_label="Overt (Linear)", title="Overt (Linear): Individual & Overall ROC", filename="overt_lin_roc_individual.png", output_dir=PLOTS_DIR)

    # Imagined Linear
    print("\n" + "="*30 + "\n Scenario: Same-Train Imagined (Linear Kernel)\n" + "="*30)
    if X_img is not None:
         img_results_lin = perform_two_level_cv(X_img, y_img, data_label="Imagined", kernel='linear')
         if img_results_lin:
            save_results(img_results_lin, "results_imagined_linear_2level.json")
            plot_svm_weights_stem(img_results_lin.get('fold1_coefficients'), title="Imagined (Linear): SVM Weights (Fold 1)", filename="img_lin_weights_stem.png", output_dir=PLOTS_DIR)
            plot_weights_on_brain(img_results_lin.get('fold1_coefficients'), sensor_x, sensor_y, title="Imagined (Linear): Weight Magnitude (Fold 1)", filename="img_lin_weights_brain.png", output_dir=PLOTS_DIR)
            plot_individual_and_overall_roc(img_results_lin.get('outer_fold_true_labels'), img_results_lin.get('outer_fold_scores'), img_results_lin.get('all_true_labels'), img_results_lin.get('all_decision_scores'), data_label="Imagined (Linear)", title="Imagined (Linear): Individual & Overall ROC", filename="img_lin_roc_individual.png", output_dir=PLOTS_DIR)

# --- Scenario 3 & 4: Cross-Train (Linear Kernel) ---
if RUN_CROSS_TRAIN:
    # Overt -> Imagined Linear
    print("\n" + "="*30 + "\n Scenario: Cross-Train Overt -> Imagined (Linear Kernel)\n" + "="*30)
    if X_overt is not None and X_img is not None:
        xt_ov_img_lin = perform_cross_training(X_overt, y_overt, X_img, y_img, "Overt", "Imagined", kernel='linear')
        if xt_ov_img_lin:
             save_results(xt_ov_img_lin, "results_overt_img_linear_crosstrain.json")
             plot_overall_roc(xt_ov_img_lin.get('true_labels'), xt_ov_img_lin.get('decision_scores'), data_label="Train Overt, Test Imagined (Linear)", title="Cross-Train ROC: Overt -> Imagined (Linear)", filename="xt_ov_img_lin_roc.png", output_dir=PLOTS_DIR)

    # Imagined -> Overt Linear
    print("\n" + "="*30 + "\n Scenario: Cross-Train Imagined -> Overt (Linear Kernel)\n" + "="*30)
    if X_overt is not None and X_img is not None:
         xt_img_ov_lin = perform_cross_training(X_img, y_img, X_overt, y_overt, "Imagined", "Overt", kernel='linear')
         if xt_img_ov_lin:
             save_results(xt_img_ov_lin, "results_imagined_overt_linear_crosstrain.json")
             plot_overall_roc(xt_img_ov_lin.get('true_labels'), xt_img_ov_lin.get('decision_scores'), data_label="Train Imagined, Test Overt (Linear)", title="Cross-Train ROC: Imagined -> Overt (Linear)", filename="xt_img_ov_lin_roc.png", output_dir=PLOTS_DIR)


# Same-Train with RBF Kernel ---
if RUN_RBF_KERNEL_EXAMPLE:
    print("\n" + "="*30 + "\n Scenario: Same-Train Overt (RBF Kernel Example)\n" + "="*30)
    if X_overt is not None:
        overt_results_rbf = perform_two_level_cv(X_overt, y_overt, data_label="Overt", kernel='rbf', gamma='scale') # Using default gamma, tuning C
        if overt_results_rbf:
            save_results(overt_results_rbf, "results_overt_rbf_2level.json")
            # Weight plots will be skipped automatically by the plotting function
            plot_svm_weights_stem(overt_results_rbf.get('fold1_coefficients'), title="Overt (RBF): SVM Weights (Fold 1)", filename="overt_rbf_weights_stem.png", output_dir=PLOTS_DIR)
            plot_weights_on_brain(overt_results_rbf.get('fold1_coefficients'), sensor_x, sensor_y, title="Overt (RBF): Weight Magnitude (Fold 1)", filename="overt_rbf_weights_brain.png", output_dir=PLOTS_DIR)
            plot_individual_and_overall_roc(overt_results_rbf.get('outer_fold_true_labels'), overt_results_rbf.get('outer_fold_scores'), overt_results_rbf.get('all_true_labels'), overt_results_rbf.get('all_decision_scores'), data_label="Overt (RBF)", title="Overt (RBF): Individual & Overall ROC", filename="overt_rbf_roc_individual.png", output_dir=PLOTS_DIR)


print("\n=== BCI Analysis Workflow Finished ===")
print(f"Outputs saved in '{BASE_OUTPUT_DIR}' directory.")