# main.py
import os
import json
import numpy as np
import time # Added for timing

# Import functions assuming they are in the 'src' directory
try:
    from src.data_loader import load_feature_data, load_sensor_locations
    from src.analysis import perform_two_level_cv, perform_cross_training
    from src.plotting import (plot_svm_weights_stem, plot_weights_on_brain,
                              plot_overall_roc, plot_individual_and_overall_roc,
                              ensure_dir)
except ImportError as e:
    print(f"Error importing modules from 'src': {e}")
    print("Please ensure data_loader.py, analysis.py, and plotting.py are inside a 'src' directory.")
    exit()


print("\n=== Starting BCI Analysis Workflow ===")
script_start_time = time.time()

# --- Configuration ---
# Base directory for data relative to this script location
BASE_DATA_DIR = "assets"
# Base directory for all outputs
BASE_OUTPUT_DIR = "output"
# Specific directories for plots and results summaries
PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, "results")

# Ensure output directories exist
ensure_dir(PLOTS_DIR)
ensure_dir(RESULTS_DIR)

# Construct full paths to data files
OVERT_FILE_1 = os.path.join(BASE_DATA_DIR, 'feaSubEOvert_1.csv')
OVERT_FILE_2 = os.path.join(BASE_DATA_DIR, 'feaSubEOvert_2.csv')
IMG_FILE_1 = os.path.join(BASE_DATA_DIR, 'feaSubEImg_1.csv')
IMG_FILE_2 = os.path.join(BASE_DATA_DIR, 'feaSubEImg_2.csv')
SENSOR_FILE = os.path.join(BASE_DATA_DIR, 'BCIsensor_xy.csv')

# Check if data files exist before proceeding
required_files = [OVERT_FILE_1, OVERT_FILE_2, IMG_FILE_1, IMG_FILE_2, SENSOR_FILE]
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print("Error: The following required data files were not found:")
    for f in missing_files:
        print(f"- {f}")
    print(f"Please ensure they are in the '{BASE_DATA_DIR}' directory.")
    exit()


# Experiment Control Flags (using more granular flags for clarity)
RUN_LINEAR_KERNEL = True
RUN_RBF_KERNEL = True      # Run RBF on both Overt and Imagined
RUN_POLY_KERNEL = True     # Run Polynomial on both Overt and Imagined
RUN_CROSS_TRAIN_LINEAR = True # Run Linear Cross-Training

# Default Kernel Params & CV Settings (can be overridden in function calls if needed)
DEFAULT_GAMMA = 'scale'
DEFAULT_DEGREE = 3
DEFAULT_C_RANGE = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
OUTER_K = 6 # Number of outer folds
INNER_K = 5 # Number of inner folds for C tuning

# --- Utility Function for Saving Results ---
def save_results(results, filename, results_dir=RESULTS_DIR):
    """Saves the results dictionary to a JSON file, handling numpy types."""
    ensure_dir(results_dir)
    save_path = os.path.join(results_dir, filename)

    if not isinstance(results, dict):
        print(f"Warning: Attempted to save non-dictionary results to {filename}. Skipping.")
        return

    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            # Handle potential NaNs or Infs if they might occur
            if np.isnan(value).any() or np.isinf(value).any():
                 value = np.where(np.isnan(value), None, value) # Replace NaN with None
                 value = np.where(np.isinf(value), None, value) # Replace Inf with None? Or string? Using None.
            serializable_results[key] = value.tolist()
        elif isinstance(value, list) and value and all(isinstance(i, np.ndarray) for i in value):
             # Handle lists of numpy arrays (like fold scores/labels)
             serializable_results[key] = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in value]
        # Handle numpy scalar types explicitly
        elif isinstance(value, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
             serializable_results[key] = int(value)
        elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
             # Handle potential NaN/Inf floats from calculations like nanmean
             if np.isnan(value): serializable_results[key] = None
             elif np.isinf(value): serializable_results[key] = None # Or use '+inf'/'-inf' string?
             else: serializable_results[key] = float(value)
        elif isinstance(value, (np.bool_)):
             serializable_results[key] = bool(value)
        # Keep standard Python types as is
        elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
             serializable_results[key] = value
        else:
             # For other types, store their representation string, with a warning
             print(f"Warning: Data type {type(value)} for key '{key}' in results is not directly JSON serializable. Storing its representation.")
             serializable_results[key] = repr(value)

    try:
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"Results summary saved to {save_path}")
    except TypeError as e:
        # This might catch issues not handled above
        print(f"Error serializing results to JSON for {filename}: {e}. Some data types might not be compatible.")
    except IOError as e:
        print(f"Error writing results file {save_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred saving results to {save_path}: {e}")


# --- Load Sensor Locations ---
print("\n--- Loading Sensor Locations ---")
sensor_x, sensor_y = load_sensor_locations(SENSOR_FILE)
if sensor_x is None or sensor_y is None:
    print("WARNING: Sensor locations failed to load. Brain plots will be skipped.")

# --- Load Feature Data ---
print("\n--- Loading Feature Data ---")
X_overt, y_overt = load_feature_data(OVERT_FILE_1, OVERT_FILE_2, label1=0, label2=1)
X_img, y_img = load_feature_data(IMG_FILE_1, IMG_FILE_2, label1=0, label2=1)

# === Run Analyses ===

# --- Scenario: Same-Train (Linear Kernel) ---
if RUN_LINEAR_KERNEL:
    # Overt Linear
    print("\n" + "="*30 + "\n Scenario: Same-Train Overt (Linear Kernel)\n" + "="*30)
    if X_overt is not None and y_overt is not None:
        overt_results_lin = perform_two_level_cv(X_overt, y_overt, data_label="Overt", kernel='linear',
                                                 C_range=DEFAULT_C_RANGE, outer_k=OUTER_K, inner_k=INNER_K)
        if overt_results_lin:
            save_results(overt_results_lin, "results_overt_linear_2level.json", results_dir=RESULTS_DIR)
            plot_svm_weights_stem(overt_results_lin.get('fold1_coefficients'), title="Overt (Linear): SVM Weights (Fold 1)", filename="overt_lin_weights_stem.png", output_dir=PLOTS_DIR)
            if sensor_x is not None and sensor_y is not None:
                plot_weights_on_brain(overt_results_lin.get('fold1_coefficients'), sensor_x, sensor_y, title="Overt (Linear): Weight Magnitude (Fold 1)", filename="overt_lin_weights_brain.png", output_dir=PLOTS_DIR)
            plot_individual_and_overall_roc(overt_results_lin.get('outer_fold_true_labels'), overt_results_lin.get('outer_fold_scores'), overt_results_lin.get('all_true_labels'), overt_results_lin.get('all_decision_scores'), data_label="Overt (Linear)", title="Overt (Linear): Individual & Overall ROC", filename="overt_lin_roc_individual.png", output_dir=PLOTS_DIR)
    else: print("Skipping Overt (Linear) due to missing data.")

    # Imagined Linear
    print("\n" + "="*30 + "\n Scenario: Same-Train Imagined (Linear Kernel)\n" + "="*30)
    if X_img is not None and y_img is not None:
         img_results_lin = perform_two_level_cv(X_img, y_img, data_label="Imagined", kernel='linear',
                                                C_range=DEFAULT_C_RANGE, outer_k=OUTER_K, inner_k=INNER_K)
         if img_results_lin:
            save_results(img_results_lin, "results_imagined_linear_2level.json", results_dir=RESULTS_DIR)
            plot_svm_weights_stem(img_results_lin.get('fold1_coefficients'), title="Imagined (Linear): SVM Weights (Fold 1)", filename="img_lin_weights_stem.png", output_dir=PLOTS_DIR)
            if sensor_x is not None and sensor_y is not None:
                plot_weights_on_brain(img_results_lin.get('fold1_coefficients'), sensor_x, sensor_y, title="Imagined (Linear): Weight Magnitude (Fold 1)", filename="img_lin_weights_brain.png", output_dir=PLOTS_DIR)
            plot_individual_and_overall_roc(img_results_lin.get('outer_fold_true_labels'), img_results_lin.get('outer_fold_scores'), img_results_lin.get('all_true_labels'), img_results_lin.get('all_decision_scores'), data_label="Imagined (Linear)", title="Imagined (Linear): Individual & Overall ROC", filename="img_lin_roc_individual.png", output_dir=PLOTS_DIR)
    else: print("Skipping Imagined (Linear) due to missing data.")

# --- Scenario: Same-Train (RBF Kernel) ---
if RUN_RBF_KERNEL:
    # Overt RBF
    print("\n" + "="*30 + "\n Scenario: Same-Train Overt (RBF Kernel)\n" + "="*30)
    if X_overt is not None and y_overt is not None:
        overt_results_rbf = perform_two_level_cv(X_overt, y_overt, data_label="Overt", kernel='rbf', gamma=DEFAULT_GAMMA,
                                                 C_range=DEFAULT_C_RANGE, outer_k=OUTER_K, inner_k=INNER_K)
        if overt_results_rbf:
            save_results(overt_results_rbf, "results_overt_rbf_2level.json", results_dir=RESULTS_DIR)
            coeffs_rbf_ov = overt_results_rbf.get('fold1_coefficients')
            note_ov = " (Approximated)" if coeffs_rbf_ov is not None else ""
            plot_svm_weights_stem(coeffs_rbf_ov, title=f"Overt (RBF): SVM Weights{note_ov} (Fold 1)", filename="overt_rbf_weights_stem.png", output_dir=PLOTS_DIR)
            if sensor_x is not None and sensor_y is not None:
                plot_weights_on_brain(coeffs_rbf_ov, sensor_x, sensor_y, title=f"Overt (RBF): Weight Magnitude{note_ov} (Fold 1)", filename="overt_rbf_weights_brain.png", output_dir=PLOTS_DIR)
            plot_individual_and_overall_roc(overt_results_rbf.get('outer_fold_true_labels'), overt_results_rbf.get('outer_fold_scores'), overt_results_rbf.get('all_true_labels'), overt_results_rbf.get('all_decision_scores'), data_label="Overt (RBF)", title="Overt (RBF): Individual & Overall ROC", filename="overt_rbf_roc_individual.png", output_dir=PLOTS_DIR)
    else: print("Skipping Overt (RBF) due to missing data.")

    # Imagined RBF
    print("\n" + "="*30 + "\n Scenario: Same-Train Imagined (RBF Kernel)\n" + "="*30)
    if X_img is not None and y_img is not None:
        img_results_rbf = perform_two_level_cv(X_img, y_img, data_label="Imagined", kernel='rbf', gamma=DEFAULT_GAMMA,
                                               C_range=DEFAULT_C_RANGE, outer_k=OUTER_K, inner_k=INNER_K)
        if img_results_rbf:
            save_results(img_results_rbf, "results_imagined_rbf_2level.json", results_dir=RESULTS_DIR)
            coeffs_rbf_im = img_results_rbf.get('fold1_coefficients')
            note_im = " (Approximated)" if coeffs_rbf_im is not None else ""
            plot_svm_weights_stem(coeffs_rbf_im, title=f"Imagined (RBF): SVM Weights{note_im} (Fold 1)", filename="img_rbf_weights_stem.png", output_dir=PLOTS_DIR)
            if sensor_x is not None and sensor_y is not None:
                plot_weights_on_brain(coeffs_rbf_im, sensor_x, sensor_y, title=f"Imagined (RBF): Weight Magnitude{note_im} (Fold 1)", filename="img_rbf_weights_brain.png", output_dir=PLOTS_DIR)
            plot_individual_and_overall_roc(img_results_rbf.get('outer_fold_true_labels'), img_results_rbf.get('outer_fold_scores'), img_results_rbf.get('all_true_labels'), img_results_rbf.get('all_decision_scores'), data_label="Imagined (RBF)", title="Imagined (RBF): Individual & Overall ROC", filename="img_rbf_roc_individual.png", output_dir=PLOTS_DIR)
    else: print("Skipping Imagined (RBF) due to missing data.")


# --- Scenario: Same-Train (Polynomial Kernel) ---
if RUN_POLY_KERNEL:
    # Overt Poly
    print("\n" + "="*30 + "\n Scenario: Same-Train Overt (Polynomial Kernel)\n" + "="*30)
    if X_overt is not None and y_overt is not None:
        overt_results_poly = perform_two_level_cv(X_overt, y_overt, data_label="Overt", kernel='poly',
                                                  gamma=DEFAULT_GAMMA, degree=DEFAULT_DEGREE,
                                                  C_range=DEFAULT_C_RANGE, outer_k=OUTER_K, inner_k=INNER_K)
        if overt_results_poly:
            save_results(overt_results_poly, "results_overt_poly_2level.json", results_dir=RESULTS_DIR)
            coeffs_poly_ov = overt_results_poly.get('fold1_coefficients')
            note_ov = " (Approximated)" if coeffs_poly_ov is not None else ""
            plot_svm_weights_stem(coeffs_poly_ov, title=f"Overt (Poly d={DEFAULT_DEGREE}): SVM Weights{note_ov} (Fold 1)", filename="overt_poly_weights_stem.png", output_dir=PLOTS_DIR)
            if sensor_x is not None and sensor_y is not None:
                plot_weights_on_brain(coeffs_poly_ov, sensor_x, sensor_y, title=f"Overt (Poly d={DEFAULT_DEGREE}): Weight Magnitude{note_ov} (Fold 1)", filename="overt_poly_weights_brain.png", output_dir=PLOTS_DIR)
            plot_individual_and_overall_roc(overt_results_poly.get('outer_fold_true_labels'), overt_results_poly.get('outer_fold_scores'), overt_results_poly.get('all_true_labels'), overt_results_poly.get('all_decision_scores'), data_label=f"Overt (Poly d={DEFAULT_DEGREE})", title=f"Overt (Poly d={DEFAULT_DEGREE}): Individual & Overall ROC", filename="overt_poly_roc_individual.png", output_dir=PLOTS_DIR)
    else: print("Skipping Overt (Poly) due to missing data.")

    # Imagined Poly
    print("\n" + "="*30 + "\n Scenario: Same-Train Imagined (Polynomial Kernel)\n" + "="*30)
    if X_img is not None and y_img is not None:
        img_results_poly = perform_two_level_cv(X_img, y_img, data_label="Imagined", kernel='poly',
                                                gamma=DEFAULT_GAMMA, degree=DEFAULT_DEGREE,
                                                C_range=DEFAULT_C_RANGE, outer_k=OUTER_K, inner_k=INNER_K)
        if img_results_poly:
            save_results(img_results_poly, "results_imagined_poly_2level.json", results_dir=RESULTS_DIR)
            coeffs_poly_im = img_results_poly.get('fold1_coefficients')
            note_im = " (Approximated)" if coeffs_poly_im is not None else ""
            plot_svm_weights_stem(coeffs_poly_im, title=f"Imagined (Poly d={DEFAULT_DEGREE}): SVM Weights{note_im} (Fold 1)", filename="img_poly_weights_stem.png", output_dir=PLOTS_DIR)
            if sensor_x is not None and sensor_y is not None:
                plot_weights_on_brain(coeffs_poly_im, sensor_x, sensor_y, title=f"Imagined (Poly d={DEFAULT_DEGREE}): Weight Magnitude{note_im} (Fold 1)", filename="img_poly_weights_brain.png", output_dir=PLOTS_DIR)
            plot_individual_and_overall_roc(img_results_poly.get('outer_fold_true_labels'), img_results_poly.get('outer_fold_scores'), img_results_poly.get('all_true_labels'), img_results_poly.get('all_decision_scores'), data_label=f"Imagined (Poly d={DEFAULT_DEGREE})", title=f"Imagined (Poly d={DEFAULT_DEGREE}): Individual & Overall ROC", filename="img_poly_roc_individual.png", output_dir=PLOTS_DIR)
    else: print("Skipping Imagined (Poly) due to missing data.")


# --- Scenario: Cross-Train (Linear Kernel Only) ---
if RUN_CROSS_TRAIN_LINEAR:
    # Overt -> Imagined Linear
    print("\n" + "="*30 + "\n Scenario: Cross-Train Overt -> Imagined (Linear Kernel)\n" + "="*30)
    if X_overt is not None and y_overt is not None and X_img is not None and y_img is not None:
        xt_ov_img_lin = perform_cross_training(X_overt, y_overt, X_img, y_img, "Overt", "Imagined", kernel='linear')
        if xt_ov_img_lin:
             save_results(xt_ov_img_lin, "results_xt_overt_img_linear.json", results_dir=RESULTS_DIR)
             plot_overall_roc(xt_ov_img_lin.get('true_labels'), xt_ov_img_lin.get('decision_scores'),
                              data_label="Train Overt, Test Imagined (Linear)",
                              title="Cross-Train ROC: Overt -> Imagined (Linear)",
                              filename="xt_overt_img_lin_roc.png", output_dir=PLOTS_DIR)
    else: print("Skipping Cross-Train Overt->Imagined due to missing data.")

    # Imagined -> Overt Linear
    print("\n" + "="*30 + "\n Scenario: Cross-Train Imagined -> Overt (Linear Kernel)\n" + "="*30)
    if X_overt is not None and y_overt is not None and X_img is not None and y_img is not None:
         xt_img_ov_lin = perform_cross_training(X_img, y_img, X_overt, y_overt, "Imagined", "Overt", kernel='linear')
         if xt_img_ov_lin:
             save_results(xt_img_ov_lin, "results_xt_img_overt_linear.json", results_dir=RESULTS_DIR)
             plot_overall_roc(xt_img_ov_lin.get('true_labels'), xt_img_ov_lin.get('decision_scores'),
                              data_label="Train Imagined, Test Overt (Linear)",
                              title="Cross-Train ROC: Imagined -> Overt (Linear)",
                              filename="xt_img_overt_lin_roc.png", output_dir=PLOTS_DIR)
    else: print("Skipping Cross-Train Imagined->Overt due to missing data.")


# --- End ---
script_end_time = time.time()
print("\n" + "="*40)
print(f"=== BCI Analysis Workflow Finished ===")
print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds")
print(f"Outputs saved in '{os.path.abspath(BASE_OUTPUT_DIR)}'")
print("="*40)