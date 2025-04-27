# analysis.py
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer
import time # Added for timing folds
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress ConvergenceWarning from SVC (can happen with low max_iter or difficult data)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

def approximate_svm_weights(svm_model):
    """
    Approximates the primal weights w for non-linear SVM kernels using dual coefficients.
    w = sum(alpha_i * y_i * x_i) = dual_coef_ @ support_vectors_
    Requires a fitted SVC model. Returns None on failure.
    """
    try:
        # Ensure the model is fitted and has the necessary attributes
        if not hasattr(svm_model, 'dual_coef_') or not hasattr(svm_model, 'support_vectors_'):
             print("Warning: Cannot approximate weights. Model missing 'dual_coef_' or 'support_vectors_'.")
             return None
        if svm_model.dual_coef_ is None or svm_model.support_vectors_ is None:
             print("Warning: Cannot approximate weights. 'dual_coef_' or 'support_vectors_' is None.")
             return None

        # dual_coef_ shape is (n_classes - 1, n_SV) which is (1, n_SV) for binary.
        # support_vectors_ shape is (n_SV, n_features).
        # The result is shape (1, n_features).
        weights = svm_model.dual_coef_ @ svm_model.support_vectors_
        return weights.flatten()
    except Exception as e:
        print(f"Error approximating SVM weights: {e}")
        return None

def perform_two_level_cv(X, y, data_label="Data", kernel='linear', gamma='scale', degree=3, C_range=None, outer_k=6, inner_k=5):
    """
    Performs two-level stratified CV for SVM. Tunes only C in the inner loop.
    Approximates weights for non-linear kernels for Fold 1 visualization.

    Args:
        X, y: Feature matrix and labels.
        data_label (str): Identifier for the dataset (e.g., "Overt").
        kernel (str): SVM kernel type ('linear', 'rbf', 'poly', 'sigmoid').
        gamma (float or 'scale'/'auto'): Kernel coefficient (for 'rbf', 'poly', 'sigmoid').
        degree (int): Degree for 'poly' kernel.
        C_range (list, optional): List of C values for inner loop tuning. Uses default if None.
        outer_k (int): Number of outer cross-validation folds.
        inner_k (int): Number of inner cross-validation folds for hyperparameter tuning.

    Returns:
        dict: Dictionary containing CV results (accuracy, scores, labels, weights for fold 1).
              Returns None if input data is invalid.
    """
    if X is None or y is None:
        print(f"Error: Missing input data (X or y) for CV on {data_label}.")
        return None

    if C_range is None:
        C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] # Default C values

    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=42)

    # Initialization
    outer_fold_accuracies, optimal_Cs = [], []
    all_true_labels, all_decision_scores = [], []
    outer_fold_true_labels, outer_fold_scores = [], []
    fold1_coefficients = None # Store weights (actual or approximated) for fold 1

    print(f"\n--- Starting {outer_k}-Fold Outer CV for {data_label} Data (Kernel: {kernel}) ---")
    start_time_cv = time.time()

    # Outer loop (Training/Testing Split)
    for i, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X, y)):
        fold_start_time = time.time()
        print(f"  Outer Fold {i+1}/{outer_k}")
        X_train_outer, X_test_outer = X[train_outer_idx], X[test_outer_idx]
        y_train_outer, y_test_outer = y[train_outer_idx], y[test_outer_idx]

        # Scale data *within* the outer fold
        scaler = StandardScaler()
        X_train_outer_scaled = scaler.fit_transform(X_train_outer)
        X_test_outer_scaled = scaler.transform(X_test_outer) # Use same scaler for test set

        # Inner loop (Hyperparameter Tuning - only C)
        best_inner_acc = -1
        best_C_for_fold = C_range[0]
        for C_val in C_range:
            inner_fold_accuracies = []
            for j, (train_inner_idx, val_inner_idx) in enumerate(inner_cv.split(X_train_outer_scaled, y_train_outer)):
                X_train_inner, X_val_inner = X_train_outer_scaled[train_inner_idx], X_train_outer_scaled[val_inner_idx]
                y_train_inner, y_val_inner = y_train_outer[train_inner_idx], y_train_outer[val_inner_idx]

                # Train SVM for inner fold validation - increase max_iter
                svm_inner = SVC(kernel=kernel, C=C_val, gamma=gamma, degree=degree,
                                random_state=42, probability=False, max_iter=10000) # Increased max_iter
                try:
                     svm_inner.fit(X_train_inner, y_train_inner)
                     accuracy = svm_inner.score(X_val_inner, y_val_inner)
                     inner_fold_accuracies.append(accuracy)
                except Exception as e:
                     print(f"    Warning: Error during inner CV fit/score (C={C_val}, fold {j+1}): {e}")
                     inner_fold_accuracies.append(0) # Penalize failures

            # Average accuracy over inner folds for this C value
            avg_inner_acc = np.mean(inner_fold_accuracies) if inner_fold_accuracies else 0
            if avg_inner_acc > best_inner_acc:
                best_inner_acc = avg_inner_acc
                best_C_for_fold = C_val

        optimal_Cs.append(best_C_for_fold)

        # Train final SVM for the outer fold using the best C found
        # Increase max_iter for final model too
        final_svm = SVC(kernel=kernel, C=best_C_for_fold, gamma=gamma, degree=degree,
                        random_state=42, probability=False, max_iter=20000) # Increased max_iter
        try:
            final_svm.fit(X_train_outer_scaled, y_train_outer)
        except Exception as e:
            print(f"  Error fitting final SVM for Outer Fold {i+1}: {e}. Skipping fold results.")
            # Append NaNs or handle appropriately? Appending empty/None where possible.
            outer_fold_accuracies.append(np.nan)
            outer_fold_true_labels.append(np.array([])) # Append empty array
            outer_fold_scores.append(np.array([]))     # Append empty array
            continue # Skip to next fold

        # Evaluate final model on the outer test set
        outer_accuracy = final_svm.score(X_test_outer_scaled, y_test_outer)
        outer_fold_accuracies.append(outer_accuracy)
        try:
             decision_scores = final_svm.decision_function(X_test_outer_scaled)
        except Exception as e:
             print(f"  Error getting decision function for Outer Fold {i+1}: {e}")
             decision_scores = np.full_like(y_test_outer, np.nan, dtype=float) # Fill with NaNs

        # Store results for aggregation and plotting
        all_true_labels.extend(y_test_outer)
        all_decision_scores.extend(decision_scores)
        outer_fold_true_labels.append(y_test_outer)    # Store as list of arrays
        outer_fold_scores.append(decision_scores) # Store as list of arrays

        # Store Fold 1 coefficients (actual for linear, approximated for others)
        if i == 0:
            if kernel == 'linear':
                try:
                    fold1_coefficients = final_svm.coef_.flatten()
                    print("    Stored actual coefficients for linear kernel (Fold 1).")
                except AttributeError:
                    print("    Warning: Could not retrieve .coef_ for linear kernel (Fold 1).")
                    fold1_coefficients = None
            else:
                print(f"    Attempting to approximate coefficients for {kernel} kernel (Fold 1)...")
                fold1_coefficients = approximate_svm_weights(final_svm)
                if fold1_coefficients is not None:
                    print("    Successfully approximated and stored coefficients (Fold 1).")
                else:
                     print("    Approximation failed or not possible (Fold 1).")

        fold_end_time = time.time()
        print(f"  Outer Fold {i+1} completed. Accuracy: {outer_accuracy:.4f}. Optimal C: {best_C_for_fold}. Time: {fold_end_time - fold_start_time:.1f}s")

    # --- End Outer Loop ---
    end_time_cv = time.time()
    # Calculate average accuracy, ignoring potential NaNs from failed folds
    average_accuracy = np.nanmean(outer_fold_accuracies)
    std_dev_accuracy = np.nanstd(outer_fold_accuracies)

    print(f"\nFinished {outer_k}-fold CV for {data_label} data (Kernel: {kernel}).")
    print(f"Total CV Time: {end_time_cv - start_time_cv:.1f}s")
    print(f"Optimal C found per fold: {optimal_Cs}")
    print(f"Accuracy per fold: {[f'{acc:.4f}' if not np.isnan(acc) else 'NaN' for acc in outer_fold_accuracies]}")
    print(f"Average Accuracy: {average_accuracy:.4f} +/- {std_dev_accuracy:.4f}")

    # Prepare results dictionary
    results = {
        'avg_accuracy': average_accuracy, 'std_accuracy': std_dev_accuracy,
        'optimal_Cs': optimal_Cs, 'fold_accuracies': outer_fold_accuracies,
        'all_true_labels': np.array(all_true_labels),
        'all_decision_scores': np.array(all_decision_scores),
        'fold1_coefficients': fold1_coefficients, # Contains actual (linear) or approximated weights for fold 1
        'outer_fold_true_labels': outer_fold_true_labels, # List of arrays
        'outer_fold_scores': outer_fold_scores,       # List of arrays
        'kernel': kernel # Store kernel used
    }
    return results

def tune_hyperparameters(X_train, y_train, kernel, C_range, gamma_range=None, degree_range=None, n_splits=5):
    """
    Tunes SVM hyperparameters using GridSearchCV with internal scaling.

    Args:
        X_train, y_train: Training data and labels.
        kernel (str): SVM kernel type.
        C_range (list): List of C values to test.
        gamma_range (list, optional): List of gamma values for 'rbf', 'poly', 'sigmoid'.
        degree_range (list, optional): List of degree values for 'poly'.
        n_splits (int): Number of cross-validation folds for tuning.

    Returns:
        dict: Best parameters found by GridSearchCV, or None if tuning fails.
    """
    print(f"    Tuning hyperparameters (Kernel: {kernel}) using {n_splits}-fold CV...")

    if X_train is None or y_train is None or len(X_train) == 0:
        print("    Error: Invalid input data for hyperparameter tuning.")
        return None

    # Scale data once before tuning (GridSearchCV doesn't handle this internally robustly)
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
    except Exception as e:
        print(f"    Error scaling data for tuning: {e}")
        return None

    # Define parameter grid based on kernel type
    param_grid = {'C': C_range}
    if kernel in ['rbf', 'poly', 'sigmoid'] and gamma_range:
        param_grid['gamma'] = gamma_range
    if kernel == 'poly' and degree_range:
        param_grid['degree'] = degree_range

    # Use accuracy for scoring, matching the 2-level CV inner loop
    scorer = make_scorer(accuracy_score)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize GridSearchCV
    # Increase max_iter here too if convergence warnings were an issue
    grid_search = GridSearchCV(SVC(kernel=kernel, random_state=42, probability=False, max_iter=10000), # Increased max_iter
                               param_grid, scoring=scorer, cv=cv, n_jobs=-1, error_score='raise') # Use all CPU cores, raise errors

    try:
        grid_search.fit(X_train_scaled, y_train)
        print(f"    Best parameters found: {grid_search.best_params_} (Best CV Score: {grid_search.best_score_:.4f})")
        return grid_search.best_params_
    except Exception as e:
        print(f"    Error during GridSearchCV fitting: {e}")
        print("    Hyperparameter tuning failed.")
        return None # Indicate failure


def perform_cross_training(X_train, y_train, X_test, y_test, train_label, test_label, kernel='linear', gamma='scale', degree=3):
    """
    Performs cross-training: Tunes hyperparameters on the training set,
    trains a final model, and tests on the separate test set.

    Args:
        X_train, y_train: Training features and labels.
        X_test, y_test: Testing features and labels.
        train_label (str): Identifier for training data (e.g., "Overt").
        test_label (str): Identifier for testing data (e.g., "Imagined").
        kernel (str): SVM kernel type.
        gamma, degree: Default kernel parameters (used if not tuned).

    Returns:
        dict: Results including test accuracy, true labels, decision scores,
              and best parameters found. Returns None on error.
    """
    if X_train is None or y_train is None or X_test is None or y_test is None:
        print("Error: Missing input data for cross-training.")
        return None

    print(f"\n--- Cross-Training: Train on {train_label}, Test on {test_label} (Kernel: {kernel}) ---")
    start_time_xt = time.time()

    # 1. Tune Hyperparameters on the *entire* Training Set
    # Define parameter ranges for tuning - Adjust as needed
    C_range_xt = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gamma_range_xt = ['scale', 'auto', 0.001, 0.01, 0.1, 1] # Expanded range example for RBF/Poly
    degree_range_xt = [2, 3, 4]                            # Example for Poly

    # Determine which params to tune based on kernel
    tune_gamma = kernel in ['rbf', 'poly', 'sigmoid']
    tune_degree = kernel == 'poly'

    best_params = tune_hyperparameters(
        X_train, y_train, kernel, C_range_xt,
        gamma_range=gamma_range_xt if tune_gamma else None,
        degree_range=degree_range_xt if tune_degree else None
    )

    # Handle tuning failure
    if best_params is None:
        print("Cross-training aborted due to hyperparameter tuning failure.")
        return None

    # Extract best parameters, using defaults only if *not tuned* (shouldn't happen for C if tune_hyperparameters succeeded)
    best_C = best_params.get('C') # Required from tuning results
    best_gamma = best_params.get('gamma', gamma) # Use tuned gamma if available, else default
    best_degree = best_params.get('degree', degree) # Use tuned degree if available, else default
    print(f"    Using parameters for final model: C={best_C}, gamma={best_gamma}, degree={best_degree}")


    # 2. Train Final Model on Scaled Training Data
    print("    Training final model on entire training set...")
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
    except Exception as e:
        print(f"    Error scaling training data: {e}")
        return None

    # Increase max_iter if needed for final model
    final_svm = SVC(kernel=kernel, C=best_C, gamma=best_gamma, degree=best_degree,
                    random_state=42, probability=False, max_iter=20000) # Increased max_iter
    try:
        final_svm.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"    Error fitting final SVM model: {e}")
        return None

    # 3. Test on Scaled Testing Data
    print("    Testing model on test set...")
    try:
        X_test_scaled = scaler.transform(X_test) # Use scaler fitted on training data
        accuracy = final_svm.score(X_test_scaled, y_test)
        decision_scores = final_svm.decision_function(X_test_scaled)
    except Exception as e:
        print(f"    Error during testing (scaling/predicting): {e}")
        return None # Abort if testing fails

    end_time_xt = time.time()
    print(f"Cross-Training completed. Time: {end_time_xt - start_time_xt:.1f}s")
    print(f"Test Accuracy ({test_label}): {accuracy:.4f}")

    # Return results
    results = {
        'accuracy': accuracy,
        'true_labels': y_test,
        'decision_scores': decision_scores,
        'best_params': best_params,
        'kernel': kernel
    }
    return results