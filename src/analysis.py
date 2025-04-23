import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

def perform_two_level_cv(X, y, data_label="Data", kernel='linear', gamma='scale', degree=3, outer_k=6, inner_k=5):
    C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] # C = 1 / alpha

    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=42)

    outer_fold_accuracies, optimal_Cs = [], []
    all_true_labels, all_decision_scores = [], []
    outer_fold_true_labels, outer_fold_scores = [], []
    fold1_coefficients = None # Only for linear kernel

    print(f"\n--- Starting {outer_k}-Fold Outer CV for {data_label} Data (Kernel: {kernel}) ---")

    # Outer loop
    for i, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X, y)):
        print(f"  Outer Fold {i+1}/{outer_k}")
        X_train_outer, X_test_outer = X[train_outer_idx], X[test_outer_idx]
        y_train_outer, y_test_outer = y[train_outer_idx], y[test_outer_idx]

        scaler = StandardScaler()
        X_train_outer_scaled = scaler.fit_transform(X_train_outer)
        X_test_outer_scaled = scaler.transform(X_test_outer)

        # Find best C
        best_inner_acc = -1
        best_C_for_fold = C_range[0]
        for C_val in C_range:
            inner_fold_accuracies = []
            for j, (train_inner_idx, val_inner_idx) in enumerate(inner_cv.split(X_train_outer_scaled, y_train_outer)):
                X_train_inner, X_val_inner = X_train_outer_scaled[train_inner_idx], X_train_outer_scaled[val_inner_idx]
                y_train_inner, y_val_inner = y_train_outer[train_inner_idx], y_train_outer[val_inner_idx]
                svm_inner = SVC(kernel=kernel, C=C_val, gamma=gamma, degree=degree,
                                random_state=42, probability=False)
                svm_inner.fit(X_train_inner, y_train_inner)
                accuracy = svm_inner.score(X_val_inner, y_val_inner)
                inner_fold_accuracies.append(accuracy)
            avg_inner_acc = np.mean(inner_fold_accuracies)
            if avg_inner_acc > best_inner_acc:
                best_inner_acc = avg_inner_acc
                best_C_for_fold = C_val

        optimal_Cs.append(best_C_for_fold)

        final_svm = SVC(kernel=kernel, C=best_C_for_fold, gamma=gamma, degree=degree,
                        random_state=42, probability=False)
        final_svm.fit(X_train_outer_scaled, y_train_outer)

        outer_accuracy = final_svm.score(X_test_outer_scaled, y_test_outer)
        outer_fold_accuracies.append(outer_accuracy)
        decision_scores = final_svm.decision_function(X_test_outer_scaled)
        all_true_labels.extend(y_test_outer)
        all_decision_scores.extend(decision_scores)
        outer_fold_true_labels.append(y_test_outer)
        outer_fold_scores.append(decision_scores)

        # Store fold 1 coefficients ONLY for linear kernel
        if i == 0 and kernel == 'linear':
            fold1_coefficients = final_svm.coef_.flatten()
        elif i == 0 and kernel != 'linear':
             print("    Note: Coefficients (.coef_) not available for non-linear kernels. Skipping weight plots.")


        print(f"  Outer Fold {i+1} completed. Accuracy: {outer_accuracy:.4f}. Optimal C: {best_C_for_fold}")

    average_accuracy = np.mean(outer_fold_accuracies)
    std_dev_accuracy = np.std(outer_fold_accuracies)

    print(f"\nFinished {outer_k}-fold CV for {data_label} data (Kernel: {kernel}).")
    print(f"Optimal C found per fold: {optimal_Cs}")
    print(f"Accuracy per fold: {[f'{acc:.4f}' for acc in outer_fold_accuracies]}")
    print(f"Average Accuracy: {average_accuracy:.4f} +/- {std_dev_accuracy:.4f}")

    results = {
        'avg_accuracy': average_accuracy, 'std_accuracy': std_dev_accuracy,
        'optimal_Cs': optimal_Cs, 'fold_accuracies': outer_fold_accuracies,
        'all_true_labels': np.array(all_true_labels),
        'all_decision_scores': np.array(all_decision_scores),
        'fold1_coefficients': fold1_coefficients,
        'outer_fold_true_labels': outer_fold_true_labels,
        'outer_fold_scores': outer_fold_scores,
        'kernel': kernel # Store kernel used
    }
    return results

def tune_hyperparameters(X_train, y_train, kernel, C_range, gamma_range=None, degree_range=None, n_splits=5):
    print(f"    Tuning hyperparameters (Kernel: {kernel}) using {n_splits}-fold CV...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    param_grid = {'C': C_range}
    if kernel in ['rbf', 'poly', 'sigmoid'] and gamma_range:
        param_grid['gamma'] = gamma_range
    if kernel == 'poly' and degree_range:
        param_grid['degree'] = degree_range

    scorer = make_scorer(accuracy_score)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid_search = GridSearchCV(SVC(kernel=kernel, random_state=42, probability=False),
                               param_grid, scoring=scorer, cv=cv, n_jobs=-1) # Doesn't make a difference on my computer, but uses all CPU cores
    grid_search.fit(X_train_scaled, y_train)

    print(f"    Best parameters found: {grid_search.best_params_} (Best Score: {grid_search.best_score_:.4f})")
    return grid_search.best_params_

def perform_cross_training(X_train, y_train, X_test, y_test, train_label, test_label, kernel='linear', gamma='scale', degree=3):
    print(f"\n--- Cross-Training: Train on {train_label}, Test on {test_label} (Kernel: {kernel}) ---")

    C_range_xt = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # gamma_range_xt = ['scale', 'auto'] # Tune gamma?
    # degree_range_xt = [2, 3, 4]     # Tune degree?

    best_params = tune_hyperparameters(X_train, y_train, kernel, C_range_xt)
                                        # gamma_range=gamma_range_xt if kernel != 'linear' else None,
                                        # degree_range=degree_range_xt if kernel == 'poly' else None)

    best_C = best_params.get('C', 1.0) # Default C=1.0 if tuning fails? Risky, maybe error out
    best_gamma = best_params.get('gamma', gamma)
    best_degree = best_params.get('degree', degree)

    print("    Training final model on entire training set...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    final_svm = SVC(kernel=kernel, C=best_C, gamma=best_gamma, degree=best_degree,
                    random_state=42, probability=False) # probability = false goes way faster
    final_svm.fit(X_train_scaled, y_train)

    print("    Testing model on test set...")
    X_test_scaled = scaler.transform(X_test)
    accuracy = final_svm.score(X_test_scaled, y_test)
    decision_scores = final_svm.decision_function(X_test_scaled)

    print(f"Test Accuracy: {accuracy:.4f}")

    results = {
        'accuracy': accuracy,
        'true_labels': y_test,
        'decision_scores': decision_scores,
        'best_params': best_params,
        'kernel': kernel
    }
    return results