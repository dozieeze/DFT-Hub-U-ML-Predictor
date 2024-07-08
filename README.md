# DFT-Hub-U-ML-Predictor
Repository for integrating Density Functional Theory (DFT) with Hubbard U correction and Machine Learning (ML) to predict band gaps and lattice parameters of metal oxides. Includes data, scripts, and model information for reproducibility and further research.



# SCRIPT - ALL SYSTEM MULTI-TARGET + FEATURE IMPORTANCE

This script contains a comprehensive analysis of various regression models for predicting multiple target properties using multiple features. The models are evaluated using cross-validation techniques, and feature importance is displayed for the best-performing models.

## Dependencies

Ensure you have the following packages installed:

```bash
!pip install scikit-optimize
!pip install xgboost
!pip install numpy
!pip install pandas
!pip install scikit-learn
!pip install matplotlib
```

## Usage

### 1. Import Necessary Libraries

The script begins by importing necessary libraries for data manipulation, model training, evaluation, and visualization.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
```

### 2. Define Cross-Validation and Model Evaluation Functions

The following functions are defined for performing Leave-One-Out Cross-Validation (LOO-CV) and K-Fold Cross-Validation, training the models, and calculating performance metrics.

```python
def loo_cv_cross_val_train_eval_models(models, X, y):

    """
    This function performs Leave-One-Out Cross-Validation (LOO-CV) for multiple regression models.
    It trains and evaluates the models on each target variable, returning the evaluation results
    and best fold data.

    Parameters:
    - models: Dictionary of regression models to be evaluated.
    - X: Feature dataframe.
    - y: Target dataframe with multiple columns (one for each target variable).

    Returns:
    - evaluation_results: Dictionary with average train and test metrics for each model and target.
    - best_fold_data: Dictionary with detailed data of the best performing fold for each model and target.
    """

    evaluation_results = {target: {} for target in y.columns}
    best_fold_data = {target: {} for target in y.columns}
    loocv = LeaveOneOut()

    for target in y.columns:
        for name, model in models.items():
            fold_metrics = {'train': [], 'test': []}
            best_fold_mse = float('inf')
            best_fold_index = -1

            for i, (train_index, test_index) in enumerate(loocv.split(X)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[target].iloc[train_index], y[target].iloc[test_index]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model.fit(X_train_scaled, y_train)
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)

                train_metrics = calculate_metrics(y_train, y_train_pred)
                test_metrics = calculate_metrics(y_test, y_test_pred)
                
                fold_metrics['train'].append(train_metrics)
                fold_metrics['test'].append(test_metrics)

                if test_metrics['mse'] < best_fold_mse:
                    best_fold_mse = test_metrics['mse']
                    best_fold_index = i
                    best_fold_data[target][name] = {
                        'train_index': train_index, 'test_index': test_index,
                        'model': model,
                        'best_fold_train_metrics': train_metrics,
                        'best_fold_test_metrics': test_metrics,
                        'best_train_actuals': y_train,
                        'best_train_predictions': y_train_pred,
                        'best_test_actuals': y_test,
                        'best_test_predictions': y_test_pred
                    }

            avg_metrics = {
                'train': {k: np.mean([fold[k] for fold in fold_metrics['train']]) for k in fold_metrics['train'][0]},
                'test': {k: np.mean([fold[k] for fold in fold_metrics['test']]) for k in fold_metrics['test'][0]}
            }
            evaluation_results[target][name] = {
                'avg_train_metrics': avg_metrics['train'],
                'avg_test_metrics': avg_metrics['test'],
                'best_fold': best_fold_index
            }

    return evaluation_results, best_fold_data




def cross_val_train_eval_models(models, X, y, folds=10):
    evaluation_results = {target: {} for target in y.columns}
    best_fold_data = {target: {} for target in y.columns}
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)

    for target in y.columns:
        for name, model in models.items():
            fold_metrics = {'train': [], 'test': []}
            best_fold_mse = float('inf')
            best_fold_index = -1

            for i, (train_index, test_index) in enumerate(kfold.split(X)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[target].iloc[train_index], y[target].iloc[test_index]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model.fit(X_train_scaled, y_train)
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)

                train_metrics = calculate_metrics(y_train, y_train_pred)
                test_metrics = calculate_metrics(y_test, y_test_pred)
                
                fold_metrics['train'].append(train_metrics)
                fold_metrics['test'].append(test_metrics)

                if test_metrics['mse'] < best_fold_mse:
                    best_fold_mse = test_metrics['mse']
                    best_fold_index = i
                    best_fold_data[target][name] = {
                        'train_index': train_index, 'test_index': test_index,
                        'model': model,
                        'best_fold_train_metrics': train_metrics,
                        'best_fold_test_metrics': test_metrics,
                        'best_train_actuals': y_train,
                        'best_train_predictions': y_train_pred,
                        'best_test_actuals': y_test,
                        'best_test_predictions': y_test_pred
                    }

            avg_metrics = {
                'train': {k: np.mean([fold[k] for fold in fold_metrics['train']]) for k in fold_metrics['train'][0]},
                'test': {k: np.mean([fold[k] for fold in fold_metrics['test']]) for k in fold_metrics['test'][0]}
            }
            evaluation_results[target][name] = {
                'avg_train_metrics': avg_metrics['train'],
                'avg_test_metrics': avg_metrics['test'],
                'best_fold': best_fold_index
            }

    return evaluation_results, best_fold_data




def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Only calculate R^2 if there's more than one sample
    if len(y_true) > 1:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan  # Set R^2 to NaN for single-sample cases
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2
    }
```

### 3. Plotting Functions

Functions for plotting the results and displaying feature importance are defined as follows:

```python
def plot_best_fold_all(sorted_evaluation_results, X, y, best_fold_data):
    num_targets = len(y.columns)
    num_models = len(sorted_evaluation_results[y.columns[0]])
    fig, axes = plt.subplots(nrows=num_targets, ncols=num_models, figsize=(num_models * 5, num_targets * 5))
    
    if num_targets == 1:
        axes = [axes]
    if num_models == 1:
        axes = [[ax] for ax in axes]

    for target_index, target in enumerate(y.columns):
        target_name = target.replace('_', ' ').replace('Calculated ', '')
        for model_index, (name, _) in enumerate(sorted_evaluation_results[target]):
            ax = axes[target_index][model_index]
            data = best_fold_data[target][name]
            
            plot_scatter(ax, data['best_train_actuals'], data['best_train_predictions'], 
                         data['best_test_actuals'], data['best_test_predictions'])
            
            ax.set_title(f'{name} - {target_name}', fontsize=18, weight='bold')
            ax.set_xlabel('DFT+U', fontsize=18, weight='bold')
            ax.set_ylabel('ML Predicted', fontsize=18, weight='bold')
            
            ax.set_aspect('equal', adjustable='box')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            
            ax.legend(prop={'size': 12, 'weight': 'bold'}, title_fontsize='12', shadow=True, frameon=True)
    
    plt.tight_layout()
    plt.show()

def plot_scatter(ax, y_train, y_train_pred, y_test, y_test_pred):
    ax.scatter(y_train, y_train_pred, color='blue', label='Train Data', alpha=0.6)
    ax.scatter(y_test, y_test_pred, color='red', label='Test Data', alpha=0.6)
    
    all_vals = np.concatenate([y_train, y_train_pred, y_test, y_test_pred])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    buffer = (max_val - min_val) * 0.05
    
    ax.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer], 'k--', lw=2)
    ax.set_xlim([min_val - buffer, max_val + buffer])
    ax.set_ylim([min_val - buffer, max_val + buffer])

def display_feature_importance(models, X, best_fold_data, y):
    for target in y.columns:
        print(f"\nFeature importance for {target}:")
        for name, data in best_fold_data[target].items():
            model = data['model']
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                print(f"  {name} model does not support feature importance extraction.")
                continue

            sorted_idx = importance.argsort()[::-1]
            sorted_features = X.columns[sorted_idx]
            sorted_importance = importance[sorted_idx]

            plt.figure(figsize=(10, 6))
            plt.bar(range(len(sorted_importance)), sorted_importance, align='center')
            plt.xticks(range(len(sorted_importance)), sorted_features, rotation=90)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title(f'Feature Importance for {name} Model - {target}')
            plt.tight_layout()
            plt.show()

            print(f"  {name} model:")
            for feat, imp in zip(sorted_features, sorted_importance):
                print(f"    {feat}: {imp:.4f}")
            print()
```

### 4. Load Data

Load the dataset and define features and target variables.

```python
data = pd.read_excel('All System Model May 2024.xlsx', sheet_name='Sheet1')
features = ["Up_Value/eV", "Ud_Value/eV", "alpha_oxide/degree", "beta_oxide/degree", "gamma_oxide/degree", 
            "Number of X atoms", "Number of O atoms", "Lattice_constant_a_of_X_pm", "Lattice_constant_b_of_X_pm", 
            "Lattice_constant_c_of_X_pm", "Atomic_radius/pm_of_X",  "Van_der_Waals_radius/pm_of_X", "Atomic_No_of_X", 
            "Atomic_Mass/amu_of_X", "Period_of_X", "First_ionization_energy/KJ/mol_of_X", "Density/Kg/m^3_of_X", 
            "Electron_Affinity/ev_of_X", "Work_Function/ev_of_X", "Pauling_Electronegativity/units_of_X", "d-shell_of_X", 
            "Lattice_angle_alpha_of_X_degree", "Lattice_angle_beta_of_X_degree", "Lattice_angle_gamma_of_X_degree", 
            "Lattice_constant_a_of_O_pm", "Lattice_constant_b_of_O_pm", "Lattice_constant_c_of_O_pm", "Atomic_radius/pm_of_O", 
            "Van_der_Waals_radius/pm_of_O", "Atomic_No_of_O", "Atomic_Mass/amu_of_O", "Period_of_O", 
            "First_ionization_energy/KJ/mol_of_O", "Density/Kg/m^3_of_O", "Electron_Affinity/ev_of_O", 
            "Pauling_Electronegativity/units_of_O", "Lattice_angle_alpha_of_O_degree", "Lattice_angle_beta_of_O_degree", 
            "Lattice_angle_gamma_of_O_degree"]

X = data[features]
y = data[['Calculated_Band_Gap/eV', 'Calculated_a/angstrom', 'Calculated_b/angstrom', 'Calculated_c/angstrom']]
```

### 5. Define Models

Define a dictionary of models to be evaluated.

```python
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(),
    'Gaussian Process': GaussianProcessRegressor(),
    'Polynomial': make_pipeline(PolynomialFeatures(2), LinearRegression()),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor(),
    'SVM': SVR()
}
```

### 6. Train and Evaluate Models

Train and evaluate the models using cross-validation techniques.

```python
# Leave-one-out CV
# evaluation_results, best_fold_data = loo_cv_cross_val_train_eval_models(models, X, y)

# KFold CV
evaluation_results, best_fold_data = cross_val_train_eval_models(models, X, y, folds=50)

# Sort models based on average test MSE
sorted_evaluation_results = {target: sorted(evaluation_results[target].items(), 
                                            key=lambda x: x[1]['avg_test_metrics']['mse']) 
                             for target in y.columns}
```

### 7. Print Model Performance Metrics

Print the performance metrics for each model and target variable.

```python
for target in y.columns:
    print(f"\nModel Performance Metrics for {target} (sorted by Test MSE):")
    for name, metrics in sorted_evaluation_results[target]:
        avg_train = metrics['avg_train_metrics']
        avg_test = metrics['avg_test_metrics']
        best_train = best_fold_data[target][name]['best_fold_train_metrics']
        best_test = best_fold_data[target][name]['best_fold_test_metrics']

        print(f"{name}:")
        print(f"  Average Train Metrics - MSE = {avg_train['mse']:.2f}, RMSE = {avg_train['rmse']:.2f}, "
              f"MAE = {avg_train['mae']:.2f}, R2 = {avg_train['r2']:.2f}")
        print(f"  Average Test Metrics - MSE = {avg_test['mse']:.2f}, RMSE = {avg_test['rmse']:.2f}, "
              f"MAE = {avg_test['mae']:.2f}, R2 = {avg_test['r2']:.2f}")
        print(f"  Best Fold Train Metrics - MSE = {best_train['mse']:.2f}, RMSE = {best_train['rmse']:.2f}, "
              f"MAE = {best_train['mae']:.2f}, R2 = {best_train['r2']:.2f}")
        print(f"  Best Fold Test Metrics - MSE = {best_test['mse']:.2f}, RMSE = {best_test['rmse']:.2f}, "
              f"MAE = {best_test['mae']:.2f}, R2 = {best_test['r2']:.2f}\n")
```

### 8. Plot Results

Plot the results for the best performing folds.

```python
plot_best_fold_all(sorted_evaluation_results, X, y, best_fold_data)
```

### 9. Display Feature Importance

Display the feature importance for the best-performing models.

```python
display_feature_importance(models, X, best_fold_data, y)
```

## Conclusion

This repository provides a robust framework for evaluating and comparing the performance of various regression models on multiple target properties. The cross-validation techniques and feature importance analysis offer insights into the model's predictive power and the significance of each feature.
