# DFT-Hub-U-ML-Predictor
Repository for integrating Density Functional Theory (DFT) with Hubbard U correction and Machine Learning (ML) to predict band gaps and lattice parameters of metal oxides. Includes data, scripts, and model information for reproducibility and further research.

## Environment Setup and Requirements

### Anaconda Recommendation

We recommend using Anaconda for managing your Python environment and packages. Anaconda provides a user-friendly way to handle dependencies and create isolated environments for your projects.

You can download Anaconda from: https://www.anaconda.com/download

Choose the version appropriate for your operating system (Windows, macOS, or Linux) and follow the installation instructions on the Anaconda website.

### Create a New Environment

After installing Anaconda, follow these steps to create a new environment for this project:

1. Open Anaconda Prompt (on Windows) or a terminal (on macOS or Linux)

2. Create a new conda environment:
   ```bash
   conda create --name dft-hub-u-ml python=3.9
   ```

3. Activate the new environment:
   ```bash
   conda activate dft-hub-u-ml
   ```

### Requirements

Once you've activated the new environment, proceed with installing the required packages.

#### Jupyter Notebook
Install Jupyter Notebook using:

```bash
conda install -c conda-forge jupyter
```

#### Other Required Packages
Install the following packages:

```bash
pip install scikit-optimize==0.10.2 xgboost==2.1.1 numpy==2.0.0 pandas==2.2.2 scikit-learn==1.5.1 matplotlib==3.9.1
```

### Verifying Installation

After installation, verify that everything is set up correctly:

1. List installed packages:
   ```bash
   pip list
   ```

2. Create a Jupyter kernel for this environment:
   ```bash
   python -m ipykernel install --user --name dft-hub-u-ml --display-name "Python (dft-hub-u-ml)"
   ```

### Launching Jupyter Notebook

To start working on your project:

1. Ensure you're in the dft-hub-u-ml environment:
   ```bash
   conda activate dft-hub-u-ml
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. In the Jupyter interface, create a new notebook or open an existing one, and select the "Python (dft-hub-u-ml)" kernel.

### Deactivating the Environment

When you're done working on the project, you can deactivate the environment:

```bash
conda deactivate
```

By following these steps, you'll have a dedicated environment for your DFT-Hub-U-ML project with all the necessary dependencies installed and isolated from other projects. Using Anaconda simplifies the process of managing Python environments and packages, making it easier to maintain consistent and reproducible project setups.


### Jupyter Notebook Usage

#### 1. Import Necessary Libraries

The script begins by importing necessary libraries for data manipulation, model training, evaluation, and visualization.

```python
# Basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
import scipy.sparse


# Validation
from sklearn.model_selection import KFold, LeaveOneOut, GroupKFold

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Linear models
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Tree-based models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Support Vector Machines
from sklearn.svm import SVR

# Gaussian Process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# XGBoost
from xgboost import XGBRegressor

from sklearn.neural_network import MLPRegressor

# Skopt (scikit-optimize)
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Permutation feature importance
from sklearn.inspection import permutation_importance

# Importing SHAP
import shap

# Cloning
from sklearn.base import clone

# Multi-output regressors
from sklearn.multioutput import MultiOutputRegressor

# Pipelines
from sklearn.pipeline import Pipeline, make_pipeline
```
<br/><br/>  
#### 2. Define Validation and Model Evaluation Functions

The following functions are defined for performing Leave-One-Out Cross-Validation (LOO-CV) and K-Fold Cross-Validation, training the models, and calculating performance metrics.



##### 2.1. Cross-Validation Function

Cross-validation is used to ensure that the model's performance is generalized and not overfitting to a specific subset of the data.

```python
def cross_val_train_eval_models_per_fold(models, X, y, cv=10, groups=None, test_ids=None):
    evaluation_results = {target: {} for target in y.columns}
    best_fold_data = {target: {} for target in y.columns}

    if test_ids is not None:
        if groups is None:
            raise ValueError("Groups must be provided when test_ids are specified.")
        test_indices = groups[groups.isin(test_ids)].index
        train_indices = groups[~groups.isin(test_ids)].index

        train_MP_IDs = set(groups.iloc[train_indices])
        test_MP_IDs = set(groups.iloc[test_indices])
        overlap = train_MP_IDs.intersection(test_MP_IDs)
        assert len(overlap) == 0, f"Overlap detected between train and test MP-IDs: {overlap}"

        splits = [(train_indices, test_indices)]
    else:
        if cv == 'loo':
            cv = LeaveOneOut()
        elif groups is not None:
            cv = GroupKFold(n_splits=cv)
            splits = list(cv.split(X, y, groups=groups))

            for i, (train_index, test_index) in enumerate(splits):
                train_MP_IDs = set(groups.iloc[train_index])
                test_MP_IDs = set(groups.iloc[test_index])
                overlap = train_MP_IDs.intersection(test_MP_IDs)
                assert len(overlap) == 0, f"Overlap detected in fold {i} between train and test MP-IDs: {overlap}"
        else:
            cv = KFold(n_splits=cv, shuffle=True, random_state=100)
            splits = list(cv.split(X, y))

    for target in y.columns:
        for name, model in models.items():
            fold_metrics = []
            best_fold_mse = float('inf')
            best_fold_index = -1

            for i, (train_index, test_index) in enumerate(splits):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[target].iloc[train_index], y[target].iloc[test_index]

                assert set(X_train.index).isdisjoint(set(X_test.index)), "Data leakage detected"

                model_clone = clone(model)

                model_clone.fit(X_train, y_train)
                y_train_pred = model_clone.predict(X_train)
                y_test_pred = model_clone.predict(X_test)

                train_metrics = calculate_metrics(y_train, y_train_pred)
                test_metrics = calculate_metrics(y_test, y_test_pred)

                fold_metric_entry = {
                    'fold_index': i,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                }
                if groups is not None:
                    fold_metric_entry['test_MP_IDs'] = groups.iloc[test_index].tolist()
                fold_metrics.append(fold_metric_entry)

                if test_metrics['mse'] < best_fold_mse:
                    best_fold_mse = test_metrics['mse']
                    best_fold_index = i
                    best_fold_data[target][name] = {
                        'train_index': train_index, 'test_index': test_index,
                        'model': model_clone,
                        'best_fold_train_metrics': train_metrics,
                        'best_fold_test_metrics': test_metrics,
                        'best_train_actuals': y_train,
                        'best_train_predictions': y_train_pred,
                        'best_test_actuals': y_test,
                        'best_test_predictions': y_test_pred
                    }

            avg_train_metrics = {k: np.mean([fold['train_metrics'][k] for fold in fold_metrics]) for k in fold_metrics[0]['train_metrics']}
            avg_test_metrics = {k: np.mean([fold['test_metrics'][k] for fold in fold_metrics]) for k in fold_metrics[0]['test_metrics']}
            evaluation_results[target][name] = {
                'avg_train_metrics': avg_train_metrics,
                'avg_test_metrics': avg_test_metrics,
                'best_fold': best_fold_index,
                'fold_metrics': fold_metrics 
            }

    return evaluation_results, best_fold_data
```

**Explanation:**
- **Purpose:** This function performs K-Fold Cross-Validation (default 5 folds)  or Leave-One-Out Cross-Validation (cv == 'loo') to evaluate multiple regression models on multiple target variables.
- **Process:** K-Fold CV: For each model and each target variable, it splits the data into K folds, trains the model on K-1 folds, and tests it on the remaining fold. This process repeats K times with each fold used once as the test set. LOO CV: For each model and each target variable, it trains the model on all but one sample (leaving one out) and tests on the left-out sample. This process repeats until every sample has been left out once.
- **Outputs:** It include average training and testing metrics across all folds, as well as metrics for the fold with the lowest Mean Squared Error (MSE), indicating the best performing fold for each model and target variable.


##### 2.2. Evaluation Function without Cross Validation


```python
def evaluate_models(models, X_train, y_train, X_test, y_test):
    evaluation_results = {target: {} for target in y_train.columns}
    model_data = {target: {} for target in y_train.columns}
    
    for target in y_train.columns:
        for name, model in models.items():
            model_copy = clone(model)
            model_copy.fit(X_train, y_train[target])
            y_train_pred = model_copy.predict(X_train)
            y_test_pred = model_copy.predict(X_test)

            train_metrics = calculate_metrics(y_train[target], y_train_pred)
            test_metrics = calculate_metrics(y_test[target], y_test_pred)
            
            evaluation_results[target][name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }
            
            model_data[target][name] = {
                'model': model_copy,
                'test_actuals': y_test[target],
                'test_predictions': y_test_pred
            }

    return evaluation_results, model_data
```
**Explanation:**
- **Purpose:** This function evaluates multiple regression models on multiple target variables using a fixed train and test split, without cross-validation. It's designed for scenarios where you have a separate, independent test set, such as predicting properties of new, unseen materials. 
- **Process:** For each model and each target variable, it trains the model on the entire training set and then makes predictions on both the training and test sets. It calculates performance metrics for both sets.
- **Outputs:** It returns two dictionaries:

**evaluation_results**: Contains training and testing metrics for each model and target variable.<br/><br/>
**model_data**: Stores the trained model objects, actual test values, and test predictions for each model and target variable.


##### 2.3. Metric Calculation Function

```python
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2
    }
```

**Explanation:**
- **Purpose:** This function calculates and returns key regression metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R².
- **Note:** R² is only calculated if there is more than one sample, as it requires multiple samples to be meaningful. For single-sample cases, R² is set to NaN (Not a Number).

<br/><br/>
#### 3. Define Plotting Functions

Functions for plotting the results are defined as follows:

##### 3.1. Plot Best Fold Results for All Models and Targets

```python
def plot_best_fold_all(sorted_evaluation_results, X, y, best_fold_data):
    num_targets = len(y.columns)
    num_models = len(sorted_evaluation_results[y.columns[0]])
    fig, axes = plt.subplots(nrows=num_targets, ncols=num_models, figsize=(num_models * 5, num_targets * 5))
    
    if num_targets == 1:
        axes = [axes]
    if num_models == 1:
        axes = [[ax] for ax in axes]

    model_abbreviations = {
        'Linear': 'LR',
        'Random Forest': 'RFR',
        'Gradient Boosting': 'GBR',
        'XGBoost': 'XGBR',
        'Gaussian Process': 'GPR',
        'Polynomial': 'PR',
        'Ridge': 'RR',
        'Decision Tree': 'DTR'
    }
    
    target_labels = {
        'Calculated_Band_Gap/eV': ('rPBE band gap (eV)', 'ML band gap (eV)'),
        #'Calculated_a/angstrom': ('rPBE a ≡ b (Å)', 'ML a ≡ b (Å)'),
        'Calculated_a/angstrom': ('rPBE a (Å)', 'ML a (Å)'),
        'Calculated_b/angstrom': ('rPBE b (Å)', 'ML b (Å)'),
        'Calculated_c/angstrom': ('rPBE c (Å)', 'ML c (Å)')
    }

    for target_index, target in enumerate(y.columns):
        for model_index, (name, _) in enumerate(sorted_evaluation_results[target]):
            ax = axes[target_index][model_index]
            data = best_fold_data[target][name]
            
            plot_scatter(ax, data['best_train_actuals'], data['best_train_predictions'], 
                         data['best_test_actuals'], data['best_test_predictions'])
            
            model_abbr = model_abbreviations.get(name, name)
            alphabet_label = chr(97 + model_index)  # 97 is the ASCII code for 'a'
            ax.set_title(f'({alphabet_label}). {model_abbr}', fontsize=18, weight='bold')
            #ax.set_title(f'{model_abbr}', fontsize=18, weight='bold')
            
            x_label, y_label = target_labels.get(target, ('PBE', 'ML Predicted'))
            ax.set_xlabel(x_label, fontsize=18, weight='bold')
            ax.set_ylabel(y_label, fontsize=18, weight='bold')
            
            ax.set_aspect('equal', adjustable='box')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            
            ax.legend(prop={'size': 12, 'weight': 'bold'}, title_fontsize='12', shadow=True, frameon=True)
    
    plt.tight_layout()
    plt.show()

```

**Explanation:**
- **Purpose:** This function creates a grid of scatter plots to visualize the performance of multiple regression models across different target variables, focusing on the best performing fold from cross-validation.
- **Process:** Sets up a grid of subplots (targets as rows, models as columns).
For each target-model combination, plots train and test data from the best fold.
Uses plot_scatter to create scatter plots of true vs. predicted values.
Applies custom formatting, including model abbreviations and axis labels.
- **Output:** A figure showing the best fold performance for each model and target, useful for assessing model performance in cross-validation scenarios.



##### 3.2. Plot Test Set Results for All Models and Targets
```python
def plot_model_results(sorted_evaluation_results, X, y, model_data):
    num_targets = len(y.columns)
    num_models = len(sorted_evaluation_results[y.columns[0]])
    fig, axes = plt.subplots(nrows=num_targets, ncols=num_models, figsize=(num_models * 5, num_targets * 5))
    
    if num_targets == 1:
        axes = [axes]
    if num_models == 1:
        axes = [[ax] for ax in axes]

    model_abbreviations = {
        'Linear': 'LR',
        'Random Forest': 'RFR',
        'Gradient Boosting': 'GBR',
        'XGBoost': 'XGBR',
        'Gaussian Process': 'GPR',
        'Polynomial': 'PR'
    }
    
    target_labels = {
        '<target_1>': ('rPBE <target_1>', 'ML <target_1>'),
        '<target_2>': ('rPBE <target_2>', 'ML <target_2>'),
        '<target_3>': ('rPBE <target_3>', 'ML <target_3>'),
        '<target_4>': ('rPBE <target_4>', 'ML <target_4>')
    }

    for target_index, target in enumerate(y.columns):
        for model_index, (name, _) in enumerate(sorted_evaluation_results[target]):
            ax = axes[target_index][model_index]
            data = model_data[target][name]
            
            scatter = plot_scatter(ax, data['test_actuals'], data['test_predictions'])
            
            model_abbr = model_abbreviations.get(name, name)
            alphabet_label = chr(97 + model_index)  # 97 is the ASCII code for 'a'
            ax.set_title(f'({alphabet_label}). {model_abbr}', fontsize=18, weight='bold')
            
            x_label, y_label = target_labels.get(target, ('rPBE', 'ML Predicted'))
            ax.set_xlabel(x_label, fontsize=18, weight='bold')
            ax.set_ylabel(y_label, fontsize=18, weight='bold')
            
            ax.set_aspect('equal', adjustable='box')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            
            ax.legend(handles=[scatter], prop={'size': 12, 'weight': 'bold'}, loc='upper left')
    
    plt.tight_layout()
    plt.show()
```
**Explanation:**
- **Purpose:** This function also creates a grid of scatter plots for multiple models and targets, but it's designed for scenarios without cross-validation, typically for evaluating on a separate test set.
- **Process:** Sets up a similar grid of subplots as plot_best_fold_all. For each target-model combination, plots only the test data, uses a modified plot_scatter function that only handles test data, and applies similar custom formatting as plot_best_fold_all.
- **Output:** A figure showing the test performance for each model and target, useful for assessing how models perform on unseen data or in extrapolation tasks.


##### 3.3. Scatter Plot Function (train and test)

```python
def plot_scatter(ax, y_train, y_train_pred, y_test, y_test_pred):
    ax.scatter(y_train, y_train_pred, color='blue', label='Train Data', alpha=0.6)
    ax.scatter(y_test, y_test_pred, color='red', label='Test Data', alpha=0.6)
    
    all_vals = np.concatenate([y_train, y_train_pred, y_test, y_test_pred])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    buffer = (max_val - min_val) * 0.05
    
    ax.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer], 'k--', lw=2)
    ax.set_xlim([min_val - buffer, max_val + buffer])
    ax.set_ylim([min_val - buffer, max_val + buffer])
```

**Explanation:**
- **Purpose:** This function creates a scatter plot of the true vs. predicted values for both the training and test sets.
- **Process:** It plots the true vs. predicted values for the training set in blue and the test set in red. It also plots a diagonal line representing the ideal case where predicted values equal true values. The function adjusts the plot limits to include all data points with a small buffer.

##### 3.4. Scatter Plot Function (test only)

```python
def plot_scatter_(ax, y_test, y_test_pred):
    scatter = ax.scatter(y_test, y_test_pred, color='brown', alpha=0.6, label='Extrapolation')
    
    all_vals = np.concatenate([y_test, y_test_pred])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    buffer = (max_val - min_val) * 0.05
    
    ax.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer], 'k--', lw=2)
    ax.set_xlim([min_val - buffer, max_val + buffer])
    ax.set_ylim([min_val - buffer, max_val + buffer])
    
    return scatter
```

**Explanation:**
- **Purpose:** This function creates a scatter plot of the true vs. predicted values for the test set only, specifically designed for extrapolation or specific material analysis.
- **Process:** It plots the true vs. predicted values for the test set in brown, labeling it as 'Extrapolation' (or alternatively, 'm-ZrO₂' if uncommented). Like the first version, it plots a diagonal line for the ideal case and adjusts plot limits.
- **Outputs:** This function returns the scatter plot object, allowing for further customization if needed.

<br/><br/>
#### 4. Define Feature Importance Function

```python
def plot_feature_importance(best_fold_data, X, y):
    import scipy.sparse 
    import xgboost as xgb
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance

    for target in y.columns:
        print(f"\nFeature importance for {target}:")
        for name, data in best_fold_data[target].items():
            model = data['model']

            if hasattr(model, 'named_steps'):
                preprocessor = model.named_steps.get('preprocessor')
                regressor = model.named_steps.get('regressor')
            else:
                preprocessor = None
                regressor = model

            if preprocessor is not None:
                feature_names = get_feature_names(preprocessor, X.columns)
                X_transformed = preprocessor.transform(X)
                if scipy.sparse.issparse(X_transformed):
                    X_transformed = X_transformed.toarray()
                X_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
            else:
                feature_names = X.columns.tolist()
                X_transformed = X

            if hasattr(regressor, 'feature_importances_'):
                importance = regressor.feature_importances_
            elif hasattr(regressor, 'coef_'):
                importance = regressor.coef_
                if importance.ndim > 1:
                    importance = importance.ravel()
            elif isinstance(regressor, xgb.XGBRegressor):
                booster = regressor.get_booster()
                importance_dict = booster.get_score(importance_type='gain')  # Gain metric
                importance = np.zeros(len(feature_names))
                for i, feat in enumerate(feature_names):
                    importance[i] = importance_dict.get(f'f{i}', 0)
            else:
                try:
                    perm_importance = permutation_importance(
                        regressor, X_transformed.values, y[target], n_repeats=10, random_state=100
                    )
                    importance = perm_importance.importances_mean
                except ValueError as e:
                    print(f"Permutation importance failed for {name} model: {e}")
                    continue

            if len(importance) != len(feature_names):
                print(f"  Warning: Number of feature importances ({len(importance)}) does not match number of features ({len(feature_names)}).")
                continue

            print(f"  Raw feature importances for {name} model:")
            for feat, imp in zip(feature_names, importance):
                print(f"    {feat}: {imp:.4f}")

            sorted_idx = np.argsort(np.abs(importance))[::-1]
            sorted_features = np.array(feature_names)[sorted_idx]
            sorted_importance = importance[sorted_idx]

            N = 20
            top_features = sorted_features[:N]
            top_importance = sorted_importance[:N]

            plt.figure(figsize=(12, 6))
            plt.bar(range(len(top_importance)), top_importance, align='center')
            plt.xticks(range(len(top_importance)), top_features, rotation=90)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title(f'Feature Importance for {name} Model - {target}')
            plt.tight_layout()
            plt.show()

            print(f"  {name} model:")
            for feat, imp in zip(top_features, top_importance):
                print(f"    {feat}: {imp:.4f}")
            print()
```
**Explanation:**
- **Purpose:** This function analyzes, displays, and visualizes the importance of features for each model and target variable, providing insights into which features are most influential in the predictions.
- **Process:** Iterates through each target variable and model and extracts feature importance based on the model type:
**Gaussian Process:** Uses permutation importance.
**Polynomial and Linear models:** Uses coefficients.
**Tree-based models (e.g., Random Forest, XGBoost):** Uses built-in feature importance.
Sorts features by their absolute importance, then creates a bar plot of feature importances and prints the numerical importance values for each feature.
- **Output:** For each target and model combination: A bar plot showing feature importances, as well as a printed list of features sorted by importance with their corresponding values.


<br/><br/>
#### 5. Define Models

Define a dictionary of models to be evaluated.

```python
models = {
    'Linear': Pipeline([
        ('preprocessor', preprocessor_without_scaling),
        ('regressor', LinearRegression())
    ]),
    'Polynomial': Pipeline([
        ('preprocessor', preprocessor_without_scaling),
        ('poly', PolynomialFeatures(2)),
        ('regressor', LinearRegression())
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor_with_scaling),
        ('regressor', RandomForestRegressor(random_state=100))
    ]),
    'Gradient Boosting': Pipeline([
        ('preprocessor', preprocessor_with_scaling),
        ('regressor', GradientBoostingRegressor(random_state=100))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor_with_scaling),
        ('regressor', XGBRegressor(random_state=100))
    ]),
    'Gaussian Process': Pipeline([
        ('preprocessor', preprocessor_with_scaling),
        ('regressor', GaussianProcessRegressor(kernel=RBF(), alpha=0.01, random_state=100))
    ]),
    'Ridge': Pipeline([
        ('preprocessor', preprocessor_without_scaling),
        ('regressor', Ridge())
    ]),
    'Decision Tree': Pipeline([
        ('preprocessor', preprocessor_with_scaling),
        ('regressor', DecisionTreeRegressor(random_state=100))
    ]),
    'Neural Network': Pipeline([
        ('preprocessor', preprocessor_with_scaling),
        ('regressor', MLPRegressor( 
            max_iter=500,   
            random_state=100
        ))
    ]),
}
```

**Explanation:**
- **Purpose:** This section defines a dictionary of regression models to be evaluated, providing a diverse set of algorithms for predicting material properties.
- **Process:** Define and configure a diverse set of regression models as scikit-learn Pipelines, combining preprocessing steps and estimators, to be systematically evaluated for predicting material properties.

<br/><br/>
#### 6. Load Data
##### 6.1. Load Data for Cross-Validation
This section outlines how to load the dataset and define features and target variables for cross-validation scenarios.

**Load Dataset**

```python
data = pd.read_excel('<materials_data>.xlsx', sheet_name='<sheet_name>')
```
**Define Features**
```python
features = ["<feature_1>", "<feature_2>", ............, "<feature_n>"]
```
**Define Target Variables**

```python
targets = ['<target_1>', '<target_2>', '<target_3>', '<target_4>']
```
**Prepare Data for Cross-Validation**
 
```python
X = data[features]
y = data[targets]
```

##### 6.2. Load Data for Separate Training and Test Sets

**Load Training Data**


```python
train_data = pd.read_excel('<materials_data>.xlsx', sheet_name='<train_data_sheet_name>')
```

**Load Test Data**

```python
test_data = pd.read_excel('<materials_data>.xlsx', sheet_name='<test_data_sheet_name>')
```

**Define Features**

```python
features = ["<feature_1>", "<feature_2>", ............, "<feature_n>"]
```

**Define Target Variables**
```python
targets = ['<target_1>', '<target_2>', '<target_3>', '<target_4>']
```
**Prepare Training and Test Data**
```python
X_train = train_data[features]
y_train = train_data[targets]
X_test = test_data[features]
y_test = test_data[targets]
```
**Explanation:**
- **Purpose:** This section loads the dataset(s) from Excel file(s) and defines the features and target variables. The process varies depending on whether you're using cross-validation or separate training and test sets.
  
- **Process:** For cross-validation, the Excel file is named '<materials_data>.xlsx', typically using the sheet named '<sheet_name>'. For separate sets, two sheets or files may be used, one for training (e.g., '<train_data_sheet_name>') and one for testing (e.g., '<test_data_sheet_name>').


**Processing Categorical and Numerical Feature**

```python
categorical_features = ["<categorical_feature_1>", "<categorical_feature_2>", ............, "<categorical_feature_n>"]

numerical_features = [col for col in features if col not in categorical_features]

preprocessor_with_scaling = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

preprocessor_without_scaling = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features), 
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

X.loc[:, categorical_features] = X[categorical_features].astype(str)
```
**Explanation:**
- **Purpose:**
  This section processes features by scaling numerical data and encoding categorical data to prepare for model training. Two configurations are provided: one with scaling and one without.
  
- **Process:** categorical_features specifies the categorical columns, while numerical_features includes all others. The preprocessor_with_scaling scales numerical features using StandardScaler and encodes categorical features using OneHotEncoder. The preprocessor_without_scaling passes numerical features unchanged but encodes categorical features the same way.


<br/><br/>
#### 7. Train and Evaluate Models

Train and evaluate the models using validation techniques.

```python

# Leave-one-out CV
evaluation_results, best_fold_data = cross_val_train_eval_models_per_fold(models, X, y, cv=loo)
```
```python
# KFold CV
evaluation_results, best_fold_data = cross_val_train_eval_models_per_fold(models, X, y, cv=5)
```

```python
# No CV
evaluation_results, model_data = evaluate_models(models, X_train, y_train, X_test, y_test)
```


**Explanation:**
- **Purpose:** This section trains and evaluates the models using different cross-validation techniques or a simple train-test split.
- **Process:** The code provides three options for model evaluation: Leave-One-Out Cross-Validation (LOO-CV), K-Fold Cross-Validation (K = 5), and No Cross-Validation. 

<br/><br/>
#### 8. Sort Print Model Performance Metrics

Sort and print the performance metrics for each model and target variable.

```python
sorted_evaluation_results = {target: sorted(evaluation_results[target].items(),
                                            key=lambda x: x[1]['avg_test_metrics']['mse'])
                             for target in y.columns}

for target in y.columns:
    print(f"\nModel Performance Metrics for {target} (sorted by Test MSE):")
    for name, metrics in sorted_evaluation_results[target]:
        avg_train = metrics['avg_train_metrics']
        avg_test = metrics['avg_test_metrics']
        best_train = best_fold_data[target][name]['best_fold_train_metrics']
        best_test = best_fold_data[target][name]['best_fold_test_metrics']

        print(f"{name}:")
        print(f"  Average Train Metrics - MSE = {avg_train['mse']:.4f}, RMSE = {avg_train['rmse']:.4f}, "
              f"MAE = {avg_train['mae']:.4f}, R2 = {avg_train['r2']:.4f}")
        print(f"  Average Test Metrics - MSE = {avg_test['mse']:.4f}, RMSE = {avg_test['rmse']:.4f}, "
              f"MAE = {avg_test['mae']:.4f}, R2 = {avg_test['r2']:.4f}")
        print(f"  Best Fold Train Metrics - MSE = {best_train['mse']:.4f}, RMSE = {best_train['rmse']:.4f}, "
              f"MAE = {best_train['mae']:.4f}, R2 = {best_train['r2']:.4f}")
        print(f"  Best Fold Test Metrics - MSE = {best_test['mse']:.4f}, RMSE = {best_test['rmse']:.4f}, "
              f"MAE = {best_test['mae']:.4f}, R2 = {best_test['r2']:.4f}\n")

        fold_metrics = metrics['fold_metrics']
        sorted_fold_metrics = sorted(fold_metrics, key=lambda x: x['test_metrics']['mse'])
        print(f"  Per-Fold Metrics (sorted by Test MSE):")
        for fold in sorted_fold_metrics:
            fold_index = fold['fold_index']
            test_mse = fold['test_metrics']['mse']
            test_rmse = fold['test_metrics']['rmse']
            test_mae = fold['test_metrics']['mae']
            test_r2 = fold['test_metrics']['r2']
            test_MP_IDs = fold.get('test_MP_IDs', None) 
            if test_MP_IDs is not None:
                unique_test_MP_IDs = set(test_MP_IDs)
                print(f"    Fold {fold_index}: Test MSE = {test_mse:.4f}, RMSE = {test_rmse:.4f}, "
                      f"MAE = {test_mae:.4f}, R2 = {test_r2:.4f}, Unique Test MP-IDs: {unique_test_MP_IDs}")
            else:
                print(f"    Fold {fold_index}: Test MSE = {test_mse:.4f}, RMSE = {test_rmse:.4f}, "
                      f"MAE = {test_mae:.4f}, R2 = {test_r2:.4f}")

```

**Explanation:**
- **Purpose:** This section organizes and presents the performance metrics for each model and target variable. It allows for easy comparison of model performance across different material properties and provides a comprehensive view of how well each model generalizes to unseen data.
- **Process:** For each target, it iterates through the sorted evaluation results and prints the metrics.

<br/><br/>
#### 9. Plot Results

Plot the results.

```python
plot_best_fold_all(sorted_evaluation_results, X, y, best_fold_data)
```
```python
plot_model_results(sorted_evaluation_results, X_test, y_test, model_data)
```

**Explanation:**
- **Purpose:** This section creates visual representations of model performance for all target variables across different models. It allows for a quick, intuitive comparison of how well each model predicts various material properties, and how these predictions compare to the actual values.
- **Process:** Offers two different plotting functions, each serving a specific purpose: **plot_best_fold_all function** when cross-validation is performed, and **plot_model_results** function when a single train-test split is employed (no cross-validation).

<br/><br/>
#### 10. Display Feature Importance

Display the feature importance for the models.

```python
plot_feature_importance(best_fold_data, X, y)
```

**Explanation:**
- **Purpose:** This section displays and plots the feature importance for the various models for each target variable.
- **Process:** It uses the `plot_feature_importance` function to extract, sort, and plot the feature importance for each model and target.

## Conclusion

This notebooks and scripts provides a robust framework for evaluating and comparing the performance of various regression models on multiple target properties. The cross-validation techniques and feature importance analysis offer insights into the model's predictive power and the significance of each feature. By following the steps outlined in this README, users can load their data, define models, train and evaluate them, and visualize the results to make informed decisions about their machine learning models.
