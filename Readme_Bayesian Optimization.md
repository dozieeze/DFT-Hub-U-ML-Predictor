This README explains the Notebook that uses Polynomial Regression to predict material properties and optimize Hubbard U parameters. The script is designed to work with material data, specifically for predicting properties like band gap and lattice constants.

#### 1. Model Training

```python
pr_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
multi_target_model = MultiOutputRegressor(pr_model)
multi_target_model.fit(X, y)
```

**Explanation:** This step creates and trains a Polynomial Regression model to predict multiple material properties.  
- **Purpose:** To develop a model capable of simultaneously predicting various material properties using polynomial regression.  
- **Process:** Combines PolynomialFeatures (degree 2) with LinearRegression in a pipeline and wraps it in a MultiOutputRegressor to handle multiple target variables. Fits the model to input data (X) and target variables (y).  
- **Output:** A trained multi-target Polynomial Regression model.

#### 2. Feature Coefficient Extraction
```python
def extract_feature_coefficients(model, feature_names):
    coefficients_list = []
    intercepts = []
    for estimator in model.estimators_:
        linear_regressor = estimator.named_steps['linearregression']
        poly_features = estimator.named_steps['polynomialfeatures']
        feature_names_poly = poly_features.get_feature_names_out(feature_names)
        coefficients = linear_regressor.coef_
        intercept = linear_regressor.intercept_
        coefficients_list.append(coefficients)
        intercepts.append(intercept)
    
    return pd.DataFrame(np.array(coefficients_list).reshape(len(model.estimators_), -1), columns=feature_names_poly), np.array(intercepts)

coefficients, intercept = extract_feature_coefficients(multi_target_model, features)
print("Feature Coefficients:")
print(coefficients)
print("Intercepts:")
print(intercept)
```
**Explanation:** This step extracts and displays the coefficients and intercepts of the trained model for each target variable.  
- **Purpose:** To analyze the significance of each feature in predicting target variables.  
- **Process:** Iterates through each estimator in the multi-output model, extracts coefficients and intercepts for polynomial features, and organizes these into a DataFrame and array.  
- **Output:** A DataFrame of coefficients for each feature and target, an array of intercepts for each target, and a printed summary of coefficients and intercepts.


#### 3. Visualization of Feature Coefficients
```python
def plot_feature_coefficients(coefficients, intercept, target_names):
    for i, target in enumerate(target_names):
        plt.figure(figsize=(10, 6))
        plt.bar(coefficients.columns, coefficients.iloc[i])
        plt.title(f'Feature Coefficients for {target}')
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
```

```python
target_names = ['Calculated_Band_Gap/eV', 'Calculated_a/angstrom', 'Calculated_b/angstrom', 'Calculated_c/angstrom']
plot_feature_coefficients(coefficients, intercept, target_names)
```
**Explanation:** This step visualizes the importance of each feature in predicting different material properties.  
- **Purpose:** To provide a visual representation of each feature's impact on the prediction of various material properties.  
- **Process:** Creates a bar plot for each target variable where each bar represents the magnitude of a feature's coefficient.  
- **Output:** A series of bar plots, one for each target variable, showing the relative importance of features.

#### 4. New Data Prediction
```python
new_data = pd.read_excel('<materials_data>.xlsx', sheet_name='<sheet_name>')
X_new = new_data[features]
```

```python
#exp_band_gap, exp_lattice_a, exp_lattice_b, exp_lattice_c = <material_exp_band_gap>, <material_exp_lattice_a>, <material_exp_lattice_b>, <material_exp_lattice_c>
experimental_values = np.array([exp_band_gap, exp_lattice_a, exp_lattice_b, exp_lattice_c])
```

```python
weights = np.array([1, 1, 1, 1])  # Equal weights for all properties
weights = np.array([1, 0, 0, 0])  # Emphasis on band gap
```

**Explanation:** This step prepares for making predictions on new, unseen data and establishes comparison metrics.  
- **Purpose:** To set up the prediction framework for new data and define metrics for comparing predicted and experimental values.  
- **Process:** Loads new data from an Excel file, defines experimental values for comparison, and sets weights for different properties in the optimization process.  
- **Output:** Prepared new data (X_new) and defined experimental values and weights for comparison.




#### 5. Optimization of Hubbard U Parameters
```python
def calculate_weighted_ape(predicted, experimental, weights):
    absolute_percentage_errors = np.abs((predicted - experimental) / experimental)
    weighted_absolute_percentage_errors = np.sum(weights * absolute_percentage_errors) / np.sum(weights) 
    return weighted_absolute_percentage_errors
```
```python
search_space_both = [
    Real(0.01, 40.00, name='up_value'),
    Real(0.01, 40.00, name='ud_value')
]

search_space_ud_only = [
    Real(0.01, 40.00, name='ud_value')
]
```
```python
@use_named_args(search_space_both)
def objective_both(up_value, ud_value):
    X_new_temp = X_new.copy()
    X_new_temp['Up_Value/eV'] = up_value
    X_new_temp['Ud_Value/eV'] = ud_value
    
    y_pred = multi_target_model.predict(X_new_temp)
    loss = calculate_weighted_ape(y_pred, experimental_values, weights)
    return loss
```
```python
@use_named_args(search_space_ud_only)
def objective_ud_only(ud_value):
    X_new_temp = X_new.copy()
    X_new_temp['Up_Value/eV'] = 0  # Fix Up to zero
    X_new_temp['Ud_Value/eV'] = ud_value
    
    y_pred = multi_target_model.predict(X_new_temp)
    loss = calculate_weighted_ape(y_pred, experimental_values, weights)
    return loss
```
```python
result_both = gp_minimize(objective_both, search_space_both, n_calls=200, random_state=100)
result_ud_only = gp_minimize(objective_ud_only, search_space_ud_only, n_calls=200, random_state=100)
```

**Explanation:** This step optimizes the Hubbard U parameters to minimize discrepancies between predicted and experimental values.  
- **Purpose:** To determine the optimal Hubbard U parameters (Up and Ud/f) that reduce the difference between model predictions and experimental results.  
- **Process:** Defines a weighted mean absolute percentage error (WMAPE) function, sets up two search spaces (one for both Up and Ud/f, another for Ud/f only), creates objective functions for each scenario, and performs Bayesian optimization using gp_minimize.  
- **Output:** Optimization results containing the best-found Up and Ud/f values for both scenarios.


#### 6. Results Evaluation

```python
def evaluate_and_print(result, case_name):
    if len(result.x) == 2:
        best_up_value, best_ud_value = result.x
    else:
        best_up_value, best_ud_value = 0, result.x[0]
    
    X_new_temp = X_new.copy()
    X_new_temp['Up_Value/eV'] = best_up_value
    X_new_temp['Ud_Value/eV'] = best_ud_value
    best_pred = multi_target_model.predict(X_new_temp)[0]
    
    percentage_differences = 100 * (best_pred - experimental_values) / experimental_values
    
    print(f"\n{case_name} results:")
    print(f"Up_Value/eV: {best_up_value:.2f}, Ud_Value/eV: {best_ud_value:.2f}")
    print(f"Predicted values: Band Gap: {best_pred[0]:.4f} eV, Lattice constant a: {best_pred[1]:.4f} Å, Lattice constant b: {best_pred[2]:.4f} Å, Lattice constant c: {best_pred[3]:.4f} Å")
    print(f"Best loss: {result.fun:.10f}")
    print(f"Percentage differences: Band Gap: {percentage_differences[0]:.2f}%, Lattice constant a: {percentage_differences[1]:.2f}%, Lattice constant b: {percentage_differences[2]:.2f}%, Lattice constant c: {percentage_differences[3]:.2f}%")

evaluate_and_print(result_both, "Original case (Both Up and Ud varying)")
evaluate_and_print(result_ud_only, "Modified case (Up fixed to zero, Ud varying)")
```

**Explanation:** This step evaluates and presents the results of the optimization of Hubbard U parameters.  
- **Purpose:** To assess the effectiveness of the optimized Hubbard U parameters in predicting material properties.  
- **Process:** Uses the optimized parameters to make predictions on new data, calculates percentage differences from experimental values, and provides a detailed summary including optimal parameters, predicted values, and percentage differences.  
- **Output:** A printed summary for each optimization scenario, including the best Up and Ud/f values, predicted material properties, percentage differences from experimental values, and overall loss (WMAPE).

This script provides a comprehensive approach to predicting material properties using Polynomial Regression and optimizing Hubbard U parameters. It can be adapted for various regressors, materials and properties by adjusting the regressor, input data, experimental values, and optimization parameters.
