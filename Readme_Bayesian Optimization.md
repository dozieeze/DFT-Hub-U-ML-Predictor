This README explains the Python script that uses Polynomial Regression to predict material properties and optimize Hubbard U parameters. The script is designed to work with material science data, specifically for predicting properties like band gap and lattice constants.

#### 1. Model Training

```python
pr_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
multi_target_model = MultiOutputRegressor(pr_model)
multi_target_model.fit(X, y)
```

This section creates and trains the Polynomial Regression model:

A pipeline is created with PolynomialFeatures(2) (for quadratic terms) and LinearRegression().
MultiOutputRegressor is used to handle multiple target variables simultaneously.
The model is trained on the input data X and target variables y.

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
This section defines a function to extract feature coefficients from the trained model:

It iterates through each estimator in the multi-output model.
Extracts coefficients and intercepts for each target variable.
Returns a DataFrame of coefficients and an array of intercepts.
The extracted coefficients and intercepts are then printed.

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

target_names = ['Calculated_Band_Gap/eV', 'Calculated_a/angstrom', 'Calculated_b/angstrom', 'Calculated_c/angstrom']
plot_feature_coefficients(coefficients, intercept, target_names)

```
This section visualizes the feature coefficients:

A function is defined to create bar plots of feature coefficients for each target variable.
The function is called with the extracted coefficients, intercepts, and target names.
This visualization helps in understanding the importance of each feature for predicting different material properties.

#### 4. New Data Prediction
```python
new_data = pd.read_excel('All System Model May 2024.xlsx', sheet_name='C-CeO2')
X_new = new_data[features]

# Experimental values for Cubic CeO2
exp_band_gap = 3.2
exp_lattice_a = 5.411
exp_lattice_b = 5.411
exp_lattice_c = 5.411

experimental_values = np.array([exp_band_gap, exp_lattice_a, exp_lattice_b, exp_lattice_c])

# Define the weights
weights = np.array([1, 1, 1, 1])  # Equal weights for all properties
This section prepares for prediction on new data:
```
Loads new data from an Excel file.
Sets experimental values for comparison (in this case, for Cubic CeO2).
Defines weights for each property in the optimization process.



#### 5. Optimization of Hubbard U Parameters
```python
def calculate_weighted_ape(predicted, experimental, weights):
    absolute_percentage_errors = np.abs((predicted - experimental) / experimental)
    weighted_absolute_percentage_errors = np.sum(weights * absolute_percentage_errors) / np.sum(weights) 
    return weighted_absolute_percentage_errors

search_space_both = [
    Real(0.01, 40.00, name='up_value'),
    Real(0.01, 40.00, name='ud_value')
]

search_space_ud_only = [
    Real(0.01, 40.00, name='ud_value')
]

@use_named_args(search_space_both)
def objective_both(up_value, ud_value):
    X_new_temp = X_new.copy()
    X_new_temp['Up_Value/eV'] = up_value
    X_new_temp['Ud_Value/eV'] = ud_value
    
    y_pred = multi_target_model.predict(X_new_temp)
    loss = calculate_weighted_ape(y_pred, experimental_values, weights)
    return loss

@use_named_args(search_space_ud_only)
def objective_ud_only(ud_value):
    X_new_temp = X_new.copy()
    X_new_temp['Up_Value/eV'] = 0  # Fix Up to zero
    X_new_temp['Ud_Value/eV'] = ud_value
    
    y_pred = multi_target_model.predict(X_new_temp)
    loss = calculate_weighted_ape(y_pred, experimental_values, weights)
    return loss

result_both = gp_minimize(objective_both, search_space_both, n_calls=200, random_state=100)
result_ud_only = gp_minimize(objective_ud_only, search_space_ud_only, n_calls=50, random_state=100)

```

This section performs Bayesian optimization to find optimal Hubbard U parameters:

Defines a function to calculate Weighted Mean Absolute Percentage Error (WMAPE).
Sets up two search spaces: one for optimizing both Up and Ud, another for optimizing only Ud.
Defines objective functions for both cases.
Uses gp_minimize for Bayesian optimization to find the best parameters.

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

This section evaluates and prints the optimization results:

Defines a function to evaluate the results and print detailed information.
Calls this function for both optimization cases (varying both Up and Ud, and varying only Ud).
Prints the best-found parameters, predicted values, and percentage differences from experimental values.

This script provides a comprehensive approach to predicting material properties using Polynomial Regression and optimizing Hubbard U parameters. It can be adapted for various materials and properties by adjusting the input data, experimental values, and optimization parameters.
