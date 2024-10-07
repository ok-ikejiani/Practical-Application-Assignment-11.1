import pandas as pd
import numpy as np

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns

#Model evaluation
from sklearn.metrics import mean_squared_error, r2_score

def analyze_missing_values(df):
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    print("\nMissing Values Analysis:")
    print("{:<20} {:>10} {:>10}".format("Feature", "#Missing", "%Missing"))
    print("-" * 45)
    for feature, count in missing_values.items():
        percentage = missing_percentage[feature]
        print("{:<20} {:>10} {:>10.2f}%".format(feature, count, percentage))
    
def correlation_analysis(df, target_column, corr_threshold=0.8):
    # Step 1: Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Step 2: Get correlation of all columns with the target column
    target_corr = corr_matrix[target_column].sort_values(ascending=False)
    print(f"Correlation of features with {target_column}:")
    print(target_corr)
    
    # Step 3: Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f"Correlation Matrix")
    plt.show()

    # Step 4: Find highly correlated columns
    highly_corr_columns = []
    columns = df.columns.tolist()
    
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            corr_value = corr_matrix.loc[col1, col2]
            if abs(corr_value) > corr_threshold:
                # Calculate variance difference between the two columns
                variance_diff = df[col1].var() - df[col2].var()
                
                highly_corr_columns.append({
                    "columns": [col1, col2],
                    "correlation_value": corr_value,
                    "variance_delta": variance_diff
                })

    # Step 5: Print and return the list of highly correlated columns
    if highly_corr_columns:
        print("\nHighly correlated columns (above threshold):")
        for entry in highly_corr_columns:
            cols = entry["columns"]
            print(f"Columns: {cols}, Correlation: {entry['correlation_value']:.2f}, Variance Delta: {entry['variance_delta']:.2f}")
        
        # Step 6: Plot bar chart of highly correlated columns with variance deltas
        plt.figure(figsize=(10, 6))
        names = [f"{col_pair[0]} & {col_pair[1]}" for col_pair in [entry["columns"] for entry in highly_corr_columns]]
        correlation_values = [entry["correlation_value"] for entry in highly_corr_columns]
        variance_deltas = [entry["variance_delta"] for entry in highly_corr_columns]

        plt.barh(names, correlation_values, color='lightblue', label='Correlation')
        plt.barh(names, variance_deltas, color='orange', label='Variance Delta', alpha=0.6)
        plt.xlabel('Correlation / Variance Delta')
        plt.title('Highly Correlated Columns with Variance Deltas')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return target_corr, highly_corr_columns

def print_unique_values(df):
    """
    This function takes a DataFrame as input and prints the number of unique values in each column.
    
    Parameters:
    df (pd.DataFrame): Input dataframe to be processed
    
    Returns:
    None
    """
    for column in df.columns:
        unique_count = df[column].nunique()
        print(f"Column '{column}' has {unique_count} unique values.")

def drop_outliers(df, column, min_value, max_value):
    """
    This function takes a dataframe and drops rows where the values in the specified column
    are below the minimum or above the maximum specified values.
    
    Parameters:
    df (pd.DataFrame): Input dataframe to be processed
    column (str): The column name for which outliers need to be removed
    min_value (float): The minimum acceptable value for the specified column
    max_value (float): The maximum acceptable value for the specified column
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    df_filtered = df[(df[column] >= min_value) & (df[column] <= max_value)]
    return df_filtered


def uniques(df, columns):
    unique_values = [df[column].unique().tolist() for column in columns]
    return unique_values


from sklearn.model_selection import GridSearchCV

def train_models(X_train=None, y_train=None, pipeline=None, param_grid=None):
    """
    Trains the models using GridSearchCV and returns an array of best models.

    Args:
        param_grid (dict): The parameter grid to search over.
        pipeline (Pipeline): The pipeline with preprocessing and model steps.
        X_train (pd.DataFrame or np.array): Training feature set.
        y_train (pd.Series or np.array): Training target values.

    Returns:
        best_models (list): List of dictionaries containing best models and associated metadata.
    """
    # Check if all required arguments are provided
    if X_train is None or y_train is None or pipeline is None or param_grid is None:
        raise ValueError("X_train, y_train, pipeline, and param_grid must be provided.")

    # Perform GridSearchCV to find the best models
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_models = []

    # Loop through the models and store best parameters and scores
    for i, params in enumerate(grid_search.cv_results_['params']):
        model_name = type(params['model']).__name__
        best_rmse = np.sqrt(-grid_search.cv_results_['mean_test_score'][i])
        best_r2 = grid_search.cv_results_['mean_test_score'][i]
        best_params = params



        # Store the best model pipeline, name, and training performance
        best_models.append({
            'Model Name': model_name,
            'Best Parameters': best_params,
            'Best Pipeline': grid_search.best_estimator_,
            'Train RMSE (CV)': best_rmse
        })

    return best_models


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def analyze_and_visualize_model_performance(best_models, X_test=None, y_test=None):
    """
    Analyzes and visualizes the performance of the best models on either training or test data.
    If test data is provided, it evaluates the models on the test data.
    If test data is not provided, it uses the pre-computed training data results.

    Args:
        best_models (list): List of dictionaries containing best models and associated metadata.
        X_test (pd.DataFrame or np.array, optional): Test feature set. If None, only training data results are shown.
        y_test (pd.Series or np.array, optional): Test target values. If None, only training data results are shown.
    """
    best_results = {}

    if X_test is not None and y_test is not None:
        # Test data is provided, so we evaluate models on the test set
        for model_info in best_models:
            model_pipeline = model_info['Best Pipeline']
            model_name = model_info['Model Name']

            # Make predictions on the test set
            y_pred_test = model_pipeline.predict(X_test)

            # Calculate test RMSE and R² score
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_r2 = r2_score(y_test, y_pred_test)

            # Store the best result for each model (based on RMSE)
            if model_name not in best_results or test_rmse < best_results[model_name]['RMSE']:
                best_results[model_name] = {
                    'RMSE': test_rmse,
                    'R²': test_r2,
                    'Params': model_info['Best Parameters']
                }
    else:
        # Test data is not provided, so we display training data results
        for model in best_models:
            model_name = model['Model Name']
            rmse = model['Train RMSE (CV)']
            # Store the best result for each model (based on RMSE)
            if model_name not in best_results or rmse < best_results[model_name]['RMSE']:
                best_results[model_name] = {
                    'RMSE': rmse,
                    'R²': None,  # R² isn't typically computed in cross-validation
                    'Params': model['Best Parameters']
                }

    # Display the results as a table and plot the comparison
    display_best_results(best_results)


from tabulate import tabulate

def display_best_results(best_results):
    """
    Displays the best results in a tabular format using tabulate and plots the RMSE comparison.

    Args:
        best_results (dict): Dictionary containing model names and their best RMSE, R², and parameters.
    """
    # Prepare data for display
    table_data = []
    for model_name, result in best_results.items():
        table_data.append({
            'Model': model_name,
            'RMSE': result['RMSE'],
            'R²': result['R²'] if result['R²'] is not None else 'N/A',
            'Params': result['Params']
        })

    # Convert to DataFrame for better readability
    df = pd.DataFrame(table_data)

    # Use tabulate to format the DataFrame in a cleaner table format
    print("\nBest Performance Comparison:")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False, numalign='right', floatfmt=".4f"))

    # Plot comparison of models based on RMSE
    plt.figure(figsize=(10, 6))
    plt.bar(df['Model'], df['RMSE'], color='lightcoral')
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title(f'Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

def analyze_and_visualize_model_performances(*, best_models, X, y):
    """
    Analyzes and visualizes the performance of the best models using the provided feature set (X) and target values (y).
    This function accepts both training or test data and calculates necessary values.

    Args:
        best_models (list): List of dictionaries containing best models and associated metadata.
        X (pd.DataFrame or np.array): Feature set (can be training or test).
        y (pd.Series or np.array): Target values (can be training or test).
    """
    best_results = {}

    for model_info in best_models:
        model_pipeline = model_info['Best Pipeline']
        model_name = model_info['Model Name']

        # Make predictions on the provided feature set (X)
        y_pred = model_pipeline.predict(X)

        # Calculate RMSE and R² score
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        # Store the result for each model (based on RMSE)
        best_results[model_name] = {
            'RMSE': rmse,
            'R²': r2,
            'Params': model_info['Best Parameters']
        }

    # Display the results as a table and plot the comparison
    display_best_results2(best_results)

def display_best_results2(best_results):
    """
    Displays the best results in a tabular format using tabulate and plots the RMSE comparison.

    Args:
        best_results (dict): Dictionary containing model names and their best RMSE, R², and parameters.
    """
    # Prepare data for display
    table_data = []
    for model_name, result in best_results.items():
        table_data.append({
            'Model': model_name,
            'RMSE': result['RMSE'],
            'R²': result['R²'] if result['R²'] is not None else 'N/A',
            'Params': result['Params']
        })

    # Convert to DataFrame for better readability
    df = pd.DataFrame(table_data)

    # Use tabulate to format the DataFrame in a cleaner table format
    print("\nBest Performance Comparison:")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False, numalign='right', floatfmt=".4f"))

    # Plot comparison of models based on RMSE
    plt.figure(figsize=(10, 6))
    plt.bar(df['Model'], df['RMSE'], color='lightcoral')
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title(f'Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
