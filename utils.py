import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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