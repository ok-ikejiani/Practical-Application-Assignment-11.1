import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_missing(df):
    # Check for any rows missing important data and print the percentage of missing values
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    # Print the total number and percentage of missing values for each column
    # print("Missing Values:\n", missing_values)
    print("\nPercentage of Missing Values:\n", missing_percentage)
    
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

# Function to drop all columns from a specified list of columns
def drop_columns(data, columns):
    # Drop rows where any of the specified columns have null values
    data = data.drop(columns, axis=1)
    return data

# Function to drop null values from a specified list of columns
def drop_nulls_from_columns(data, columns):
    # Drop rows where any of the specified columns have null values
    data = data.dropna(subset=columns)
    return data
def clean_and_encode(df):
    """
    This function takes a dataframe, replaces missing/null values with 'unknown',
    and applies one-hot encoding on the non-numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe to be processed
    
    Returns:
    pd.DataFrame: Cleaned and one-hot encoded dataframe
    """
    # # Replace missing/null values with 'unknown' for non-numeric columns
    # df_cleaned = df.fillna('unknown')
    
    # # Select non-numeric columns
    # non_numeric_cols = df_cleaned.select_dtypes(include=['object']).columns
    # print(non_numeric_cols)
    
    # # Apply one-hot encoding to non-numeric columns
    # df_encoded = pd.get_dummies(df_cleaned, columns=non_numeric_cols, drop_first=True)
    
    # Separate numeric and non-numeric columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    
    # Fill missing values for non-numeric columns with 'unknown'
    df[non_numeric_cols] = df[non_numeric_cols].fillna('unknown')
    
    # Apply one-hot encoding to non-numeric columns only
    df_encoded = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)

    return df_encoded

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