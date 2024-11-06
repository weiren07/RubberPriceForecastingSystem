import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from prettytable import PrettyTable
from scipy.stats import boxcox
import os


DATA_URL = 'data/raw_data.csv' 
OUTPUT_CSV = 'data/transform_preprocessed_data.csv'
NUMERIC_COLUMNS = ['usd/rm price', 'wti price', 'SMR20']
CATEGORICAL_COLUMNS = []


def load_data(url):
    """Load dataset from a given URL."""
    try:
        df = pd.read_csv(url)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def display_head(df, n=5):
    """Display the first n rows of the DataFrame."""
    print("\n--- DataFrame Head ---")
    print(df.head(n))

def display_info(df):
    """Display DataFrame information and missing values."""
    print("\n--- DataFrame Info ---")
    df.info()
    print("\n--- Missing Values per Column ---")
    print(df.isnull().sum())

def display_shape(df):
    """Display the shape of the DataFrame."""
    print(f"\n--- DataFrame Shape: {df.shape} ---")

def display_data_types(df):
    """Display data types of each column."""
    print("\n--- Data Types ---")
    print(df.dtypes)

def drop_missing_values(df):
    """Remove rows with any missing values."""
    initial_shape = df.shape
    df_cleaned = df.dropna()
    final_shape = df_cleaned.shape
    print(f"\nDropped missing values: {initial_shape} -> {final_shape}")
    return df_cleaned

def create_pretty_table(df, numeric_cols):
    """Create PrettyTable for numeric and categorical columns."""
    # Numeric Columns Table
    numeric_table = PrettyTable()
    numeric_table.field_names = ['Column', 'Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max']
    for col in numeric_cols:
        numeric_table.add_row([
            col,
            df[col].count(),
            round(df[col].mean(), 2),
            round(df[col].std(), 2),
            round(df[col].min(), 2),
            round(df[col].quantile(0.25), 2),
            round(df[col].quantile(0.5), 2),
            round(df[col].quantile(0.75), 2),
            round(df[col].max(), 2)
        ])
    
    # Categorical Columns Table
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    categorical_table = PrettyTable()
    categorical_table.field_names = ['Column', 'Count', 'Unique']
    for col in categorical_cols:
        categorical_table.add_row([
            col,
            df[col].count(),
            df[col].nunique()
        ])
    
    print("\n--- Categorical Columns ---")
    print(categorical_table)
    print("\n--- Numeric Columns ---")
    print(numeric_table)

def check_unique_dates(df, date_column='date'):
    """Check if all dates in the date_column are unique."""
    if date_column not in df.columns:
        print(f"Column '{date_column}' does not exist in the DataFrame.")
        return df
    if df[date_column].is_unique:
        print("\nAll dates are unique.")
    else:
        duplicates = df[df.duplicated(subset=date_column)][date_column].unique()
        print("\nDuplicate dates found:", duplicates)
        df = df.drop_duplicates(subset=date_column)
        print(f"Duplicates removed. New shape: {df.shape}")
    return df

def remove_negative_values(df, numeric_cols):
    """Remove rows with negative values in specified numeric columns."""
    initial_shape = df.shape
    df_cleaned = df[(df[numeric_cols] >= 0).all(axis=1)]
    final_shape = df_cleaned.shape
    print(f"\nRemoved negative values: {initial_shape} -> {final_shape}")
    return df_cleaned

def plot_histograms(df, numeric_cols):
    """Plot histograms for each numeric column with individual colors."""
    num_cols = len(numeric_cols)
    
    # Define a color palette (extend if more columns are added)
    color_palette = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    # Check if there are enough colors in the palette
    if num_cols > len(color_palette):
        # If not enough, extend the palette by repeating colors
        extended_palette = color_palette * (num_cols // len(color_palette) + 1)
    else:
        extended_palette = color_palette
    
    fig, axs = plt.subplots(num_cols, 1, figsize=(10, 5 * num_cols))
    
    if num_cols == 1:
        axs = [axs]
    
    for ax, col, color in zip(axs, numeric_cols, extended_palette):
        ax.hist(df[col], bins=50, color=color, alpha=0.7)
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def calculate_outliers_iqr(series):
    """Calculate outliers using the IQR method for a pandas Series."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers

def display_outliers(df, numeric_cols):
    """Display the number of outliers."""
    print("\n--- Outliers Detected ---")
    for col in numeric_cols:
        outliers = calculate_outliers_iqr(df[col])
        print(f"{col}: {len(outliers)} outliers")

def apply_boxcox_transformation(df, numeric_cols):
    """Apply Box-Cox transformation to specified numeric columns."""
    transformed_data = {}
    lambda_values = {}
    for col in numeric_cols:
        try:
            # Box-Cox requires all data to be positive
            if (df[col] <= 0).any():
                print(f"Cannot apply Box-Cox transformation to '{col}' because it contains non-positive values.")
                continue
            transformed, lambda_val = boxcox(df[col])
            df[col] = transformed
            transformed_data[col] = transformed
            lambda_values[col] = lambda_val
            print(f"Box-Cox transformation applied to '{col}' with lambda={lambda_val:.4f}")
        except ValueError as e:
            print(f"Box-Cox transformation failed for '{col}': {e}")
    return df, lambda_values

def plot_density(df, numeric_cols):
    """Plot density plots"""
    num_cols = len(numeric_cols)
    fig, axs = plt.subplots(num_cols, 1, figsize=(10, 5 * num_cols))
    
    if num_cols == 1:
        axs = [axs]
    
    for ax, col in zip(axs, numeric_cols):
        sns.kdeplot(df[col], ax=ax, fill=True)
        ax.set_title(f'{col} Density Plot')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    """Plot a heatmap of the correlation matrix for numeric columns only."""
    numeric_df = df.select_dtypes(include=[np.number])

    print("\n--- Numeric Columns for Correlation ---")
    print(numeric_df.columns)
    
    # Compute the correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()

def save_dataframe(df, filepath):
    """Save the DataFrame to a CSV file."""
    try:
        df.to_csv(filepath, index=False)
        print(f"\nDataFrame saved to '{filepath}'.")
    except Exception as e:
        print(f"Error saving DataFrame: {e}")

# ------------------------------
# Main Execution Flow
# ------------------------------

def main():
    # Load Data
    df = load_data(DATA_URL)
    if df is None:
        return

    # Initial Data Exploration
    display_head(df)
    display_info(df)
    display_shape(df)
    display_data_types(df)

    # Data Cleaning
    df = drop_missing_values(df)
    display_shape(df)

    # Display Statistical Tables
    create_pretty_table(df, NUMERIC_COLUMNS)

    # Check and Remove Duplicate Dates
    df = check_unique_dates(df, date_column='date')
    display_shape(df)

    # Remove Negative Values
    df = remove_negative_values(df, NUMERIC_COLUMNS)
    display_shape(df)

    # Re-display Statistical Tables after Cleaning
    create_pretty_table(df, NUMERIC_COLUMNS)

    # Save Cleaned Data (Optional)
    save_dataframe(df, OUTPUT_CSV)

    # Visualization
    plot_histograms(df, NUMERIC_COLUMNS)

    # Outlier Detection
    display_outliers(df, NUMERIC_COLUMNS)

    # Box-Cox Transformation
    df, lambdas = apply_boxcox_transformation(df, NUMERIC_COLUMNS)

    # Save Transformed Data (Optional)
    transformed_csv = 'transformed_' + OUTPUT_CSV
    save_dataframe(df, transformed_csv)

    # Additional Visualizations
    df.hist(bins=10, figsize=(15, 10), color='skyblue', edgecolor='black')
    plt.suptitle('Univariate Histograms', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    plot_density(df, NUMERIC_COLUMNS)

    plot_correlation_heatmap(df)

if __name__ == "__main__":
    main()
