"""
Data Processing Module

This module contains functions for cleaning and preprocessing the Mall Customers dataset.

Author: Nino Gagnidze
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw data from CSV file.

    Parameters:
    -----------
    file_path : str
        Path to the raw CSV file

    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values in the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        Summary of missing values
    """
    missing_summary = pd.DataFrame(
        {
            "Missing_Count": df.isnull().sum(),
            "Missing_Percentage": (df.isnull().sum() / len(df) * 100).round(2),
        }
    )
    return missing_summary[missing_summary["Missing_Count"] > 0]


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Handle missing values in the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with potential missing values
    strategy : str, optional
        Strategy for filling missing values ('mean', 'median', 'mode', 'drop')
        Default is 'mean'

    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    df_clean = df.copy()

    if df_clean.isnull().sum().sum() == 0:
        print("No missing values found.")
        return df_clean

    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns

    if strategy == "mean":
        df_clean[numerical_cols] = df_clean[numerical_cols].fillna(
            df_clean[numerical_cols].mean()
        )
    elif strategy == "median":
        df_clean[numerical_cols] = df_clean[numerical_cols].fillna(
            df_clean[numerical_cols].median()
        )
    elif strategy == "mode":
        for col in df_clean.columns:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    elif strategy == "drop":
        df_clean = df_clean.dropna()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"Missing values handled using {strategy} strategy.")
    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        Dataframe without duplicates
    """
    initial_shape = df.shape
    df_clean = df.drop_duplicates()
    final_shape = df_clean.shape

    removed_count = initial_shape[0] - final_shape[0]
    print(f"Removed {removed_count} duplicate rows.")

    return df_clean


def detect_outliers_iqr(
    df: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to check for outliers

    Returns:
    --------
    Tuple[pd.DataFrame, Dict]
        Tuple containing outlier rows and statistics dictionary
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    stats = {
        "column": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": len(outliers),
        "outlier_percentage": (len(outliers) / len(df)) * 100,
    }

    return outliers, stats


def handle_outliers(
    df: pd.DataFrame, column: str, method: str = "keep"
) -> pd.DataFrame:
    """
    Handle outliers in a specified column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to handle outliers
    method : str, optional
        Method to handle outliers: 'keep', 'remove', 'cap'
        Default is 'keep'

    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers handled
    """
    df_clean = df.copy()
    outliers, stats = detect_outliers_iqr(df_clean, column)

    if method == "keep":
        print(f"{column}: Keeping {stats['outlier_count']} outliers.")
        return df_clean

    elif method == "remove":
        df_clean = df_clean[~df_clean.index.isin(outliers.index)]
        print(f"{column}: Removed {stats['outlier_count']} outliers.")
        return df_clean

    elif method == "cap":
        df_clean.loc[df_clean[column] < stats["lower_bound"], column] = stats[
            "lower_bound"
        ]
        df_clean.loc[df_clean[column] > stats["upper_bound"], column] = stats[
            "upper_bound"
        ]
        print(f"{column}: Capped {stats['outlier_count']} outliers.")
        return df_clean

    else:
        raise ValueError(f"Unknown method: {method}")


def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create age group categories.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'Age' column

    Returns:
    --------
    pd.DataFrame
        Dataframe with 'Age_Group' column added
    """
    df_copy = df.copy()

    bins = [0, 25, 35, 50, 100]
    labels = ["Young (18-25)", "Adult (26-35)", "Middle-Aged (36-50)", "Senior (50+)"]

    df_copy["Age_Group"] = pd.cut(
        df_copy["Age"], bins=bins, labels=labels, include_lowest=True
    )

    print("Age groups created successfully.")
    return df_copy


def create_income_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create income category groups.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'Annual Income (k$)' column

    Returns:
    --------
    pd.DataFrame
        Dataframe with 'Income_Category' column added
    """
    df_copy = df.copy()

    bins = [0, 40, 70, 100, 200]
    labels = [
        "Low Income (<40k)",
        "Medium Income (40-70k)",
        "High Income (70-100k)",
        "Very High Income (>100k)",
    ]

    df_copy["Income_Category"] = pd.cut(
        df_copy["Annual Income (k$)"], bins=bins, labels=labels, include_lowest=True
    )

    print("Income categories created successfully.")
    return df_copy


def create_spending_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create spending score category groups.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'Spending Score (1-100)' column

    Returns:
    --------
    pd.DataFrame
        Dataframe with 'Spending_Category' column added
    """
    df_copy = df.copy()

    bins = [0, 35, 65, 100]
    labels = ["Low Spender", "Medium Spender", "High Spender"]

    df_copy["Spending_Category"] = pd.cut(
        df_copy["Spending Score (1-100)"], bins=bins, labels=labels, include_lowest=True
    )

    print("Spending categories created successfully.")
    return df_copy


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features for machine learning.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with categorical columns

    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded categorical features
    """
    df_encoded = df.copy()

    # Encode Gender (Male=1, Female=0)
    if "Gender" in df_encoded.columns:
        df_encoded["Gender_Encoded"] = df_encoded["Gender"].map(
            {"Male": 1, "Female": 0}
        )
        print("Gender encoded: Male=1, Female=0")

    return df_encoded


def preprocess_pipeline(
    df: pd.DataFrame,
    handle_missing: bool = True,
    remove_duplicates_flag: bool = True,
    handle_outliers_flag: bool = False,
    outlier_method: str = "keep",
    create_features: bool = True,
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw input dataframe
    handle_missing : bool, optional
        Whether to handle missing values
    remove_duplicates_flag : bool, optional
        Whether to remove duplicate rows
    handle_outliers_flag : bool, optional
        Whether to handle outliers
    outlier_method : str, optional
        Method for handling outliers ('keep', 'remove', 'cap')
    create_features : bool, optional
        Whether to create derived features

    Returns:
    --------
    pd.DataFrame
        Fully preprocessed dataframe
    """
    print("Starting preprocessing pipeline...")
    print("=" * 80)

    df_processed = df.copy()

    # Handle missing values
    if handle_missing:
        print("\n1. Handling missing values...")
        df_processed = handle_missing_values(df_processed, strategy="mean")

    # Remove duplicates
    if remove_duplicates_flag:
        print("\n2. Removing duplicates...")
        df_processed = remove_duplicates(df_processed)

    # Handle outliers
    if handle_outliers_flag:
        print("\n3. Handling outliers...")
        numerical_features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
        for col in numerical_features:
            df_processed = handle_outliers(df_processed, col, method=outlier_method)

    # Create derived features
    if create_features:
        print("\n4. Creating derived features...")
        df_processed = create_age_groups(df_processed)
        df_processed = create_income_categories(df_processed)
        df_processed = create_spending_categories(df_processed)
        df_processed = encode_categorical_features(df_processed)

    print("\n" + "=" * 80)
    print(f"Preprocessing complete! Final shape: {df_processed.shape}")

    return df_processed


def save_processed_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save processed data to CSV file.

    Parameters:
    -----------
    df : pd.DataFrame
        Processed dataframe to save
    file_path : str
        Path where to save the processed data
    """
    df.to_csv(file_path, index=False)
    print(f"Processed data saved to: {file_path}")


def generate_preprocessing_report(
    df_original: pd.DataFrame, df_processed: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generate a report summarizing preprocessing steps.

    Parameters:
    -----------
    df_original : pd.DataFrame
        Original dataframe before preprocessing
    df_processed : pd.DataFrame
        Dataframe after preprocessing

    Returns:
    --------
    Dict[str, Any]
        Preprocessing report dictionary
    """
    report = {
        "Original_Shape": df_original.shape,
        "Processed_Shape": df_processed.shape,
        "Rows_Removed": df_original.shape[0] - df_processed.shape[0],
        "Features_Added": df_processed.shape[1] - df_original.shape[1],
        "Original_Columns": df_original.columns.tolist(),
        "Processed_Columns": df_processed.columns.tolist(),
        "New_Features": [
            col for col in df_processed.columns if col not in df_original.columns
        ],
    }

    return report
