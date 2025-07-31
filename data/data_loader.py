import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("data/housing.csv")
    print('✅ Data loaded')
    return df

def drop_outliers(df, iqr_multiplier=1.5):
    """
    Remove outliers from a pandas DataFrame using IQR method.
    
    Args:
        df: pandas DataFrame
        iqr_multiplier: multiplier for IQR to determine outlier bounds (default: 1.5)
    
    Returns:
        pandas DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    # Get numeric columns only
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        # Remove outliers
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    
    rows_removed = len(df) - len(df_clean)
    print(f'✅ 🗑️ {rows_removed} rows removed as outliers')
    
    return df_clean


def preprocess_data(df, one_hot_encode=True):
    # Count missing values before dropping
    rows_before = len(df)
    
    # Drop the rows with missing values
    df = df.dropna()
    rows_after = len(df)
    rows_dropped = rows_before - rows_after
    
    print(f'✅ 🗑️ {rows_dropped} rows dropped because of missing values')

    if one_hot_encode:
        # One hot encoding for categorical data
        df_encoded = pd.get_dummies(df, columns=['ocean_proximity'])
    else:
        # Drop the ocean_proximity column
        df_encoded = df.drop(columns=['ocean_proximity'])


    return df_encoded

