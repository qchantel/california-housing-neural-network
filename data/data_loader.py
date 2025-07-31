import pandas as pd

def load_data():
    df = pd.read_csv("data/housing.csv")
    print('✅ Data loaded')
    return df

def preprocess_data(df):
    # Count missing values before dropping
    rows_before = len(df)
    
    # Drop the rows with missing values
    df = df.dropna()
    rows_after = len(df)
    rows_dropped = rows_before - rows_after
    
    print(f'✅ 🗑️ {rows_dropped} rows dropped because of missing values')

    # One hot encoding for categorical data
    df_encoded = pd.get_dummies(df, columns=['ocean_proximity'])


    return df_encoded

