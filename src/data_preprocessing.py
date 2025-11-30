import pandas as pd

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    
    cols_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Utilities']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    df = pd.get_dummies(df, drop_first=True)
    
    return df