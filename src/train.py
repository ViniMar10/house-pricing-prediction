# src/train.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import pandas as pd
import numpy as np

def train_models(df):
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    rf = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    metrics = {
        'Regressão Linear': {
            'MAE': mean_absolute_error(y_test, lr_pred),
            'RMSE': root_mean_squared_error(y_test, lr_pred),
            'R2': r2_score(y_test, lr_pred)
        },
        'Random Forest': {
            'MAE': mean_absolute_error(y_test, rf_pred),
            'RMSE': root_mean_squared_error(y_test, rf_pred),
            'R2': r2_score(y_test, rf_pred)
        }
    }
    
    joblib.dump(rf, 'models/random_forest.pkl')
    joblib.dump(lr, 'models/linear_regression.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
    
    print("Modelos salvos com sucesso!")
    print(f"Random Forest → R² = {metrics['Random Forest']['R2']:.4f}")
    print(f"Regressão Linear → R² = {metrics['Regressão Linear']['R2']:.4f}")
    
    return rf, lr, metrics, X_test, y_test, rf_pred