import joblib
import pandas as pd
from typing import Dict, Any
import os

MODEL_PATH = "models/random_forest.pkl"
FEATURES_PATH = "models/feature_names.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado! Rode o treinamento primeiro: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

def predict_price(user_input: Dict[str, Any]) -> float:
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
    
    for key, value in user_input.items():
        if key in input_df.columns:
            input_df.at[0, key] = float(value)
    
    price = model.predict(input_df)[0]
    
    return round(float(price), 2)


def predict_formatted(user_input: Dict[str, Any]) -> str:
    price = predict_price(user_input)
    return f"US$ {price:,.0f}"