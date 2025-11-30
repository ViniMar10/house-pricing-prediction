import os
from src.data_preprocessing import load_and_preprocess
from src.train import train_models

print("Iniciando o treinamento dos modelos...")
print("=" * 60)

os.makedirs("models", exist_ok=True)

print("Carregando e pré-processando o dataset...")
df = load_and_preprocess("train.csv") 
print(f"Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")

print("Treinando os modelos...")
rf, lr, metrics, X_test, y_test, y_pred = train_models(df)

print("=" * 60)
print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
print(f"Random Forest    → R² = {metrics['Random Forest']['R2']:.4f} | MAE = ${metrics['Random Forest']['MAE']:,.0f}")
print(f"Regressão Linear → R² = {metrics['Regressão Linear']['R2']:.4f} | MAE = ${metrics['Regressão Linear']['MAE']:,.0f}")
print("Arquivos salvos em /models/:")
print("   - random_forest.pkl")
print("   - linear_regression.pkl")
print("   - feature_names.pkl")
print("=" * 60)
