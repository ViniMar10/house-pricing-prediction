# 🏡 Previsão de Preços de Casas — Ames, Iowa

Regressão Linear x Random Forest Regressor

Nesse trabalho acadêmicos, desenvolvemos uma aplicação em Python + Streamlit para treinar modelos e prever o preço de casas usando o dataset House Prices encontrado no Kaggle.

## Dataset Utilizado

House Prices - Kaggle
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

## Objetivo do Projeto
- pré-processamento dos dados
- Treinamento de Regressão Linear e Random Forest
- Comparação das métricas dos modelos
- Análise de importância de variáveis
- Análise dos valores reais x previstos
- Interface interativa em Streamlit para entrada de dados e predição

## Instalação
### 1. Criar ambiente virtual
```
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
### 2. Instalar dependências
```
pip install -r requirements.txt
```
## Treinamento dos modelos
Antes de rodar a interface é necessário realizar o treinamento dos modelos, para isso rode:
```
python3 train_models.py
```
Esse script vai:
- pré-processar o train.csv
- Treinar Regressão Linear e Random Forest
- Calcular métricas (MAE, RMSE, R²)
- Salvar os arquivos em `./models/`:
```
models/
├── feature_names.pkl
├── linear_regression.pkl
└── random_forest.pkl
```
## Executando a Aplicação Streamlit
Após treinar os modelos, basta executar:
```
streamlit run app.py
```
Dessa forma uma interface abrirá no navegador

# Dupla
Vinícius Martins e Henrique Martins
