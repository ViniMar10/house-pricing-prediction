import streamlit as st
import joblib
import pandas as pd

from src.data_preprocessing import load_and_preprocess
from src.train import train_models
from src.predict import predict_price
from src.utils import (
    plot_feature_importance,
    plot_predicted_vs_actual,
    plot_price_distribution,
    get_metrics_table
)

st.set_page_config(
    page_title="Previsão de Preço de Casas - Ames, Iowa",
    page_icon="house",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Previsão de Preço de Casas em Ames, Iowa")
st.markdown("**Modelos:** Regressão Linear x Random Forest Regressor")
st.markdown("Dataset: [House Prices - Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)")

@st.cache_resource
def load_models_and_results():
    df = load_and_preprocess("./train.csv")
    rf, lr, metrics, X_test, y_test, y_pred = train_models(df)
    feature_names = joblib.load("models/feature_names.pkl")
    return rf, lr, metrics, X_test, y_test, y_pred, feature_names, df

rf, lr, metrics, X_test, y_test, y_pred, feature_names, df_original = load_models_and_results()

st.sidebar.header("Características da Casa")

with st.sidebar.form(key="input_form"):
    st.subheader("Informações da Casa")

    col1, col2 = st.columns(2)

    with col1:
        OverallQual = st.slider("Qualidade geral (1-10)", 1, 10, 7, help="Nota geral de materiais e acabamento")
        OverallCond = st.slider("Condição geral (1-10)", 1, 10, 5, help="Condição atual da casa")
        GrLivArea = st.number_input("Área habitável (ft²)", 500, 6000, 1500)
        TotalBsmtSF = st.number_input("Área do porão (ft²)", 0, 6000, 1000)
        YearBuilt = st.slider("Ano de construção", 1870, 2010, 1970)

        BedroomAbvGr = st.slider("Número de quartos", 0, 8, 3)
        KitchenAbvGr = st.slider("Número de cozinhas", 0, 3, 1)
        TotRmsAbvGrd = st.slider("Total de cômodos (sem banheiros)", 3, 15, 7)

    with col2:
        FullBath = st.slider("Banheiros completos", 0, 4, 2)
        Fireplaces = st.slider("Lareiras", 0, 4, 1)
        GarageCars = st.slider("Vagas na garagem", 0, 4, 2)
        PoolArea = st.number_input("Área da piscina (ft²)", 0, 1000, 0)

        KitchenQual = st.selectbox("Qualidade da cozinha", 
                                   options=['Fa', 'TA', 'Gd', 'Ex'], 
                                   index=2, format_func=lambda x: {'Fa':'Razoável', 'TA':'Típica', 'Gd':'Boa', 'Ex':'Excelente'}[x])

        Neighborhood = st.selectbox("Bairro", 
            options=['NAmes','CollgCr','OldTown','Edwards','Somerst','NridgHt','Gilbert','Sawyer','Crawfor','NoRidge',
                     'Mitchel','Timber','StoneBr','ClearCr','BrkSide','NWAmes','SawyerW','IDOTRR','MeadowV','BrDale'],
            index=0)

        Heating = st.selectbox("Sistema de aquecimento", 
                               options=['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'],
                               index=0,
                               format_func=lambda x: {'GasA':'Gás A', 'GasW':'Gás W', 'Grav':'Gravidade', 
                                                      'Wall':'Parede', 'OthW':'Outro', 'Floor':'Piso'}[x])

    submitted = st.form_submit_button("Prever Preço da Casa", type="primary", use_container_width=True)

    if submitted:
        input_data = {
            'OverallQual': OverallQual,
            'OverallCond': OverallCond,
            'GrLivArea': GrLivArea,
            'TotalBsmtSF': TotalBsmtSF,
            'YearBuilt': YearBuilt,
            'BedroomAbvGr': BedroomAbvGr,
            'KitchenAbvGr': KitchenAbvGr,
            'TotRmsAbvGrd': TotRmsAbvGrd,
            'FullBath': FullBath,
            'Fireplaces': Fireplaces,
            'GarageCars': GarageCars,
            'PoolArea': PoolArea,

            f'KitchenQual_{KitchenQual}': 1,
            f'Neighborhood_{Neighborhood}': 1,
            f'Heating_{Heating}': 1,
        }

        try:
            price = predict_price(input_data)
            st.sidebar.success("**Preço Estimado**")
            st.sidebar.metric("Valor Previsto", f"US$ {price:,.0f}")
            st.balloons()
        except Exception as e:
            st.sidebar.error(f"Erro na predição: {e}")
            st.sidebar.info("Verifique se você já rodou o treinar_modelos.py")

tab1, tab2, tab3, tab4 = st.tabs(["Resultados", "Importância das Variáveis", "Real x Previsto", "Distribuição"])

with tab1:
    st.subheader("Comparação entre Modelos")
    st.dataframe(get_metrics_table(metrics), use_container_width=True)

with tab2:
    st.subheader("15 Variáveis Mais Importantes (Random Forest)")
    fig = plot_feature_importance(rf, feature_names, top_n=15)
    st.pyplot(fig)

with tab3:
    st.subheader("Preço Real vs Previsto (Teste)")
    fig = plot_predicted_vs_actual(y_test, y_pred)
    st.pyplot(fig)

with tab4:
    st.subheader("Distribuição dos Preços no Dataset")
    fig = plot_price_distribution(df_original)
    st.pyplot(fig)

st.markdown("---")
st.caption("Trabalho de Inteligência Artificial - 2025 | Dupla: Vinícius Martins & Henrique Martins")