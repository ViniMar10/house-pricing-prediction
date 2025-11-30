import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Tuple

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12

def plot_feature_importance(model, feature_names: list, top_n: int = 15) -> plt.Figure:
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:][::-1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(indices))
    
    ax.barh(y_pos, importances[indices], color='#3498db', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
    ax.set_xlabel('Importância Relativa', fontsize=13)
    ax.set_title(f'Top {top_n} Variáveis Mais Importantes (Random Forest)', fontsize=16, pad=20)
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig

def plot_predicted_vs_actual(y_test: pd.Series, y_pred: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 7))
    
    ax.scatter(y_test, y_pred, alpha=0.6, color='#2ecc71', edgecolors='black', linewidth=0.5)
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Linha Perfeita')
    
    ax.set_xlabel('Preço Real (US$)')
    ax.set_ylabel('Preço Previsto (US$)')
    ax.set_title('Preço Real vs Previsto (Random Forest)', fontsize=16, pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=14)
    
    plt.tight_layout()
    return fig

def plot_price_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['SalePrice'], kde=True, color='#9b59b6', bins=50, ax=ax)
    ax.set_title('Distribuição dos Preços das Casas', fontsize=16, pad=20)
    ax.set_xlabel('Preço de Venda (US$)')
    ax.set_ylabel('Frequência')
    plt.tight_layout()
    return fig

def get_metrics_table(metrics_dict: dict) -> pd.DataFrame:
    data = {
        'Modelo': ['Regressão Linear', 'Random Forest'],
        'MAE (US$)': [f"{metrics_dict['Regressão Linear']['MAE']:,.0f}",
                      f"{metrics_dict['Random Forest']['MAE']:,.0f}"],
        'RMSE (US$)': [f"{metrics_dict['Regressão Linear']['RMSE']:,.0f}",
                       f"{metrics_dict['Random Forest']['RMSE']:,.0f}"],
        'R²': [f"{metrics_dict['Regressão Linear']['R2']:.4f}",
               f"{metrics_dict['Random Forest']['R2']:.4f}"]
    }
    df_metrics = pd.DataFrame(data)
    return df_metrics