"""
Model Performance Page - Customer Future Value Prediction Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from utils import get_processed_data_dir


def main():
    st.title("📈 Model Performance")
    st.markdown("---")
    
    # Load data
    @st.cache_data
    def load_data():
        processed_dir = get_processed_data_dir()
        model_results = pd.read_csv(processed_dir / 'model_results.csv')
        segments_df = pd.read_csv(processed_dir / 'customer_segments.csv')
        return model_results, segments_df
    
    model_results, segments_df = load_data()
    
    # Model comparison table
    st.subheader("Model Comparison")
    
    # Format the results for display
    display_results = model_results.copy()
    display_results['MAE'] = display_results['mae'].apply(lambda x: f"{x:.4f}")
    display_results['RMSE'] = display_results['rmse'].apply(lambda x: f"{x:.4f}")
    display_results['R²'] = display_results['r2'].apply(lambda x: f"{x:.4f}")
    display_results = display_results.rename(columns={'model': 'Model'})
    
    st.dataframe(display_results[['Model', 'MAE', 'RMSE', 'R²']], use_container_width=True)
    
    # Highlight best model
    best_model = model_results.loc[model_results['r2'].idxmax()]
    st.success(f"🏆 Best Model: **{best_model['model']}** with R² = {best_model['r2']:.4f}")
    
    st.markdown("---")
    
    # Metric explanations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**MAE (Mean Absolute Error)**\n\nAverage absolute difference between predicted and actual values. Lower is better.")
    
    with col2:
        st.info("**RMSE (Root Mean Squared Error)**\n\nSquare root of average squared differences. Penalizes larger errors more. Lower is better.")
    
    with col3:
        st.info("**R² (R-Squared)**\n\nProportion of variance explained by the model. Higher is better (max 1.0).")
    
    st.markdown("---")
    
    # Model comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("MAE Comparison")
        fig = px.bar(
            model_results,
            x='model',
            y='mae',
            title='Mean Absolute Error by Model',
            color_discrete_sequence=['#00d4ff']
        )
        fig.update_layout(
            plot_bgcolor='#1a1d24',
            paper_bgcolor='#1a1d24',
            font_color='#e0e0e0',
            xaxis_title='Model',
            yaxis_title='MAE'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("R² Comparison")
        fig = px.bar(
            model_results,
            x='model',
            y='r2',
            title='R-Squared by Model',
            color_discrete_sequence=['#00d4ff']
        )
        fig.update_layout(
            plot_bgcolor='#1a1d24',
            paper_bgcolor='#1a1d24',
            font_color='#e0e0e0',
            xaxis_title='Model',
            yaxis_title='R²'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature importance (from SHAP)
    st.subheader("Feature Importance")
    
    # Load SHAP explanation
    from utils import get_outputs_dir
    import joblib
    
    outputs_dir = get_outputs_dir()
    shap_explanation = joblib.load(outputs_dir / 'figures' / 'shap_explanation.pkl')
    
    # Calculate mean absolute SHAP values for global importance
    mean_shap = np.abs(shap_explanation.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'Feature': shap_explanation.feature_names,
        'Importance': mean_shap
    }).sort_values('Importance', ascending=False).head(15)
    
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 15 Feature Importance (SHAP)',
        color_discrete_sequence=['#00d4ff']
    )
    fig.update_layout(
        plot_bgcolor='#1a1d24',
        paper_bgcolor='#1a1d24',
        font_color='#e0e0e0',
        xaxis_title='Mean Absolute SHAP Value',
        yaxis_title='Feature'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Model selection rationale
    st.subheader("Model Selection Rationale")
    st.markdown("""
    **XGBoost was selected as the final model** based on:
    
    - **Best R² score (0.7287)**: Explains ~73% of variance in future customer revenue
    - **Lowest MAE (0.4391)**: Most accurate predictions on average
    - **Lowest RMSE (0.6066)**: Better handling of larger errors
    - **Ability to capture non-linear relationships**: Customer behavior patterns are complex
    - **Feature importance**: Provides clear interpretability through SHAP values
    
    **Why not simpler models?**
    - Linear Regression (R² = 0.45): Too simple, cannot capture complex customer behavior patterns
    - Ridge Regression (R² = 0.45): Similar to Linear, regularization didn't improve performance
    - Random Forest (R² = 0.73): Performed well but XGBoost had slightly better metrics
    
    **Why not more complex models?**
    - Deep learning would be overkill for this dataset size (2,502 customers)
    - XGBoost provides excellent performance with good interpretability
    - Training time is reasonable and deployment is straightforward
    """)
