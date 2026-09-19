"""
Prediction Explanation Page - Customer Future Value Prediction Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from utils import get_processed_data_dir, get_outputs_dir


def show():
    st.title("🔮 Prediction Explanation")
    st.markdown("---")
    
    # Load data
    @st.cache_data
    def load_data():
        processed_dir = get_processed_data_dir()
        segments_df = pd.read_csv(processed_dir / 'customer_segments.csv')
        
        # Load SHAP explanations
        outputs_dir = get_outputs_dir()
        shap_explanation = joblib.load(outputs_dir / 'figures' / 'shap_explanation.pkl')
        
        return segments_df, shap_explanation
    
    segments_df, shap_explanation = load_data()
    
    # Customer selection
    st.subheader("Select Customer for Explanation")
    customer_list = segments_df['customerid'].sort_values().tolist()
    selected_customer = st.selectbox("Choose a Customer ID", customer_list)
    
    if selected_customer:
        # Get customer data
        customer_idx = segments_df[segments_df['customerid'] == selected_customer].index[0]
        customer_data = segments_df.iloc[customer_idx]
        
        # Display prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Customer ID", selected_customer)
            st.metric("Predicted Future Revenue", f"£{customer_data['predicted_future_revenue']:,.0f}")
        
        with col2:
            st.metric("Historical Revenue", f"£{customer_data['total_revenue']:,.0f}")
            st.metric("Segment", customer_data['segment'])
        
        st.markdown("---")
        
        # SHAP explanation
        st.subheader("Top Factors Influencing Prediction")
        
        # Get SHAP values for this customer
        customer_shap = shap_explanation[customer_idx]
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': customer_shap.feature_names,
            'SHAP Value': customer_shap.values
        })
        
        # Sort by absolute SHAP value
        feature_importance['Abs SHAP'] = feature_importance['SHAP Value'].abs()
        feature_importance = feature_importance.sort_values('Abs SHAP', ascending=False).head(10)
        
        # Display as table
        st.dataframe(
            feature_importance[['Feature', 'SHAP Value']].style.format({'SHAP Value': '{:.4f}'}),
            use_container_width=True
        )
        
        # Plot feature importance
        fig = px.bar(
            feature_importance.head(10),
            x='SHAP Value',
            y='Feature',
            orientation='h',
            title='Top 10 Factors Influencing Prediction',
            color='SHAP Value',
            color_continuous_scale='RdBu_r',
            range_color=[-1, 1]
        )
        fig.update_layout(
            plot_bgcolor='#1a1d24',
            paper_bgcolor='#1a1d24',
            font_color='#e0e0e0',
            xaxis_title='SHAP Value (Impact on Prediction)',
            yaxis_title='Feature'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Explanation text
        st.subheader("Explanation")
        
        positive_factors = feature_importance[feature_importance['SHAP Value'] > 0].head(5)
        negative_factors = feature_importance[feature_importance['SHAP Value'] < 0].head(5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Factors Increasing Prediction:**")
            for _, row in positive_factors.iterrows():
                st.write(f"• **{row['Feature']}**: +{row['SHAP Value']:.4f}")
        
        with col2:
            st.warning("**Factors Decreasing Prediction:**")
            for _, row in negative_factors.iterrows():
                st.write(f"• **{row['Feature']}**: {row['SHAP Value']:.4f}")
        
        st.markdown("---")
        
        # Historical vs Predicted
        st.subheader("Historical vs Predicted Behavior")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Historical Revenue", f"£{customer_data['total_revenue']:,.0f}")
        
        with col2:
            st.metric("Predicted Revenue", f"£{customer_data['predicted_future_revenue']:,.0f}")
        
        with col3:
            change = ((customer_data['predicted_future_revenue'] - customer_data['total_revenue']) / 
                      customer_data['total_revenue'] * 100)
            st.metric("Change", f"{change:+.1f}%")
