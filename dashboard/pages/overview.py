"""
Overview Page - Customer Future Value Prediction Dashboard
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


def show():
    st.title("📊 Overview")
    st.markdown("---")
    
    # Load data
    @st.cache_data
    def load_data():
        processed_dir = get_processed_data_dir()
        segments_df = pd.read_csv(processed_dir / 'customer_segments.csv')
        model_results = pd.read_csv(processed_dir / 'model_results.csv')
        return segments_df, model_results
    
    segments_df, model_results = load_data()
    
    # KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_customers = len(segments_df)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        total_historical = segments_df['total_revenue'].sum()
        st.metric("Total Historical Revenue", f"£{total_historical:,.0f}")
    
    with col3:
        avg_customer_revenue = segments_df['total_revenue'].mean()
        st.metric("Avg Customer Revenue", f"£{avg_customer_revenue:,.0f}")
    
    with col4:
        total_predicted = segments_df['predicted_future_revenue'].sum()
        st.metric("Total Predicted Revenue", f"£{total_predicted:,.0f}")
    
    with col5:
        high_value = (segments_df['segment'] == 'High Future Value').sum()
        st.metric("High-Value Customers", f"{high_value:,}")
    
    with col6:
        at_risk = (segments_df['segment'] == 'High Value At Risk').sum()
        st.metric("At-Risk High Value", f"{at_risk:,}")
    
    st.markdown("---")
    
    # Model Performance Summary
    st.subheader("Model Performance")
    best_model = model_results.loc[model_results['r2'].idxmax()]
    st.info(f"Best Model: **{best_model['model']}** with R² = {best_model['r2']:.4f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Value Distribution")
        fig = px.histogram(
            segments_df, 
            x='predicted_future_revenue',
            nbins=50,
            title='Distribution of Predicted Future Revenue',
            color_discrete_sequence=['#00d4ff']
        )
        fig.update_layout(
            plot_bgcolor='#1a1d24',
            paper_bgcolor='#1a1d24',
            font_color='#e0e0e0',
            xaxis_title='Predicted Revenue (£)',
            yaxis_title='Number of Customers'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Customer Segments")
        segment_counts = segments_df['segment'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='Customer Segment Distribution',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_layout(
            plot_bgcolor='#1a1d24',
            paper_bgcolor='#1a1d24',
            font_color='#e0e0e0'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Revenue by Segment
    st.subheader("Revenue by Segment")
    segment_revenue = segments_df.groupby('segment').agg({
        'predicted_future_revenue': 'sum',
        'total_revenue': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Predicted Revenue',
        x=segment_revenue['segment'],
        y=segment_revenue['predicted_future_revenue'],
        marker_color='#00d4ff'
    ))
    fig.add_trace(go.Bar(
        name='Historical Revenue',
        x=segment_revenue['segment'],
        y=segment_revenue['total_revenue'],
        marker_color='#ff6b6b'
    ))
    
    fig.update_layout(
        barmode='group',
        plot_bgcolor='#1a1d24',
        paper_bgcolor='#1a1d24',
        font_color='#e0e0e0',
        xaxis_title='Segment',
        yaxis_title='Revenue (£)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)
