"""
Customer Segmentation Page - Customer Future Value Prediction Dashboard
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
    st.title("👥 Customer Segmentation")
    st.markdown("---")
    
    # Load data
    @st.cache_data
    def load_data():
        processed_dir = get_processed_data_dir()
        segments_df = pd.read_csv(processed_dir / 'customer_segments.csv')
        segment_stats = pd.read_csv(processed_dir / 'segment_statistics.csv', index_col=0)
        return segments_df, segment_stats
    
    segments_df, segment_stats = load_data()
    
    # Segment distribution
    st.subheader("Segment Distribution")
    segment_counts = segments_df['segment'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    with col2:
        fig = px.bar(
            x=segment_counts.index,
            y=segment_counts.values,
            title='Customer Count by Segment',
            color_discrete_sequence=['#00d4ff']
        )
        fig.update_layout(
            plot_bgcolor='#1a1d24',
            paper_bgcolor='#1a1d24',
            font_color='#e0e0e0',
            xaxis_title='Segment',
            yaxis_title='Number of Customers'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Segment revenue
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
    
    st.markdown("---")
    
    # Segment characteristics table
    st.subheader("Segment Characteristics")
    st.dataframe(
        segment_stats.style.format({
            'Avg Predicted Revenue': '£{:,.0f}',
            'Median Predicted Revenue': '£{:,.0f}',
            'Total Predicted Revenue': '£{:,.0f}',
            'Avg Historical Revenue': '£{:,.0f}',
            'Avg Recency': '{:.0f}',
            'Avg Cancellation Rate': '{:.2%}',
            'Count': '{:.0f}'
        }),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # High-value customers
    st.subheader("High-Value Customers")
    high_value = segments_df[segments_df['segment'] == 'High Future Value'].sort_values(
        'predicted_future_revenue', ascending=False
    ).head(20)
    
    st.dataframe(
        high_value[['customerid', 'predicted_future_revenue', 'total_revenue', 
                   'total_orders', 'recency']].style.format({
            'predicted_future_revenue': '£{:,.0f}',
            'total_revenue': '£{:,.0f}',
            'recency': '{:.0f}'
        }),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # At-risk high-value customers
    st.subheader("⚠️ At-Risk High-Value Customers")
    at_risk = segments_df[segments_df['segment'] == 'High Value At Risk'].sort_values(
        'predicted_future_revenue', ascending=False
    )
    
    if len(at_risk) > 0:
        st.dataframe(
            at_risk[['customerid', 'predicted_future_revenue', 'total_revenue',
                    'cancellation_rate', 'recency']].style.format({
                'predicted_future_revenue': '£{:,.0f}',
                'total_revenue': '£{:,.0f}',
                'cancellation_rate': '{:.2%}',
                'recency': '{:.0f}'
            }),
            use_container_width=True
        )
        st.warning(f"⚠️ {len(at_risk)} high-value customers are at risk of churn. Immediate retention action recommended.")
    else:
        st.info("No high-value customers currently at risk.")
    
    st.markdown("---")
    
    # Segment filter
    st.subheader("Filter by Segment")
    selected_segment = st.selectbox("Select Segment", segments_df['segment'].unique())
    
    filtered_customers = segments_df[segments_df['segment'] == selected_segment].sort_values(
        'predicted_future_revenue', ascending=False
    )
    
    st.dataframe(
        filtered_customers[['customerid', 'predicted_future_revenue', 'total_revenue',
                          'total_orders', 'recency', 'cancellation_rate']].style.format({
            'predicted_future_revenue': '£{:,.0f}',
            'total_revenue': '£{:,.0f}',
            'recency': '{:.0f}',
            'cancellation_rate': '{:.2%}'
        }),
        use_container_width=True
    )
