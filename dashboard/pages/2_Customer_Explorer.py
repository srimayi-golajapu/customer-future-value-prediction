"""
Customer Explorer Page - Customer Future Value Prediction Dashboard
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
    st.title("🔍 Customer Explorer")
    st.markdown("---")
    
    # Load data
    @st.cache_data
    def load_data():
        processed_dir = get_processed_data_dir()
        segments_df = pd.read_csv(processed_dir / 'customer_segments.csv')
        return segments_df
    
    segments_df = load_data()
    
    # Customer selection
    st.subheader("Select Customer")
    customer_list = segments_df['customerid'].sort_values().tolist()
    selected_customer = st.selectbox("Choose a Customer ID", customer_list)
    
    if selected_customer:
        # Get customer data
        customer_data = segments_df[segments_df['customerid'] == selected_customer].iloc[0]
        
        # Customer profile card
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Customer ID", selected_customer)
        
        with col2:
            st.metric("Historical Revenue", f"£{customer_data['total_revenue']:,.0f}")
        
        with col3:
            st.metric("Predicted Revenue", f"£{customer_data['predicted_future_revenue']:,.0f}")
        
        with col4:
            segment = customer_data['segment']
            st.metric("Segment", segment)
        
        st.markdown("---")
        
        # Detailed metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Purchase Behavior")
            st.metric("Total Orders", f"{customer_data['total_orders']:.0f}")
            st.metric("Avg Order Value", f"£{customer_data['avg_order_value']:.0f}")
            st.metric("Recency (Days)", f"{customer_data['recency']:.0f}")
            st.metric("Frequency", f"{customer_data['purchase_frequency']:.2f}/month")
        
        with col2:
            st.subheader("Product Diversity")
            st.metric("Unique Products", f"{customer_data['unique_products']:.0f}")
            st.metric("Active Months", f"{customer_data['active_months']:.0f}")
            st.metric("Total Quantity", f"{customer_data['total_quantity']:.0f}")
            st.metric("Avg Quantity/Order", f"{customer_data['avg_quantity_per_order']:.1f}")
        
        with col3:
            st.subheader("Cancellation Behavior")
            st.metric("Cancellation Count", f"{customer_data['cancellation_count']:.0f}")
            st.metric("Cancellation Rate", f"{customer_data['cancellation_rate']*100:.1f}%")
            st.metric("Cancelled Revenue", f"£{customer_data['cancelled_revenue']:.0f}")
        
        st.markdown("---")
        
        # Time behavior
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Time Behavior")
            st.metric("Customer Tenure", f"{customer_data['customer_tenure_days']:.0f} days")
            st.metric("Avg Days Between", f"{customer_data['avg_days_between']:.0f}")
            st.metric("Median Days Between", f"{customer_data['median_days_between']:.0f}")
        
        with col2:
            st.subheader("Spending Trend")
            trend = customer_data['spending_trend']
            trend_color = "📈" if trend > 0 else "📉"
            st.metric("Spending Trend", f"{trend_color} {trend*100:.1f}%")
            st.metric("Country", customer_data['country_encoded'])
        
        st.markdown("---")
        
        # Risk assessment
        st.subheader("Risk Assessment")
        if segment == "High Value At Risk":
            st.error("⚠️ This customer is at high risk of churn. Immediate retention action recommended.")
        elif segment == "Low Activity":
            st.warning("⚠️ This customer shows low activity. Re-engagement campaign recommended.")
        elif segment == "High Future Value":
            st.success("✅ This is a high-value customer. Focus on retention and upselling.")
        elif segment == "Growing Customer":
            st.success("✅ This customer shows positive growth. Upselling opportunities available.")
        else:
            st.info("ℹ️ Standard customer. Apply regular engagement strategies.")
