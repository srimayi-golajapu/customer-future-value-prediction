"""
Prediction pipeline for uploaded user data.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from utils import get_models_dir
import sys


def validate_uploaded_data(df):
    """
    Validate uploaded data has required columns.
    
    Args:
        df (pd.DataFrame): Uploaded dataframe
        
    Returns:
        tuple: (is_valid, error_message, column_mapping)
    """
    required_columns = ['customerid', 'invoicedate', 'quantity', 'price']
    optional_columns = ['invoiceno', 'stockcode', 'description', 'country']
    
    df_columns_lower = {col.lower(): col for col in df.columns}
    
    # Check required columns
    missing_required = []
    column_mapping = {}
    
    for req_col in required_columns:
        if req_col in df_columns_lower:
            column_mapping[req_col] = df_columns_lower[req_col]
        else:
            missing_required.append(req_col)
    
    # Map optional columns if present
    for opt_col in optional_columns:
        if opt_col in df_columns_lower:
            column_mapping[opt_col] = df_columns_lower[opt_col]
    
    if missing_required:
        return False, f"Missing required columns: {', '.join(missing_required)}", column_mapping
    
    return True, None, column_mapping


def clean_uploaded_data(df, column_mapping):
    """
    Clean and standardize uploaded data.
    
    Args:
        df (pd.DataFrame): Raw uploaded dataframe
        column_mapping (dict): Column name mapping
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Create a copy with standardized column names
    df_clean = df.copy()
    
    # Rename columns to standard names
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    df_clean = df_clean.rename(columns=reverse_mapping)
    
    # Convert date column
    df_clean['invoicedate'] = pd.to_datetime(df_clean['invoicedate'], errors='coerce')
    
    # Remove rows with missing customerid
    df_clean = df_clean.dropna(subset=['customerid'])
    
    # Remove rows with missing or invalid dates
    df_clean = df_clean.dropna(subset=['invoicedate'])
    
    # Ensure numeric columns are numeric
    df_clean['quantity'] = pd.to_numeric(df_clean['quantity'], errors='coerce')
    df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce')
    
    # Remove invalid quantities and prices
    df_clean = df_clean[(df_clean['quantity'] > 0) & (df_clean['price'] > 0)]
    
    # Calculate revenue
    df_clean['revenue'] = df_clean['quantity'] * df_clean['price']
    
    # Identify transaction types
    df_clean['transaction_type'] = 'normal'
    
    # Check for cancellations (if invoiceno exists)
    if 'invoiceno' in df_clean.columns:
        df_clean['transaction_type'] = df_clean.apply(
            lambda row: 'cancellation' if str(row['invoiceno']).startswith('C') else 'normal',
            axis=1
        )
    
    return df_clean


def create_features_from_uploaded(df_clean):
    """
    Create customer-level features from uploaded data.
    
    Args:
        df_clean (pd.DataFrame): Cleaned transaction data
        
    Returns:
        pd.DataFrame: Customer-level features
    """
    # Only use normal transactions
    df_normal = df_clean[df_clean['transaction_type'] == 'normal'].copy()
    
    if len(df_normal) == 0:
        raise ValueError("No normal transactions found in data")
    
    # Basic customer statistics
    customer_stats = df_normal.groupby('customerid').agg({
        'invoiceno': 'nunique' if 'invoiceno' in df_normal.columns else lambda x: len(x),
        'revenue': ['sum', 'mean'],
        'quantity': ['sum', 'mean'],
        'invoicedate': ['min', 'max'],
        'stockcode': 'nunique' if 'stockcode' in df_normal.columns else lambda x: len(x)
    }).reset_index()
    
    customer_stats.columns = ['customerid', 'total_orders', 'total_revenue', 
                              'avg_line_revenue', 'total_quantity', 'avg_line_quantity',
                              'first_purchase', 'last_purchase', 'unique_products']
    
    # RFM Features
    reference_date = df_clean['invoicedate'].max()
    customer_stats['recency'] = (reference_date - customer_stats['last_purchase']).dt.days
    
    # Frequency
    customer_stats['customer_tenure_days'] = (customer_stats['last_purchase'] - customer_stats['first_purchase']).dt.days + 1
    customer_stats['customer_tenure_months'] = customer_stats['customer_tenure_days'] / 30.44
    customer_stats['purchase_frequency'] = customer_stats['total_orders'] / customer_stats['customer_tenure_months']
    
    # Monetary
    customer_stats['avg_order_value'] = customer_stats['total_revenue'] / customer_stats['total_orders']
    customer_stats['avg_quantity_per_order'] = customer_stats['total_quantity'] / customer_stats['total_orders']
    
    # Time behavior
    customer_stats['avg_days_between'] = customer_stats['customer_tenure_days'] / customer_stats['total_orders']
    customer_stats['median_days_between'] = customer_stats['avg_days_between']
    
    # Active purchase months
    df_normal['year_month'] = df_normal['invoicedate'].dt.to_period('M').astype(str)
    active_months = df_normal.groupby('customerid')['year_month'].nunique().reset_index()
    active_months.columns = ['customerid', 'active_months']
    customer_stats = customer_stats.merge(active_months, on='customerid', how='left')
    customer_stats['active_months'] = customer_stats['active_months'].fillna(1)
    
    # Spending trend (simplified)
    customer_stats['spending_trend'] = 0
    
    # Cancellation behavior
    cancellations = df_clean[df_clean['transaction_type'] == 'cancellation'].groupby('customerid').size().reset_index()
    cancellations.columns = ['customerid', 'cancellation_count']
    customer_stats = customer_stats.merge(cancellations, on='customerid', how='left')
    customer_stats['cancellation_count'] = customer_stats['cancellation_count'].fillna(0)
    customer_stats['cancellation_rate'] = customer_stats['cancellation_count'] / customer_stats['total_orders']
    
    # Cancelled revenue
    cancelled_revenue = df_clean[df_clean['transaction_type'] == 'cancellation'].groupby('customerid')['revenue'].sum().abs().reset_index()
    cancelled_revenue.columns = ['customerid', 'cancelled_revenue']
    customer_stats = customer_stats.merge(cancelled_revenue, on='customerid', how='left')
    customer_stats['cancelled_revenue'] = customer_stats['cancelled_revenue'].fillna(0)
    
    # Geographic
    if 'country' in df_clean.columns:
        customer_country = df_clean.groupby('customerid')['country'].agg(lambda x: x.mode()[0]).reset_index()
        customer_country.columns = ['customerid', 'country']
        customer_stats = customer_stats.merge(customer_country, on='customerid', how='left')
        
        # Encode country
        top_countries = customer_stats['country'].value_counts().head(10).index.tolist()
        customer_stats['country_encoded'] = customer_stats['country'].apply(lambda x: x if x in top_countries else 'Other')
    else:
        customer_stats['country_encoded'] = 'Unknown'
    
    # Select final features
    feature_columns = [
        'customerid',
        'recency',
        'total_orders',
        'total_revenue',
        'avg_order_value',
        'total_quantity',
        'avg_quantity_per_order',
        'unique_products',
        'active_months',
        'purchase_frequency',
        'customer_tenure_days',
        'avg_days_between',
        'median_days_between',
        'spending_trend',
        'cancellation_count',
        'cancellation_rate',
        'cancelled_revenue',
        'country_encoded'
    ]
    
    customer_features = customer_stats[feature_columns].copy()
    
    return customer_features


def predict_uploaded_data(customer_features):
    """
    Generate predictions for uploaded data using trained model.
    
    Args:
        customer_features (pd.DataFrame): Customer-level features
        
    Returns:
        pd.DataFrame: Features with predictions
    """
    # Load trained model
    models_dir = get_models_dir()
    model = joblib.load(models_dir / 'xgboost_model.pkl')
    
    # Prepare features
    id_col = 'customerid'
    feature_cols = [col for col in customer_features.columns if col != id_col]
    
    X = customer_features[feature_cols].copy()
    
    # Generate predictions
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    
    # Add predictions to dataframe
    customer_features['predicted_future_revenue'] = y_pred
    
    return customer_features


def create_segments_from_predictions(df):
    """
    Create customer segments based on predictions.
    
    Args:
        df (pd.DataFrame): Features with predictions
        
    Returns:
        pd.DataFrame: Features with segments
    """
    # Calculate quantiles
    revenue_80 = df['predicted_future_revenue'].quantile(0.8)
    revenue_40 = df['predicted_future_revenue'].quantile(0.4)
    
    # Initialize segment
    df['segment'] = 'Medium Future Value'
    
    # High Future Value
    df.loc[df['predicted_future_revenue'] >= revenue_80, 'segment'] = 'High Future Value'
    
    # Low Future Value
    df.loc[df['predicted_future_revenue'] < revenue_40, 'segment'] = 'Low Future Value'
    
    # High Value At Risk
    high_value_mask = df['predicted_future_revenue'] >= revenue_80
    high_risk_mask = (df['cancellation_rate'] > 0.1) | (df['recency'] > 60)
    df.loc[high_value_mask & high_risk_mask, 'segment'] = 'High Value At Risk'
    
    # Growing Customer
    growing_mask = (df['spending_trend'] > 0.1) & (df['predicted_future_revenue'] >= revenue_40)
    df.loc[growing_mask, 'segment'] = 'Growing Customer'
    
    # Low Activity
    low_activity_mask = (df['recency'] > 90) & (df['predicted_future_revenue'] >= revenue_40)
    df.loc[low_activity_mask, 'segment'] = 'Low Activity'
    
    return df


def process_uploaded_data(df_raw):
    """
    Complete pipeline to process uploaded data and generate predictions.
    
    Args:
        df_raw (pd.DataFrame): Raw uploaded dataframe
        
    Returns:
        tuple: (success, result_df, error_message)
    """
    try:
        # Validate data
        is_valid, error_msg, column_mapping = validate_uploaded_data(df_raw)
        if not is_valid:
            return False, None, error_msg
        
        # Clean data
        df_clean = clean_uploaded_data(df_raw, column_mapping)
        
        # Create features
        customer_features = create_features_from_uploaded(df_clean)
        
        # Generate predictions
        customer_features = predict_uploaded_data(customer_features)
        
        # Create segments
        customer_features = create_segments_from_predictions(customer_features)
        
        return True, customer_features, None
        
    except Exception as e:
        return False, None, str(e)
