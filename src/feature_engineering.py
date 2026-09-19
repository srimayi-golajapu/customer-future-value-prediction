"""
Feature engineering for customer-level modeling dataset.
Creates customer-level features from transaction data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils import get_processed_data_dir, ensure_dir_exists


def create_customer_features(df, cutoff_date=None):
    """
    Create customer-level features from transaction data.
    
    Args:
        df (pd.DataFrame): Clean transaction data
        cutoff_date (datetime): Only use transactions before this date for features
        
    Returns:
        pd.DataFrame: Customer-level features
    """
    print("\n=== Creating Customer Features ===")
    
    # Filter by cutoff date if provided
    if cutoff_date is not None:
        df_features = df[df['invoicedate'] < cutoff_date].copy()
        print(f"Using transactions before {cutoff_date}")
    else:
        df_features = df.copy()
        print("Using all transactions for features")
    
    # Only use normal transactions for feature calculation
    df_normal = df_features[df_features['transaction_type'] == 'normal'].copy()
    
    # Basic customer statistics
    customer_stats = df_normal.groupby('customerid').agg({
        'invoiceno': 'nunique',  # Total orders
        'revenue': ['sum', 'mean'],  # Total revenue, average revenue per line
        'quantity': ['sum', 'mean'],  # Total quantity, average quantity per line
        'invoicedate': ['min', 'max'],  # First and last purchase
        'stockcode': 'nunique'  # Unique products
    }).reset_index()
    
    customer_stats.columns = ['customerid', 'total_orders', 'total_revenue', 
                              'avg_line_revenue', 'total_quantity', 'avg_line_quantity',
                              'first_purchase', 'last_purchase', 'unique_products']
    
    # RFM Features
    # Recency: Days since last purchase
    reference_date = df_features['invoicedate'].max()
    customer_stats['recency'] = (reference_date - customer_stats['last_purchase']).dt.days
    
    # Frequency: Orders per month active
    customer_stats['customer_tenure_days'] = (customer_stats['last_purchase'] - customer_stats['first_purchase']).dt.days + 1
    customer_stats['customer_tenure_months'] = customer_stats['customer_tenure_days'] / 30.44
    customer_stats['purchase_frequency'] = customer_stats['total_orders'] / customer_stats['customer_tenure_months']
    
    # Monetary: Average order value
    customer_stats['avg_order_value'] = customer_stats['total_revenue'] / customer_stats['total_orders']
    customer_stats['avg_quantity_per_order'] = customer_stats['total_quantity'] / customer_stats['total_orders']
    
    # Time behavior features
    # Calculate days between purchases for each customer
    def calculate_days_between_purchases(customer_data):
        """Calculate statistics for days between purchases."""
        purchase_dates = customer_data.sort_values('invoicedate')['invoicedate'].dt.date.unique()
        if len(purchase_dates) <= 1:
            return pd.Series({'avg_days_between': 0, 'median_days_between': 0})
        
        days_between = pd.Series(purchase_dates).diff().dt.days.dropna()
        return pd.Series({
            'avg_days_between': days_between.mean(),
            'median_days_between': days_between.median()
        })
    
    days_between = df_normal.groupby('customerid').apply(calculate_days_between_purchases).reset_index()
    customer_stats = customer_stats.merge(days_between, on='customerid', how='left')
    customer_stats['avg_days_between'] = customer_stats['avg_days_between'].fillna(0)
    customer_stats['median_days_between'] = customer_stats['median_days_between'].fillna(0)
    
    # Active purchase months (using year-month string instead of period)
    df_normal['year_month'] = df_normal['invoicedate'].dt.to_period('M').astype(str)
    active_months = df_normal.groupby('customerid')['year_month'].nunique().reset_index()
    active_months.columns = ['customerid', 'active_months']
    customer_stats = customer_stats.merge(active_months, on='customerid', how='left')
    
    # Purchase frequency trend (recent vs historical spending)
    # Split customer history into two halves
    def calculate_spending_trend(customer_data):
        """Calculate spending trend coefficient."""
        if len(customer_data) < 2:
            return 0
        
        # Monthly spending
        monthly_spending = customer_data.set_index('invoicedate').resample('ME')['revenue'].sum()
        if len(monthly_spending) < 2:
            return 0
        
        # Simple trend: compare recent half to historical half
        mid_point = len(monthly_spending) // 2
        if mid_point == 0:
            return 0
        
        recent_avg = monthly_spending.iloc[-mid_point:].mean()
        historical_avg = monthly_spending.iloc[:mid_point].mean()
        
        if historical_avg == 0:
            return 0
        
        return (recent_avg - historical_avg) / historical_avg
    
    spending_trend = df_normal.groupby('customerid').apply(calculate_spending_trend).reset_index()
    spending_trend.columns = ['customerid', 'spending_trend']
    customer_stats = customer_stats.merge(spending_trend, on='customerid', how='left')
    customer_stats['spending_trend'] = customer_stats['spending_trend'].fillna(0)
    
    # Cancellation behavior features
    cancellations = df_features[df_features['transaction_type'] == 'cancellation'].groupby('customerid').size().reset_index()
    cancellations.columns = ['customerid', 'cancellation_count']
    customer_stats = customer_stats.merge(cancellations, on='customerid', how='left')
    customer_stats['cancellation_count'] = customer_stats['cancellation_count'].fillna(0)
    customer_stats['cancellation_rate'] = customer_stats['cancellation_count'] / customer_stats['total_orders']
    
    # Cancelled revenue
    cancelled_revenue = df_features[df_features['transaction_type'] == 'cancellation'].groupby('customerid')['revenue'].sum().abs().reset_index()
    cancelled_revenue.columns = ['customerid', 'cancelled_revenue']
    customer_stats = customer_stats.merge(cancelled_revenue, on='customerid', how='left')
    customer_stats['cancelled_revenue'] = customer_stats['cancelled_revenue'].fillna(0)
    
    # Geographic feature (country)
    # Use the most frequent country for each customer
    customer_country = df_features.groupby('customerid')['country'].agg(lambda x: x.mode()[0]).reset_index()
    customer_country.columns = ['customerid', 'country']
    customer_stats = customer_stats.merge(customer_country, on='customerid', how='left')
    
    # One-hot encode country (top 10 countries + 'Other')
    top_countries = customer_stats['country'].value_counts().head(10).index.tolist()
    customer_stats['country_encoded'] = customer_stats['country'].apply(lambda x: x if x in top_countries else 'Other')
    
    # Select final features
    feature_columns = [
        'customerid',
        # RFM
        'recency',
        'total_orders',
        'total_revenue',
        'avg_order_value',
        # Purchase behavior
        'total_quantity',
        'avg_quantity_per_order',
        'unique_products',
        'active_months',
        'purchase_frequency',
        # Time behavior
        'customer_tenure_days',
        'avg_days_between',
        'median_days_between',
        'spending_trend',
        # Cancellation behavior
        'cancellation_count',
        'cancellation_rate',
        'cancelled_revenue',
        # Geographic
        'country_encoded'
    ]
    
    customer_features = customer_stats[feature_columns].copy()
    
    print(f"Created {len(customer_features)} customer records")
    print(f"Features: {len(feature_columns) - 1} (excluding customerid)")
    
    return customer_features


def save_features(df, filename='customer_features.csv'):
    """Save customer features to CSV."""
    processed_dir = get_processed_data_dir()
    ensure_dir_exists(processed_dir)
    
    output_path = processed_dir / filename
    df.to_csv(output_path, index=False)
    print(f"Features saved to: {output_path}")
    
    return output_path


def main():
    """Main execution function."""
    from data_cleaning import load_online_retail_data, clean_data, save_cleaned_data
    
    # Load and clean data
    df = load_online_retail_data()
    df_clean = clean_data(df)
    save_cleaned_data(df_clean)
    
    # Create features
    customer_features = create_customer_features(df_clean)
    save_features(customer_features)
    
    return customer_features


if __name__ == "__main__":
    features = main()
