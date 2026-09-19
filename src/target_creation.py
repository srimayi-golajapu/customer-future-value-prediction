"""
Create temporal target variable for future 6-month revenue prediction.
This ensures no data leakage by using time-based split.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils import get_processed_data_dir, ensure_dir_exists


def create_temporal_target(df, historical_months=18, prediction_months=6):
    """
    Create temporal target variable for future revenue prediction.
    
    Args:
        df (pd.DataFrame): Clean transaction data
        historical_months (int): Number of months for historical feature period
        prediction_months (int): Number of months for prediction period
        
    Returns:
        tuple: (customer_features, target_df) - features from historical period, target from prediction period
    """
    print("\n=== Creating Temporal Target ===")
    
    # Get date range
    min_date = df['invoicedate'].min()
    max_date = df['invoicedate'].max()
    total_days = (max_date - min_date).days
    
    print(f"Full date range: {min_date} to {max_date}")
    print(f"Total days: {total_days}")
    
    # Calculate cutoff dates
    # Historical period: from start to cutoff
    # Prediction period: from cutoff to end
    cutoff_date = max_date - pd.Timedelta(days=prediction_months * 30)
    
    print(f"\nHistorical period: {min_date} to {cutoff_date}")
    print(f"Prediction period: {cutoff_date} to {max_date}")
    
    # Split data
    df_historical = df[df['invoicedate'] < cutoff_date].copy()
    df_prediction = df[df['invoicedate'] >= cutoff_date].copy()
    
    print(f"\nHistorical transactions: {len(df_historical):,}")
    print(f"Prediction transactions: {len(df_prediction):,}")
    
    # Get customers who exist in historical period
    historical_customers = set(df_historical['customerid'].unique())
    print(f"Customers in historical period: {len(historical_customers):,}")
    
    # Calculate target: future 6-month revenue for each customer
    # Only use normal transactions for target
    df_prediction_normal = df_prediction[df_prediction['transaction_type'] == 'normal']
    
    target_revenue = df_prediction_normal.groupby('customerid')['revenue'].sum().reset_index()
    target_revenue.columns = ['customerid', 'future_6_month_revenue']
    
    # Add target orders count
    target_orders = df_prediction_normal.groupby('customerid')['invoiceno'].nunique().reset_index()
    target_orders.columns = ['customerid', 'future_6_month_orders']
    
    target_df = target_revenue.merge(target_orders, on='customerid', how='outer')
    target_df = target_df.fillna(0)
    
    print(f"\nTarget statistics:")
    print(target_df['future_6_month_revenue'].describe())
    
    # Only include customers who were in historical period
    target_df = target_df[target_df['customerid'].isin(historical_customers)]
    
    print(f"\nFinal target records: {len(target_df):,}")
    print(f"Customers with zero future revenue: {(target_df['future_6_month_revenue'] == 0).sum()}")
    print(f"Customers with future purchases: {(target_df['future_6_month_revenue'] > 0).sum()}")
    
    return df_historical, target_df, cutoff_date


def create_modeling_dataset(df_historical, target_df):
    """
    Create final modeling dataset by combining features and target.
    
    Args:
        df_historical (pd.DataFrame): Historical transaction data
        target_df (pd.DataFrame): Target variable dataframe
        
    Returns:
        pd.DataFrame: Final modeling dataset
    """
    from feature_engineering import create_customer_features
    
    # Create features from historical data
    customer_features = create_customer_features(df_historical)
    
    # Merge with target
    modeling_df = customer_features.merge(target_df, on='customerid', how='inner')
    
    print(f"\n=== Modeling Dataset Created ===")
    print(f"Total records: {len(modeling_df):,}")
    print(f"Features: {len(modeling_df.columns) - 2} (excluding customerid and target)")
    
    return modeling_df


def save_modeling_dataset(df, filename='modeling_dataset.csv'):
    """Save modeling dataset to CSV."""
    processed_dir = get_processed_data_dir()
    ensure_dir_exists(processed_dir)
    
    output_path = processed_dir / filename
    df.to_csv(output_path, index=False)
    print(f"Modeling dataset saved to: {output_path}")
    
    return output_path


def main():
    """Main execution function."""
    from data_cleaning import load_online_retail_data, clean_data, save_cleaned_data
    
    # Load and clean data
    df = load_online_retail_data()
    df_clean = clean_data(df)
    save_cleaned_data(df_clean)
    
    # Create temporal target
    df_historical, target_df, cutoff_date = create_temporal_target(
        df_clean, 
        historical_months=18, 
        prediction_months=6
    )
    
    # Create modeling dataset
    modeling_df = create_modeling_dataset(df_historical, target_df)
    save_modeling_dataset(modeling_df)
    
    # Save cutoff date for reference
    processed_dir = get_processed_data_dir()
    ensure_dir_exists(processed_dir)
    with open(processed_dir / 'cutoff_date.txt', 'w') as f:
        f.write(str(cutoff_date))
    
    return modeling_df


if __name__ == "__main__":
    dataset = main()
