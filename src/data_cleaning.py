"""
Data cleaning and preprocessing for the UCI Online Retail II dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils import get_raw_data_dir, get_processed_data_dir, ensure_dir_exists


def load_online_retail_data():
    """
    Load the UCI Online Retail II dataset.
    The dataset is available as a single Excel file with two sheets (Year 2009-2010 and Year 2010-2011).
    
    Returns:
        pd.DataFrame: Combined raw dataset
    """
    raw_dir = get_raw_data_dir()
    
    # File path for the single Excel file
    data_file = raw_dir / "online_retail_II.xlsx"
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected {data_file}.\n"
            f"Download from: https://archive.ics.uci.edu/dataset/502/online+retail+ii"
        )
    
    # Load both sheets from the Excel file
    df1 = pd.read_excel(data_file, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(data_file, sheet_name="Year 2010-2011")
    
    # Combine datasets
    df = pd.concat([df1, df2], ignore_index=True)
    
    print(f"Loaded {len(df1)} transactions from Year 2009-2010")
    print(f"Loaded {len(df2)} transactions from Year 2010-2011")
    print(f"Total transactions: {len(df):,}")
    print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    
    return df


def clean_data(df):
    """
    Clean the raw transaction data.
    
    Args:
        df (pd.DataFrame): Raw transaction data
        
    Returns:
        pd.DataFrame: Cleaned transaction data
    """
    print("\n=== Starting Data Cleaning ===")
    print(f"Initial records: {len(df):,}")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Standardize column names
    # Map the actual column names to our standard names
    column_mapping = {
        'Invoice': 'invoiceno',
        'StockCode': 'stockcode',
        'Description': 'description',
        'Quantity': 'quantity',
        'InvoiceDate': 'invoicedate',
        'Price': 'unitprice',
        'Customer ID': 'customerid',
        'Country': 'country'
    }
    df = df.rename(columns=column_mapping)
    
    # Convert InvoiceDate to datetime
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    
    # 1. Remove rows with missing CustomerID
    initial_customers = df['customerid'].nunique()
    df = df.dropna(subset=['customerid'])
    df['customerid'] = df['customerid'].astype(int)
    print(f"After removing missing CustomerID: {len(df):,} records, {df['customerid'].nunique()} customers")
    
    # 2. Remove rows with missing Description
    df = df.dropna(subset=['description'])
    print(f"After removing missing Description: {len(df):,} records")
    
    # 3. Remove duplicate records
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"After removing duplicates: {len(df):,} records (removed {duplicates:,} duplicates)")
    
    # 4. Filter out invalid prices (zero or negative)
    invalid_price = (df['unitprice'] <= 0).sum()
    df = df[df['unitprice'] > 0]
    print(f"After removing invalid prices: {len(df):,} records (removed {invalid_price:,})")
    
    # 5. Identify transaction types
    df['transaction_type'] = 'normal'
    df.loc[df['invoiceno'].astype(str).str.startswith('C'), 'transaction_type'] = 'cancellation'
    df.loc[df['quantity'] < 0, 'transaction_type'] = 'return'
    
    # 6. Calculate revenue per line item
    df['revenue'] = df['quantity'] * df['unitprice']
    
    # 7. Remove test/adjustment records (based on StockCode patterns)
    # Common test codes: POST, D, M, BANK CHARGES, CRUK, etc.
    test_codes = ['POST', 'D', 'M', 'BANK CHARGES', 'CRUK', 'C2', 'DOT']
    test_pattern = df['stockcode'].isin(test_codes) | df['stockcode'].str.contains('TEST', case=False, na=False)
    removed_tests = test_pattern.sum()
    df = df[~test_pattern]
    print(f"After removing test/adjustment records: {len(df):,} records (removed {removed_tests:,})")
    
    # 8. Handle extreme outliers using IQR method for quantity and unitprice
    def remove_outliers_iqr(data, column, multiplier=3):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Only apply outlier removal to normal transactions (not cancellations/returns)
    normal_transactions = df[df['transaction_type'] == 'normal'].copy()
    other_transactions = df[df['transaction_type'] != 'normal'].copy()
    
    initial_normal = len(normal_transactions)
    normal_transactions = remove_outliers_iqr(normal_transactions, 'quantity', multiplier=3)
    normal_transactions = remove_outliers_iqr(normal_transactions, 'unitprice', multiplier=3)
    outliers_removed = initial_normal - len(normal_transactions)
    
    df = pd.concat([normal_transactions, other_transactions], ignore_index=True)
    print(f"After removing extreme outliers: {len(df):,} records (removed {outliers_removed:,} outliers)")
    
    # 9. Final validation - ensure no negative quantities for normal transactions
    invalid_normal = df[(df['transaction_type'] == 'normal') & (df['quantity'] <= 0)].shape[0]
    if invalid_normal > 0:
        print(f"Warning: Found {invalid_normal} normal transactions with invalid quantities")
        df = df[~((df['transaction_type'] == 'normal') & (df['quantity'] <= 0))]
    
    print(f"\n=== Cleaning Complete ===")
    print(f"Final records: {len(df):,}")
    print(f"Final customers: {df['customerid'].nunique():,}")
    print(f"Date range: {df['invoicedate'].min()} to {df['invoicedate'].max()}")
    print(f"\nTransaction types:")
    print(df['transaction_type'].value_counts())
    
    return df


def save_cleaned_data(df, filename='clean_transactions.csv'):
    """
    Save the cleaned dataset to the processed data directory.
    
    Args:
        df (pd.DataFrame): Cleaned transaction data
        filename (str): Output filename
    """
    processed_dir = get_processed_data_dir()
    ensure_dir_exists(processed_dir)
    
    output_path = processed_dir / filename
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    
    return output_path


def main():
    """Main execution function."""
    # Load data
    df = load_online_retail_data()
    
    # Clean data
    df_clean = clean_data(df)
    
    # Save cleaned data
    save_cleaned_data(df_clean)
    
    return df_clean


if __name__ == "__main__":
    df = main()
