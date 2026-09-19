"""
Upload Data Page - Customer Future Value Prediction Dashboard
Allows users to upload their own CSV data for prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from utils import get_processed_data_dir
from predict_uploaded import process_uploaded_data


def show():
    st.title("📤 Upload Your Data")
    st.markdown("---")
    
    # Check if data is already uploaded
    if 'uploaded_predictions' in st.session_state:
        st.success("✅ You have uploaded data successfully!")
        st.info("👉 Navigate to 'Customer Explorer' to view predictions for your data.")
        
        if st.button("Upload New Data"):
            del st.session_state['uploaded_predictions']
            del st.session_state['data_uploaded']
            st.rerun()
        return
    
    st.markdown("""
    Upload your own transaction data to get customer value predictions.
    
    **Required CSV Format:**
    Your CSV must contain the following columns:
    - `customerid`: Customer identifier (numeric)
    - `invoicedate`: Transaction date (YYYY-MM-DD format)
    - `quantity`: Quantity purchased (numeric)
    - `price`: Unit price (numeric)
    
    **Optional columns:**
    - `invoiceno`: Invoice number (can be string)
    - `stockcode`: Product code (can be string)
    - `description`: Product description (string)
    - `country`: Customer country (string)
    
    **Note:** Column names are case-insensitive. The system will automatically map your columns.
    """)
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file with transaction data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"File uploaded successfully! Shape: {df.shape}")
            
            # Display sample data
            st.subheader("Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Validate columns
            st.subheader("Column Validation")
            required_columns = ['customerid', 'invoicedate', 'quantity', 'price']
            optional_columns = ['invoiceno', 'stockcode', 'description', 'country']
            
            df_columns_lower = [col.lower() for col in df.columns]
            
            missing_required = []
            for col in required_columns:
                if col not in df_columns_lower:
                    missing_required.append(col)
            
            if missing_required:
                st.error(f"Missing required columns: {', '.join(missing_required)}")
                st.info("Please ensure your CSV contains all required columns.")
                return
            
            st.success("✅ All required columns found!")
            
            # Display column mapping
            st.subheader("Column Mapping")
            col_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in required_columns + optional_columns:
                    col_mapping[col] = col_lower
            
            mapping_df = pd.DataFrame({
                'Your Column': list(col_mapping.keys()),
                'Mapped To': list(col_mapping.values())
            })
            st.dataframe(mapping_df, use_container_width=True)
            
            # Basic statistics
            st.subheader("Data Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            
            with col2:
                unique_customers = df['customerid'].nunique() if 'customerid' in df.columns else 0
                st.metric("Unique Customers", f"{unique_customers:,}")
            
            with col3:
                unique_invoices = df['invoiceno'].nunique() if 'invoiceno' in df.columns else 0
                st.metric("Unique Invoices", f"{unique_invoices:,}")
            
            with col4:
                if 'invoicedate' in df.columns:
                    df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
                    date_range = f"{df['invoicedate'].min().date()} to {df['invoicedate'].max().date()}"
                    st.metric("Date Range", date_range)
            
            # Check for missing values
            st.subheader("Data Quality Check")
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                st.warning(f"Found {missing_data.sum()} missing values")
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values.astype(int),
                    'Missing %': (missing_data.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            else:
                st.success("✅ No missing values found!")
            
            # Process button
            st.markdown("---")
            st.subheader("Generate Predictions")
            
            if st.button("Process Data & Generate Predictions", type="primary"):
                with st.spinner("Processing your data... This may take a few minutes."):
                    # Process the data
                    success, result_df, error_msg = process_uploaded_data(df)
                    
                    if success:
                        st.session_state['uploaded_predictions'] = result_df
                        st.session_state['data_uploaded'] = True
                        
                        st.success("✅ Data processed successfully!")
                        st.info(f"Generated predictions for {len(result_df)} customers")
                        st.info("👉 Navigate to 'Customer Explorer' to view predictions for your data.")
                        
                        # Show summary
                        st.subheader("Prediction Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Customers", f"{len(result_df):,}")
                        
                        with col2:
                            total_predicted = result_df['predicted_future_revenue'].sum()
                            st.metric("Total Predicted Revenue", f"£{total_predicted:,.0f}")
                        
                        with col3:
                            high_value = (result_df['segment'] == 'High Future Value').sum()
                            st.metric("High-Value Customers", f"{high_value:,}")
                    else:
                        st.error(f"Error processing data: {error_msg}")
                        st.info("Please check your data format and try again.")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your file is a valid CSV with the correct format.")
    
    else:
        st.info("👆 Upload a CSV file to get started")
        
        # Show example format
        st.markdown("---")
        st.subheader("Example CSV Format")
        
        example_data = pd.DataFrame({
            'customerid': [12347, 12347, 12348],
            'invoiceno': ['536365', '536365', '536366'],
            'stockcode': ['85123A', '71053', '84406B'],
            'description': ['WHITE HANGING HEART T-LIGHT HOLDER', 'WHITE METAL LANTERN', 'CREAM CUPID HEARTS COAT HANGER'],
            'quantity': [6, 6, 8],
            'invoicedate': ['2010-12-01', '2010-12-01', '2010-12-01'],
            'price': [2.55, 3.39, 2.75],
            'country': ['United Kingdom', 'United Kingdom', 'United Kingdom']
        })
        
        st.dataframe(example_data, use_container_width=True)
