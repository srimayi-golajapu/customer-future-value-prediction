"""
E-commerce Customer Future Value Prediction Dashboard
A professional dark-themed Streamlit dashboard for customer value prediction.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


def main():
    # Page configuration
    st.set_page_config(
        page_title="Customer Future Value Prediction",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load custom CSS
    css_path = Path(__file__).parent / 'styles.css'
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Customer Value Prediction")
    st.sidebar.markdown("---")

    # Page navigation
    PAGES = {
        "Overview": "pages.1_Overview",
        "Customer Explorer": "pages.2_Customer_Explorer",
        "Prediction Explanation": "pages.3_Prediction_Explanation",
        "Customer Segmentation": "pages.4_Customer_Segmentation",
        "Model Performance": "pages.5_Model_Performance"
    }

    selection = st.sidebar.radio("Navigate to:", list(PAGES.keys()))

    # Load selected page
    try:
        page_module = __import__(PAGES[selection], fromlist=['main'])
        page_module.main()
    except Exception as e:
        st.error(f"Error loading page: {e}")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About**
    This dashboard predicts customer future revenue using machine learning models trained on historical transaction data.
    """)


if __name__ == "__main__":
    main()
