"""
E-commerce Customer Future Value Prediction Dashboard
A professional dark-themed Streamlit dashboard for customer value prediction.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import page functions
from pages import overview, customer_explorer, prediction_explanation, customer_segmentation, model_performance


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
    page = st.sidebar.radio("Navigate to:", [
        "Overview",
        "Customer Explorer",
        "Prediction Explanation",
        "Customer Segmentation",
        "Model Performance"
    ])

    # Load selected page
    if page == "Overview":
        overview.show()
    elif page == "Customer Explorer":
        customer_explorer.show()
    elif page == "Prediction Explanation":
        prediction_explanation.show()
    elif page == "Customer Segmentation":
        customer_segmentation.show()
    elif page == "Model Performance":
        model_performance.show()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About**
    This dashboard predicts customer future revenue using machine learning models trained on historical transaction data.
    """)


if __name__ == "__main__":
    main()
