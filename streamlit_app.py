"""
Streamlit entry point for deployment
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the dashboard
from dashboard.app import main as dashboard_main

if __name__ == "__main__":
    dashboard_main()
