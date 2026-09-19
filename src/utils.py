"""
Utility functions for the customer future value prediction project.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def get_project_root():
    """Get the root directory of the project."""
    return Path(__file__).parent.parent


def get_data_dir():
    """Get the data directory path."""
    return get_project_root() / "data"


def get_raw_data_dir():
    """Get the raw data directory path."""
    return get_data_dir() / "raw"


def get_processed_data_dir():
    """Get the processed data directory path."""
    return get_data_dir() / "processed"


def get_models_dir():
    """Get the models directory path."""
    return get_project_root() / "models"


def get_outputs_dir():
    """Get the outputs directory path."""
    return get_project_root() / "outputs"


def ensure_dir_exists(path):
    """Ensure a directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
