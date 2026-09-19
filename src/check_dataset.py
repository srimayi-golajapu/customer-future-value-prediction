"""
Check the structure of the downloaded dataset.
"""

import pandas as pd
from pathlib import Path
from utils import get_raw_data_dir

raw_dir = get_raw_data_dir()
data_file = raw_dir / "online_retail_II.xlsx"

df = pd.read_excel(data_file)

print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nDate range:", df['InvoiceDate'].min(), "to", df['InvoiceDate'].max())
