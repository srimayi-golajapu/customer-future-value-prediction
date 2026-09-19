"""
Download the UCI Online Retail II dataset.
"""

import urllib.request
from pathlib import Path
from utils import get_raw_data_dir, ensure_dir_exists


def download_file(url, destination):
    """
    Download a file from URL to destination.
    
    Args:
        url (str): URL to download from
        destination (Path): Destination path
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {destination}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
        print(f"\rProgress: {percent:.1f}%", end='')
    
    urllib.request.urlretrieve(url, destination, progress_hook)
    print(f"\nDownload complete: {destination}")


def main():
    """Download the UCI Online Retail II dataset."""
    raw_dir = get_raw_data_dir()
    ensure_dir_exists(raw_dir)
    
    # UCI ML Repository URL for Online Retail II
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
    destination = raw_dir / "online_retail_II.xlsx"
    
    try:
        download_file(url, destination)
        print("\nDataset downloaded successfully!")
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nAlternative: You can manually download from:")
        print("https://archive.ics.uci.edu/dataset/502/online+retail+ii")
        print(f"And save to: {destination}")


if __name__ == "__main__":
    main()
