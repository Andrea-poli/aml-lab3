"""Download TinyImageNet into data/ and prepare the val split.

Usage (from the repository root):
    python utils/download_dataset.py
    python utils/download_dataset.py --data-dir /content/data
"""
import argparse
import os
import sys

# When run as "python utils/download_dataset.py", Python only sees the utils/ folder:
# add the repository root to the path so that "dataset" can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.tiny_imagenet import download_tiny_imagenet  # noqa: E402


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and prepare TinyImageNet')
    parser.add_argument('--data-dir', default='data')
    args = parser.parse_args()

    root = download_tiny_imagenet(args.data_dir)
    print(f'TinyImageNet ready in {root}')
