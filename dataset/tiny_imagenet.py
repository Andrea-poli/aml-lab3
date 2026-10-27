"""TinyImageNet: download, preparation and DataLoaders.

Code from Lab01 (Dataset/DataLoader) and Lab02 (val split reorganization).
"""
import os
import shutil
import urllib.request
import zipfile

import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

# ImageNet RGB statistics (also used by utils/visualization.py to denormalize)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def download_tiny_imagenet(data_dir='data'):
    """Download and extract TinyImageNet into data_dir (only if not already there)."""
    root = os.path.join(data_dir, 'tiny-imagenet-200')
    if os.path.exists(root):
        print(f'Dataset already present in {root}')
        return root

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, 'tiny-imagenet-200.zip')
    print('Downloading TinyImageNet (~240 MB)...')
    urllib.request.urlretrieve(URL, zip_path)

    print('Extracting...')
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    os.remove(zip_path)

    prepare_val_split(root)
    return root


def prepare_val_split(root):
    """Reorganize val/ into one sub-folder per class, so ImageFolder can read it.

    val/images/*.JPEG + val_annotations.txt  ->  val/<wnid>/*.JPEG
    Safe to call twice: does nothing if val/images is already gone.
    """
    val_dir = os.path.join(root, 'val')
    images_dir = os.path.join(val_dir, 'images')
    if not os.path.exists(images_dir):
        return

    with open(os.path.join(val_dir, 'val_annotations.txt')) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
            shutil.move(os.path.join(images_dir, fn), os.path.join(val_dir, cls, fn))

    shutil.rmtree(images_dir)


def get_transform():
    """Images stay at the native 64x64 resolution (no Resize, see Lab02)."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def get_dataloaders(root, batch_size=32, num_workers=2):
    """Return (train_loader, val_loader) for TinyImageNet."""
    transform = get_transform()
    train_set = ImageFolder(os.path.join(root, 'train'), transform=transform)
    val_set = ImageFolder(os.path.join(root, 'val'), transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0)
    return train_loader, val_loader


def load_class_names(root, wnids):
    """Map WordNet IDs (e.g. 'n01443537') to readable names (e.g. 'goldfish')."""
    names = {}
    with open(os.path.join(root, 'words.txt')) as f:
        for line in f:
            wnid, name = line.rstrip('\n').split('\t')
            names[wnid] = name.split(',')[0]
    return [names[w] for w in wnids]
