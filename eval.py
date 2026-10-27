"""Evaluation logic.

- evaluate(): the validation loop of Lab02, reused by train.py at every epoch.
- Run as a script to evaluate a saved checkpoint:
      python eval.py --checkpoint checkpoints/best_model.pth
"""
import argparse
import os

import torch
from torch import nn

from dataset.tiny_imagenet import download_tiny_imagenet, get_dataloaders
from models.custom_net import CustomNet


def evaluate(model, loader, criterion, device):
    """Return (average loss, accuracy %) of the model on the loader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Evaluate a CustomNet checkpoint on TinyImageNet val')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f'Checkpoint not found: {args.checkpoint} (run train.py first)')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root = download_tiny_imagenet(args.data_dir)
    _, val_loader = get_dataloaders(root, args.batch_size, args.num_workers)

    # Same architecture first, then load the learned weights (Lab02, slide 15)
    model = CustomNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    val_loss, val_acc = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')


if __name__ == '__main__':
    main()
