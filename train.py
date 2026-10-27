"""Training logic: train CustomNet on TinyImageNet and track the run on wandb.

Examples:
    python train.py                         # default hyperparameters
    python train.py --lr 0.001 --epochs 5   # change hyperparameters from the command line
    python train.py --no-wandb              # run without wandb
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')                       # no display needed: figures are saved/logged
import torch
from torch import nn
from tqdm.auto import tqdm
import wandb

from dataset.tiny_imagenet import download_tiny_imagenet, get_dataloaders, load_class_names
from eval import evaluate
from models.custom_net import CustomNet
from utils.reproducibility import set_seed
from utils.visualization import plot_history, show_samples


def parse_args():
    parser = argparse.ArgumentParser(description='Train CustomNet on TinyImageNet')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)        # best value found in Lab02
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--project', default='aml-lab3', help='wandb project name')
    parser.add_argument('--run-name', default=None, help='wandb run name')
    parser.add_argument('--no-wandb', action='store_true', help='disable wandb logging')
    return parser.parse_args()


def train_one_epoch(epoch, model, loader, criterion, optimizer, device):
    """One pass over the training set (Lab02). Return (average loss, accuracy %)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in tqdm(loader, desc=f'Epoch {epoch}'):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(loader), 100. * correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Data
    root = download_tiny_imagenet(args.data_dir)
    train_loader, val_loader = get_dataloaders(root, args.batch_size, args.num_workers)
    class_names = load_class_names(root, train_loader.dataset.classes)

    # wandb: every hyperparameter in args is saved in the run config
    run = wandb.init(project=args.project, name=args.run_name, config=vars(args),
                     mode='disabled' if args.no_wandb else 'online')
    run.log({'samples': wandb.Image(show_samples(train_loader, class_names))})

    # Model, loss, optimizer
    model = CustomNet(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f'Epoch {epoch}: train loss {train_loss:.4f} acc {train_acc:.2f}% | '
              f'val loss {val_loss:.4f} acc {val_acc:.2f}%')
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)
        run.log({'epoch': epoch,
                   'train/loss': train_loss, 'train/acc': train_acc,
                   'val/loss': val_loss, 'val/acc': val_acc})

        # Keep only the best model (highest validation accuracy)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f'  New best model saved to {best_path}')

    print(f'Best validation accuracy: {best_acc:.2f}%')

    # Final curves: saved locally and logged to wandb
    fig = plot_history(history)
    fig.savefig(os.path.join(args.checkpoint_dir, 'history.png'), dpi=120)
    run.log({'history': wandb.Image(fig)})
    run.summary['best_val_acc'] = best_acc
    run.finish()


if __name__ == '__main__':
    main()
