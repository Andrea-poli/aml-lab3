"""Visualization helpers: dataset samples and training curves.

Every function returns a matplotlib Figure, so the caller can show it,
save it to disk (fig.savefig) or log it to wandb (wandb.Image(fig)).
"""
import matplotlib.pyplot as plt
import numpy as np

from dataset.tiny_imagenet import MEAN, STD


def denormalize(image):
    """Undo T.Normalize and convert a C x H x W tensor into an H x W x C array for imshow."""
    image = image.detach().cpu().numpy().transpose((1, 2, 0))
    image = image * np.array(STD) + np.array(MEAN)
    return np.clip(image, 0, 1)


def show_samples(loader, class_names, n_classes=10):
    """One image for each of the first n_classes distinct classes found in the loader (Lab01)."""
    rows = 2
    cols = (n_classes + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    seen = []

    for inputs, labels in loader:
        for img, label in zip(inputs, labels):
            label = label.item()
            if label not in seen:
                ax = axes[len(seen) // cols, len(seen) % cols]
                ax.imshow(denormalize(img))
                ax.set_title(class_names[label])
                ax.axis('off')
                seen.append(label)
                if len(seen) == n_classes:
                    break
        if len(seen) == n_classes:
            break

    fig.tight_layout()
    return fig


def plot_history(history):
    """Loss and accuracy curves (train vs val) over the epochs.

    history: dict with lists 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history['train_loss'], marker='o', label='train')
    ax_loss.plot(epochs, history['val_loss'], marker='o', label='val')
    ax_loss.set_title('Loss')
    ax_loss.set_xlabel('epoch')
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history['train_acc'], marker='o', label='train')
    ax_acc.plot(epochs, history['val_acc'], marker='o', label='val')
    ax_acc.set_title('Accuracy (%)')
    ax_acc.set_xlabel('epoch')
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    # Quick check with the Lab02 baseline results: python -m utils.visualization
    history = {
        'train_loss': [5.205, 4.940, 4.744, 4.571, 4.390, 4.252, 4.132, 4.016, 3.905, 3.798],
        'val_loss':   [5.058, 4.807, 4.662, 4.491, 4.335, 4.194, 4.098, 3.968, 3.859, 3.759],
        'train_acc':  [1.33, 3.41, 5.41, 7.26, 9.41, 11.29, 13.11, 14.78, 16.51, 18.23],
        'val_acc':    [2.20, 4.55, 6.14, 8.07, 10.29, 11.92, 13.47, 14.92, 17.03, 18.56],
    }
    plot_history(history).savefig('checkpoints/history_example.png', dpi=120)
    print('Saved checkpoints/history_example.png')
