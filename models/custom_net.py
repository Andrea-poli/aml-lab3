"""CustomNet: small CNN for TinyImageNet classification (from Lab02)."""
import torch
from torch import nn


class CustomNet(nn.Module):
    """3 x (Conv -> ReLU -> MaxPool) + Global Average Pooling + Linear.

    Input:  B x 3 x H x W  (64 x 64 for TinyImageNet; any size works thanks to GAP)
    Output: B x num_classes logits (no softmax: CrossEntropyLoss applies it)
    """

    def __init__(self, num_classes=200):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)   # halves H and W
        self.gap = nn.AdaptiveAvgPool2d(1)                  # global average pooling
        self.fc1 = nn.Linear(256, num_classes)

    def forward(self, x):
        # B x 3 x 64 x 64
        x = self.pool(self.conv1(x).relu())   # B x 64 x 32 x 32
        x = self.pool(self.conv2(x).relu())   # B x 128 x 16 x 16
        x = self.pool(self.conv3(x).relu())   # B x 256 x 8 x 8
        x = self.gap(x)                       # B x 256 x 1 x 1
        x = torch.flatten(x, 1)               # B x 256
        x = self.fc1(x)                       # B x num_classes (logits)
        return x


if __name__ == '__main__':
    # Quick shape check: python -m models.custom_net
    model = CustomNet()
    out = model(torch.randn(2, 3, 64, 64))
    print(out.shape)                                            # torch.Size([2, 200])
    print(sum(p.numel() for p in model.parameters()), 'params')  # 422216
