"""
cnn_definitions.py
──────────────────
ReLU-only CNN architectures for concolic robustness experiments.

Design principle
────────────────
All networks use ONLY ReLU activations — no BatchNorm, no Dropout,
no residual connections. This preserves the piecewise-linear structure
required for activation-region analysis in the concolic framework.

This is an explicit and principled design choice, not a limitation:
BatchNorm and Dropout break piecewise linearity because their behaviour
depends on the full batch or introduces stochastic switching that cannot
be encoded as deterministic activation constraints.

Networks
────────
LeNet5    : Classic CNN, MNIST/CIFAR-10, ~60K params
            Conv(1→6, 5×5) → ReLU → AvgPool →
            Conv(6→16, 5×5) → ReLU → AvgPool →
            Linear(256/400→120) → ReLU →
            Linear(120→84) → ReLU → Linear(84→10)

VGG11     : Lightweight VGG variant, CIFAR-10, ~9M params
            Based on VGG-11 but without BatchNorm
            8 conv layers + 3 FC layers, all ReLU

Usage
─────
    from utils.cnn_definitions import LeNet5, VGG11ReLU, get_cnn
    model = get_cnn('lenet5', input_channels=1, num_classes=10)
    model = get_cnn('vgg11', input_channels=3, num_classes=10)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


# ── helpers ───────────────────────────────────────────────────────────────────

def count_relu_neurons(model: nn.Module) -> int:
    """Count total ReLU neurons (output channels × spatial dims for conv)."""
    total = 0
    # walk forward once with a dummy input to count activation outputs
    hooks = []
    counts = []

    def hook_fn(module, input, output):
        counts.append(output.numel())

    for m in model.modules():
        if isinstance(m, nn.ReLU):
            hooks.append(m.register_forward_hook(hook_fn))

    dummy = torch.zeros(1, *model.input_shape)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return sum(counts)


def save_cnn(model: nn.Module, path: str, metadata: dict = None):
    """Save CNN weights + metadata to .pt file."""
    checkpoint = {
        'state_dict' : model.state_dict(),
        'class'      : model.__class__.__name__,
        'input_shape': model.input_shape,
        'num_classes': model.num_classes,
        'metadata'   : metadata or {},
    }
    torch.save(checkpoint, path)
    print(f'  Saved → {path}')


def load_cnn(path: str) -> nn.Module:
    """Load CNN from .pt checkpoint saved by save_cnn()."""
    ckpt = torch.load(path, map_location='cpu')
    cls  = ckpt['class']
    meta = ckpt['metadata']

    # reconstruct correct class
    if cls == 'LeNet5':
        in_ch = ckpt['input_shape'][0]
        model = LeNet5(input_channels=in_ch,
                       num_classes=ckpt['num_classes'])
    elif cls == 'VGG11ReLU':
        in_ch = ckpt['input_shape'][0]
        model = VGG11ReLU(input_channels=in_ch,
                          num_classes=ckpt['num_classes'])
    elif cls == 'SmallCNN':
        in_ch = ckpt['input_shape'][0]
        model = SmallCNN(input_channels=in_ch,
                         num_classes=ckpt['num_classes'])
    else:
        raise ValueError(f'Unknown class: {cls}')

    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'  Loaded ← {path}  |  metadata: {meta}')
    return model


# ── LeNet-5 ───────────────────────────────────────────────────────────────────

class LeNet5(nn.Module):
    """
    LeNet-5 with ReLU activations (no sigmoid, no tanh, no BN).

    Architecture:
        Input: (C, 32, 32) — CIFAR-10 or padded MNIST
        Conv1: 6 filters, 5×5  → ReLU → AvgPool 2×2
        Conv2: 16 filters, 5×5 → ReLU → AvgPool 2×2
        FC1  : → 120            → ReLU
        FC2  : → 84             → ReLU
        FC3  : → num_classes

    Parameters:
        input_channels : 1 for MNIST, 3 for CIFAR-10
        num_classes    : 10 for both datasets
        input_size     : spatial size (32 recommended; 28 for raw MNIST)
    """

    def __init__(
        self,
        input_channels : int = 1,
        num_classes    : int = 10,
        input_size     : int = 32,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes    = num_classes
        self.input_shape    = (input_channels, input_size, input_size)

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            flat_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 120),
            nn.ReLU(inplace=False),
            nn.Linear(120, 84),
            nn.ReLU(inplace=False),
            nn.Linear(84, num_classes),
        )

        self._flat_size = flat_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def n_relu_neurons(self) -> int:
        return count_relu_neurons(self)

    def __repr__(self):
        total = sum(p.numel() for p in self.parameters())
        return (f'LeNet5(in={self.input_channels}ch, '
                f'classes={self.num_classes}, '
                f'flat={self._flat_size}, '
                f'params={total:,})')


# ── VGG-11 (ReLU only, no BatchNorm) ─────────────────────────────────────────

class SmallCNN(nn.Module):
    """
    Lightweight custom CNN — tractable on CPU, meaningful for paper.

    Architecture (CIFAR-10, 32×32×3):
        Conv(3→32,  3×3, pad=1) → ReLU → MaxPool 2×2   → 16×16×32
        Conv(32→64, 3×3, pad=1) → ReLU → MaxPool 2×2   → 8×8×64
        Conv(64→128,3×3, pad=1) → ReLU → MaxPool 2×2   → 4×4×128
        Flatten → FC(2048→256) → ReLU → FC(256→10)

    Parameters  : ~540K (vs VGG11's 9.7M — 18× smaller)
    Expected acc: ~72–75% on CIFAR-10
    CPU time    : ~20–25 min for 30 epochs
    ReLU neurons: ~40K per sample (spatial × channel)

    Suitable as a "deep CNN" baseline for the paper — three conv layers
    with pooling gives a richer activation-region structure than LeNet5,
    while remaining trainable on a laptop.
    """

    def __init__(
        self,
        input_channels : int = 3,
        num_classes    : int = 10,
        input_size     : int = 32,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes    = num_classes
        self.input_shape    = (input_channels, input_size, input_size)

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32,  kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32,  64,  kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64,  128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
        )

        with torch.no_grad():
            dummy     = torch.zeros(1, input_channels, input_size, input_size)
            flat_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, num_classes),
        )
        self._flat_size = flat_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def n_relu_neurons(self) -> int:
        return count_relu_neurons(self)

    def __repr__(self):
        total = sum(p.numel() for p in self.parameters())
        return (f'SmallCNN(in={self.input_channels}ch, '
                f'classes={self.num_classes}, '
                f'flat={self._flat_size}, '
                f'params={total:,})')



class VGG11ReLU(nn.Module):
    """
    VGG-11 without BatchNorm — all ReLU activations, no BN.

    Adapted from the original VGG-11 architecture for CIFAR-10 (32×32).
    The FC layers are reduced from 4096→4096→1000 to 512→512→num_classes
    to match the smaller spatial resolution.

    Architecture (CIFAR-10):
        Conv(3→64, 3×3)   → ReLU → MaxPool 2×2
        Conv(64→128, 3×3) → ReLU → MaxPool 2×2
        Conv(128→256, 3×3)→ ReLU
        Conv(256→256, 3×3)→ ReLU → MaxPool 2×2
        Conv(256→512, 3×3)→ ReLU
        Conv(512→512, 3×3)→ ReLU → MaxPool 2×2
        Conv(512→512, 3×3)→ ReLU
        Conv(512→512, 3×3)→ ReLU → MaxPool 2×2
        Flatten → Linear(512→512) → ReLU
                  Linear(512→512) → ReLU
                  Linear(512→num_classes)

    Parameters:
        input_channels : 3 for CIFAR-10
        num_classes    : 10
    """

    def __init__(
        self,
        input_channels : int = 3,
        num_classes    : int = 10,
        input_size     : int = 32,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes    = num_classes
        self.input_shape    = (input_channels, input_size, input_size)

        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            flat_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, num_classes),
        )

        self._flat_size = flat_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def n_relu_neurons(self) -> int:
        return count_relu_neurons(self)

    def __repr__(self):
        total = sum(p.numel() for p in self.parameters())
        return (f'VGG11ReLU(in={self.input_channels}ch, '
                f'classes={self.num_classes}, '
                f'flat={self._flat_size}, '
                f'params={total:,})')


# ── factory ───────────────────────────────────────────────────────────────────

def get_cnn(
    name           : str,
    input_channels : int = 3,
    num_classes    : int = 10,
    input_size     : int = 32,
) -> nn.Module:
    """
    Factory function for CNN architectures.

    Args:
        name           : 'lenet5' or 'vgg11'
        input_channels : 1 (MNIST) or 3 (CIFAR-10)
        num_classes    : number of output classes
        input_size     : spatial input size (32 recommended)

    Returns:
        Untrained CNN instance.
    """
    name = name.lower()
    if name == 'lenet5':
        return LeNet5(input_channels, num_classes, input_size)
    elif name in ('vgg11', 'vgg11relu'):
        return VGG11ReLU(input_channels, num_classes, input_size)
    elif name in ('smallcnn', 'small_cnn'):
        return SmallCNN(input_channels, num_classes, input_size)
    else:
        raise ValueError(f"Unknown CNN: '{name}'. Choose: lenet5, vgg11, smallcnn")
