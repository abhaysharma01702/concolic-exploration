"""
network_definitions.py
─────────────────────
Defines the three ReLU MLP architectures used throughout the experiments.

    Small  : 2 hidden layers, 64 neurons each   (~8K params)
    Medium : 3 hidden layers, 256 neurons each  (~200K params)
    Large  : 4 hidden layers, 512 neurons each  (~800K params)

All networks use only ReLU activations (no BatchNorm, no Dropout) so that
activation-region analysis remains valid.  BatchNorm and Dropout would
break the piecewise-linear structure that the concolic framework relies on.

Usage
─────
    from utils.network_definitions import SmallMLP, MediumMLP, LargeMLP, get_network
    model = get_network("small", input_dim=784, output_dim=10)
"""

import torch
import torch.nn as nn
from typing import List


# ── base class ────────────────────────────────────────────────────────────────

class ReLUMLP(nn.Module):
    """
    Generic ReLU MLP with configurable depth and width.
    Deliberately avoids BatchNorm / Dropout to preserve piecewise linearity.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))

        self.network = nn.Sequential(*layers)

        # store linear layers separately for concolic analysis
        self.linear_layers = [m for m in self.network if isinstance(m, nn.Linear)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_activation_pattern(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns the binary activation pattern for input x.
        Each element is a bool tensor of shape (n_neurons,) for that layer.
        True  = neuron active (pre-activation > 0)
        False = neuron inactive (pre-activation <= 0)
        """
        pattern = []
        h = x.flatten()
        layer_idx = 0
        for module in self.network:
            if isinstance(module, nn.Linear):
                h = module(h)
                layer_idx += 1
            elif isinstance(module, nn.ReLU):
                pattern.append(h > 0)   # record BEFORE applying ReLU
                h = torch.relu(h)
        return pattern

    def get_pre_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns pre-activation values (z = Wx + b) for every ReLU layer.
        Used by the concolic engine to compute activation-boundary distances.
        """
        pre_acts = []
        h = x.flatten()
        for module in self.network:
            if isinstance(module, nn.Linear):
                h = module(h)
                pre_acts.append(h.detach())
            elif isinstance(module, nn.ReLU):
                h = torch.relu(h)
        # last linear has no ReLU — exclude it
        return pre_acts[:-1]

    def n_relu_neurons(self) -> int:
        """Total number of ReLU neurons across all hidden layers."""
        return sum(self.hidden_dims)

    def __repr__(self):
        total = sum(p.numel() for p in self.parameters())
        return (f"{self.__class__.__name__}("
                f"input={self.input_dim}, "
                f"hidden={self.hidden_dims}, "
                f"output={self.output_dim}, "
                f"params={total:,})")


# ── three concrete sizes ──────────────────────────────────────────────────────

class SmallMLP(ReLUMLP):
    """
    2 hidden layers × 64 neurons.
    Small enough for Marabou exact verification on single inputs.
    """
    def __init__(self, input_dim: int = 784, output_dim: int = 10):
        super().__init__(input_dim, [64, 64], output_dim)


class MediumMLP(ReLUMLP):
    """
    3 hidden layers × 256 neurons.
    MNIST-scale classifier; too large for Marabou, good for concolic.
    """
    def __init__(self, input_dim: int = 784, output_dim: int = 10):
        super().__init__(input_dim, [256, 256, 256], output_dim)


class LargeMLP(ReLUMLP):
    """
    4 hidden layers × 512 neurons.
    CIFAR-10 flattened scale; shows scalability limits.
    """
    def __init__(self, input_dim: int = 3072, output_dim: int = 10):
        super().__init__(input_dim, [512, 512, 512, 512], output_dim)


# ── factory ───────────────────────────────────────────────────────────────────

def get_network(size: str, input_dim: int, output_dim: int) -> ReLUMLP:
    """
    Factory function.

    Args:
        size       : "small" | "medium" | "large"
        input_dim  : flattened input dimension (784 for MNIST, 3072 for CIFAR-10)
        output_dim : number of classes

    Returns:
        Untrained ReLUMLP instance.
    """
    size = size.lower()
    if size == "small":
        return SmallMLP(input_dim, output_dim)
    elif size == "medium":
        return MediumMLP(input_dim, output_dim)
    elif size == "large":
        return LargeMLP(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown size '{size}'. Choose from: small, medium, large.")


# ── save / load helpers ───────────────────────────────────────────────────────

def save_model(model: ReLUMLP, path: str, metadata: dict = None):
    """
    Save model weights + architecture metadata to a .pt file.

    Args:
        model    : trained ReLUMLP instance
        path     : full file path, e.g. "models/small_mnist.pt"
        metadata : optional dict of extra info (accuracy, epochs, etc.)
    """
    checkpoint = {
        "state_dict"  : model.state_dict(),
        "input_dim"   : model.input_dim,
        "hidden_dims" : model.hidden_dims,
        "output_dim"  : model.output_dim,
        "class"       : model.__class__.__name__,
        "metadata"    : metadata or {},
    }
    torch.save(checkpoint, path)
    print(f"  Saved → {path}")


def load_model(path: str) -> ReLUMLP:
    """
    Load a model from a .pt checkpoint saved by save_model().

    Args:
        path : full file path

    Returns:
        ReLUMLP instance with loaded weights, in eval mode.
    """
    ckpt = torch.load(path, map_location="cpu")
    model = ReLUMLP(ckpt["input_dim"], ckpt["hidden_dims"], ckpt["output_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Loaded ← {path}  |  metadata: {ckpt['metadata']}")
    return model
