"""
metrics.py
──────────
Shared evaluation functions used across all experiment notebooks.

Functions
─────────
    compute_accuracy          → top-1 accuracy on a dataset
    compute_robustness_stats  → aggregate stats from a list of ConcolicResult
    compare_methods           → build comparison DataFrame (ε_upper per method)
    tightness_score           → how close upper bound is to ground truth
    save_results              → save results dict to JSON
    load_results              → load results dict from JSON
    print_table               → pretty-print a comparison table to console
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import asdict


# ── accuracy ──────────────────────────────────────────────────────────────────

def compute_accuracy(
    model      : nn.Module,
    dataloader : torch.utils.data.DataLoader,
    device     : str = "cpu",
) -> float:
    """
    Compute top-1 classification accuracy.

    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.view(X.size(0), -1)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total


# ── robustness stats from concolic results ────────────────────────────────────

def compute_robustness_stats(results: list) -> Dict[str, float]:
    """
    Aggregate statistics from a list of ConcolicResult objects.

    Returns dict with keys:
        mean_eps_upper, std_eps_upper,
        mean_eps_lower, std_eps_lower,
        mean_gap, std_gap,
        mean_runtime, total_runtime,
        mean_iterations,
        fraction_adversarial_found   (eps_upper < max_radius)
    """
    uppers   = np.array([r.eps_upper    for r in results])
    lowers   = np.array([r.eps_lower    for r in results])
    gaps     = uppers - lowers
    runtimes = np.array([r.runtime_sec  for r in results])
    iters    = np.array([r.n_iterations for r in results])
    max_r    = max(r.eps_upper for r in results)

    return {
        "mean_eps_upper"            : float(np.mean(uppers)),
        "std_eps_upper"             : float(np.std(uppers)),
        "median_eps_upper"          : float(np.median(uppers)),
        "mean_eps_lower"            : float(np.mean(lowers)),
        "std_eps_lower"             : float(np.std(lowers)),
        "mean_gap"                  : float(np.mean(gaps)),
        "std_gap"                   : float(np.std(gaps)),
        "mean_runtime_sec"          : float(np.mean(runtimes)),
        "total_runtime_sec"         : float(np.sum(runtimes)),
        "mean_iterations"           : float(np.mean(iters)),
        "fraction_adversarial_found": float(np.mean(uppers < max_r)),
        "n_samples"                 : len(results),
    }


# ── tightness vs ground truth ─────────────────────────────────────────────────

def tightness_score(
    eps_estimated : np.ndarray,
    eps_true      : np.ndarray,
    clip_max      : float = 10.0,
) -> Dict[str, float]:
    """
    Measures how tight an estimated upper bound is relative to ground truth.

    tightness = (eps_estimated - eps_true) / eps_true
    Lower is better (0.0 = perfect).

    Args:
        eps_estimated : array of estimated ε values
        eps_true      : array of true ε* values (from Marabou)
        clip_max      : clip tightness values above this (for outlier robustness)

    Returns:
        Dict with mean_tightness, median_tightness, std_tightness,
        fraction_exact (estimated within 1% of true).
    """
    eps_true = np.maximum(eps_true, 1e-8)  # avoid division by zero
    tightness = (eps_estimated - eps_true) / eps_true
    tightness = np.clip(tightness, 0.0, clip_max)

    return {
        "mean_tightness"   : float(np.mean(tightness)),
        "median_tightness" : float(np.median(tightness)),
        "std_tightness"    : float(np.std(tightness)),
        "fraction_exact"   : float(np.mean(tightness < 0.01)),
        "fraction_within5" : float(np.mean(tightness < 0.05)),
        "fraction_within20": float(np.mean(tightness < 0.20)),
    }


# ── comparison table builder ──────────────────────────────────────────────────

def build_comparison_table(
    method_results: Dict[str, Dict],
    metrics       : List[str] = None,
) -> Dict[str, Dict]:
    """
    Build a structured comparison table across methods.

    Args:
        method_results : {method_name: stats_dict}  (from compute_robustness_stats)
        metrics        : list of metric keys to include (defaults to main ones)

    Returns:
        Nested dict {method_name: {metric: value}}
    """
    if metrics is None:
        metrics = [
            "mean_eps_upper",
            "std_eps_upper",
            "mean_eps_lower",
            "mean_gap",
            "mean_runtime_sec",
            "fraction_adversarial_found",
        ]
    table = {}
    for method, stats in method_results.items():
        table[method] = {m: stats.get(m, float("nan")) for m in metrics}
    return table


# ── save / load ───────────────────────────────────────────────────────────────

def save_results(data: Any, path: str):
    """
    Save a results dict (or list of dicts) to a JSON file.
    Handles numpy scalars and arrays automatically.
    """

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            return super().default(obj)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved results → {path}")


def load_results(path: str) -> Any:
    """Load results from a JSON file saved by save_results()."""
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded results ← {path}")
    return data


# ── pretty printer ────────────────────────────────────────────────────────────

def print_table(table: Dict[str, Dict], title: str = ""):
    """
    Print a comparison table to console in a readable format.

    Args:
        table : {method_name: {metric_name: value}}
        title : optional header string
    """
    if title:
        print(f"\n{'═'*60}")
        print(f"  {title}")
        print(f"{'═'*60}")

    methods = list(table.keys())
    if not methods:
        print("  (empty table)")
        return

    metrics = list(table[methods[0]].keys())
    col_w   = max(18, max(len(m) for m in methods) + 2)
    met_w   = max(22, max(len(k) for k in metrics) + 2)

    # header
    header = f"{'Metric':<{met_w}}" + "".join(f"{m:<{col_w}}" for m in methods)
    print(header)
    print("─" * len(header))

    for metric in metrics:
        row = f"{metric:<{met_w}}"
        for method in methods:
            val = table[method].get(metric, float("nan"))
            if isinstance(val, float):
                row += f"{val:<{col_w}.4f}"
            else:
                row += f"{str(val):<{col_w}}"
        print(row)
    print()


# ── sample selector ───────────────────────────────────────────────────────────

def get_correctly_classified_samples(
    model      : nn.Module,
    X          : np.ndarray,
    y          : np.ndarray,
    n_samples  : int,
    device     : str = "cpu",
    seed       : int = 42,
) -> tuple:
    """
    Select n_samples inputs that are correctly classified by model.
    Ensures experiments only measure robustness on valid predictions.

    Returns:
        (X_selected, y_selected) as np.ndarrays
    """
    model.eval()
    rng = np.random.default_rng(seed)

    indices = rng.permutation(len(X))
    selected_X, selected_y = [], []

    with torch.no_grad():
        for idx in indices:
            x_t = torch.tensor(X[idx], dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(x_t).argmax(dim=1).item()
            if pred == int(y[idx]):
                selected_X.append(X[idx])
                selected_y.append(y[idx])
            if len(selected_X) == n_samples:
                break

    if len(selected_X) < n_samples:
        print(f"  Warning: only {len(selected_X)} correct samples found "
              f"(requested {n_samples})")

    return np.array(selected_X), np.array(selected_y)
