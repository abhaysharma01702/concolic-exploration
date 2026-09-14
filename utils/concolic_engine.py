"""
concolic_engine.py  (v2 — gradient-guided line search)
────────────────────────────────────────────────────────
Core implementation of the concolic exploration algorithm for perturbation
radius estimation in ReLU-based neural networks.

v2 Fix (over v1)
────────────────
v1 problem: closed-form δ* = (z / ||w_eff||₁) × sign(w_eff) gave loose
upper bounds because ||w_eff||₁ grows exponentially with network depth
(composed weight matrices), making the estimated perturbation too small
to actually flip the prediction in practice.

v2 fix: two-stage approach per candidate neuron:
  Stage 1 — Direction (symbolic):  use w_eff to get the correct direction
             toward the activation boundary in input space.
  Stage 2 — Magnitude (concolic):  binary search along that direction,
             verifying f(x + α·d) concretely at each step.

This is the correct concolic design: symbolic reasoning provides the
search direction, concrete execution finds the actual flip magnitude.
Cascading neuron flips are handled automatically by concrete verification.

Lower bound fix
───────────────
v1 lower bound = |z| / ||w_eff||₁ — collapses to ~0 for deep networks.
v2 lower bound = gradient margin estimate + 0.5 × ε_upper heuristic —
stable across all network sizes.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class ConcolicResult:
    eps_upper       : float
    eps_lower       : float
    adversarial_x   : Optional[np.ndarray]
    n_iterations    : int
    runtime_sec     : float
    boundary_log    : List[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"ε_upper={self.eps_upper:.4f}  "
            f"ε_lower={self.eps_lower:.4f}  "
            f"gap={self.eps_upper - self.eps_lower:.4f}  "
            f"iters={self.n_iterations}  "
            f"time={self.runtime_sec:.2f}s"
        )


# ── activation region encoder ─────────────────────────────────────────────────

class ActivationRegion:
    """
    Computes per-neuron data including input-space effective weights
    (∂z_j/∂x via Jacobian composition) and normalised boundary distances.
    """

    def __init__(self, model: nn.Module, x: np.ndarray):
        self.x = x.copy()
        self.n = len(x)
        self.neurons = []
        self._extract(model, x)

    def _extract(self, model: nn.Module, x: np.ndarray):
        linear_layers  = [m for m in model.network if isinstance(m, nn.Linear)]
        hidden_linears = linear_layers[:-1]

        h    = torch.tensor(x, dtype=torch.float32)
        n_in = len(x)

        # Jacobian of current layer's input w.r.t. original input x
        J     = np.eye(n_in)       # (current_dim, n_in)
        b_acc = np.zeros(n_in)

        for lin in hidden_linears:
            W_l = lin.weight.detach().numpy()   # (n_out, n_in_l)
            b_l = lin.bias.detach().numpy()
            z   = W_l @ h.numpy() + b_l

            W_eff_l = W_l @ J           # (n_out, n_in_original) — ∂z/∂x
            b_eff_l = W_l @ b_acc + b_l

            for idx in range(len(z)):
                w_eff = W_eff_l[idx]
                w2    = float(np.linalg.norm(w_eff))
                w1    = float(np.sum(np.abs(w_eff)))
                self.neurons.append({
                    "w"          : w_eff,
                    "b_scalar"   : float(b_eff_l[idx]),
                    "active"     : bool(z[idx] > 0),
                    "z"          : float(z[idx]),
                    "margin"     : abs(float(z[idx])),
                    "margin_l2"  : abs(float(z[idx])) / (w2 + 1e-12),
                    "margin_l1"  : abs(float(z[idx])) / (w1 + 1e-12),
                    "w_norm2"    : w2,
                    "w_norm1"    : w1,
                })

            mask  = (z > 0).astype(float)
            J     = np.diag(mask) @ W_eff_l
            b_acc = mask * b_eff_l
            h     = torch.relu(torch.tensor(z, dtype=torch.float32))

    def n_neurons(self) -> int:
        return len(self.neurons)

    def sorted_by_margin(self, norm: str = "linf") -> List[Tuple[int, dict]]:
        """Sort by normalised boundary distance — nearest boundary first."""
        key = "margin_l2" if norm == "l2" else "margin_l1"
        return sorted(enumerate(self.neurons), key=lambda t: t[1][key])

    def gradient_lower_bound(
        self,
        model : nn.Module,
        x     : np.ndarray,
        label : int,
        norm  : str = "linf",
    ) -> float:
        """
        Gradient-based lower bound on ε*.

        Uses the linearised decision boundary argument:
            ε_lb = margin / ||∇_x margin||_dual

        where margin = score[label] - max_{c≠label} score[c].

        This is numerically stable for all network depths because it
        operates on the output layer directly, not on composed weight
        matrices of intermediate layers.

        Still a heuristic (assumes locally linear loss surface within
        the current activation region) — not a formal guarantee.
        """
        x_t    = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        logits = model(x_t.unsqueeze(0))[0]

        correct  = logits[label]
        competing = logits.clone()
        competing[label] = -1e9
        best_other = competing.max()
        margin = (correct - best_other).item()

        if margin <= 0:
            return 0.0

        (correct - best_other).backward()
        grad = x_t.grad.detach().numpy()

        # dual norm: L∞ perturbation budget → L1 gradient norm
        # L2  perturbation budget → L2 gradient norm
        grad_norm = (np.sum(np.abs(grad)) if norm == "linf"
                     else np.linalg.norm(grad)) + 1e-12

        return float(max(0.0, margin / grad_norm))


# ── gradient-guided boundary optimiser ───────────────────────────────────────

class BoundaryOptimiser:
    """
    Two-stage concolic boundary search:

    Stage 1 — Symbolic direction:
        d = ±sign(w_eff) for L∞  (direction that moves z_j toward 0)
        d = ±w_eff/||w||  for L2

    Stage 2 — Concrete line search:
        Binary search for smallest α such that f(x + α·d) ≠ f(x).
        Concrete verification at each step handles cascading flips.
    """

    def __init__(self, norm: str = "linf", n_line_search: int = 15):
        assert norm in ("l2", "linf")
        self.norm          = norm
        self.n_line_search = n_line_search

    def _direction(self, neuron: dict) -> np.ndarray:
        """
        Unit direction in input space toward neuron j's activation boundary.
        Sign chosen so that moving along d decreases |z_j| toward 0.
        """
        w      = neuron["w"]
        active = neuron["active"]

        if self.norm == "linf":
            # active:   z > 0, need to decrease z → move opposite to sign(w)
            # inactive: z ≤ 0, need to increase z → move along sign(w)
            d = -np.sign(w) if active else np.sign(w)
        else:
            w2 = neuron["w_norm2"]
            if w2 < 1e-12:
                return np.zeros_like(w)
            d = -w / w2 if active else w / w2

        return d

    @staticmethod
    def _predict(model: nn.Module, x: np.ndarray) -> int:
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            return int(model(t.unsqueeze(0)).argmax(dim=1).item())

    def solve(
        self,
        neuron     : dict,
        x          : np.ndarray,
        eps_upper  : float,
        model      : nn.Module,
        original_c : int,
    ) -> Optional[np.ndarray]:
        """
        Find minimum-norm δ along symbolic direction that concretely
        flips the model prediction.

        Returns δ if found within eps_upper, else None.
        """
        d = self._direction(neuron)

        # minimum α to flip this neuron analytically
        w_proj = np.dot(neuron["w"], d)
        if abs(w_proj) < 1e-12:
            return None

        alpha_min = abs(neuron["z"]) / abs(w_proj) + 1e-7  # just past boundary

        if alpha_min >= eps_upper:
            return None  # neuron flip alone requires too large a perturbation

        # check if prediction changes anywhere in [alpha_min, eps_upper)
        hi = eps_upper * 0.9999
        x_hi = x + hi * d
        if self._predict(model, x_hi) == original_c:
            return None  # no flip along this direction — skip

        # binary search for minimum α that flips prediction
        lo         = alpha_min
        best_alpha = hi
        best_delta = hi * d

        for _ in range(self.n_line_search):
            mid   = (lo + hi) / 2
            x_mid = x + mid * d
            if self._predict(model, x_mid) != original_c:
                best_alpha = mid
                best_delta = mid * d
                hi = mid
            else:
                lo = mid

        return best_delta


# ── main concolic explorer ────────────────────────────────────────────────────

class ConcolicExplorer:
    """
    Algorithm 1 (v2): Concolic Perturbation Radius Estimation.

    Changes from v1:
    - BoundaryOptimiser uses gradient-guided direction + line search
    - Neuron selection uses normalised L2 margin (stable for deep nets)
    - Lower bound uses gradient margin + 0.5×ε_upper heuristic
    """

    def __init__(
        self,
        model          : nn.Module,
        norm           : str   = "linf",
        max_iter       : int   = 200,
        max_radius     : float = 1.0,
        n_line_search  : int   = 15,
        verbose        : bool  = False,
    ):
        self.model         = model
        self.norm          = norm
        self.max_iter      = max_iter
        self.max_radius    = max_radius
        self.n_line_search = n_line_search
        self.verbose       = verbose
        self.optimiser     = BoundaryOptimiser(norm=norm,
                                               n_line_search=n_line_search)

    def _predict(self, x: np.ndarray) -> int:
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            return int(self.model(t.unsqueeze(0)).argmax(dim=1).item())

    def _norm(self, delta: np.ndarray) -> float:
        return (float(np.max(np.abs(delta))) if self.norm == "linf"
                else float(np.linalg.norm(delta)))

    def run(self, x: np.ndarray, true_label: int) -> ConcolicResult:
        t_start = time.time()

        c = self._predict(x)
        if c != true_label:
            return ConcolicResult(0.0, 0.0, None, 0, time.time() - t_start)

        region = ActivationRegion(self.model, x)

        # initial gradient-based lower bound
        eps_lower = region.gradient_lower_bound(self.model, x, c, self.norm)
        eps_upper = self.max_radius
        best_adv  = None
        log       = []

        sorted_neurons = region.sorted_by_margin(self.norm)

        for iteration in range(self.max_iter):
            neuron_idx, neuron = sorted_neurons[iteration % len(sorted_neurons)]

            delta = self.optimiser.solve(
                neuron, x, eps_upper, self.model, c
            )

            entry = {
                "iteration"          : iteration,
                "neuron_idx"         : neuron_idx,
                "margin_l2"          : neuron["margin_l2"],
                "found_delta"        : delta is not None,
                "flip_changed_class" : False,
            }

            if delta is not None:
                norm_delta = self._norm(delta)
                c_prime    = self._predict(x + delta)

                if c_prime != c and norm_delta < eps_upper:
                    eps_upper = norm_delta
                    best_adv  = (x + delta).copy()
                    entry["flip_changed_class"] = True
                    entry["new_eps_upper"]      = eps_upper
                    if self.verbose:
                        print(f"  iter {iteration:3d} | neuron {neuron_idx:4d} | "
                              f"ε_upper → {eps_upper:.5f}")

            log.append(entry)
            if eps_upper - eps_lower < 1e-5:
                break

        # tighten lower bound if adversarial example was found
        if best_adv is not None:
            eps_lower = max(eps_lower, eps_upper * 0.5)

        eps_lower = min(eps_lower, eps_upper)

        return ConcolicResult(
            eps_upper     = eps_upper,
            eps_lower     = eps_lower,
            adversarial_x = best_adv,
            n_iterations  = len(log),
            runtime_sec   = time.time() - t_start,
            boundary_log  = log,
        )


# ── batch runner ──────────────────────────────────────────────────────────────

def run_concolic_batch(
    model         : nn.Module,
    X             : np.ndarray,
    y             : np.ndarray,
    norm          : str   = "linf",
    max_iter      : int   = 200,
    max_radius    : float = 1.0,
    n_line_search : int   = 15,
    verbose       : bool  = False,
) -> List[ConcolicResult]:
    explorer = ConcolicExplorer(
        model, norm=norm, max_iter=max_iter,
        max_radius=max_radius, n_line_search=n_line_search,
        verbose=verbose,
    )
    results = []
    for i, (x, label) in enumerate(zip(X, y)):
        r = explorer.run(x, int(label))
        results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(X)}] {r.summary()}")
    return results
