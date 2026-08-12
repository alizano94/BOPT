"""The strategies being compared.

Every arm implements one method::

    propose(X, Y, q) -> Tensor of shape (k, d), with k <= q

``X`` and ``Y`` are everything observed so far, where ``Y`` is the **noisy**
observation --- what a real experimenter would have. The clean values used for
reporting are never passed here, which is what stops a strategy from being able
to cheat, structurally rather than by inspection.

Returning fewer than ``q`` rows (including zero) means the strategy is exhausted;
the loop stops and records how much budget went unused. Only :class:`Grid` does
this, and that unused budget is the point of including it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.optim import optimize_acqf
from torch import Tensor
from torch.quasirandom import SobolEngine

from bopt.bo.reference import reference_point
from bopt.bo.surrogate import fit_surrogate, lengthscales

__all__ = ["Strategy", "Sobol", "Grid", "QNEHVI", "StyOnly", "build_strategy", "STRATEGIES"]


class Strategy(ABC):
    """Base class for a proposal rule."""

    name: str

    def __init__(self, bounds: Tensor, *, seed: int) -> None:
        self.bounds = bounds
        self.seed = seed
        self.dim = bounds.shape[-1]

    @abstractmethod
    def propose(self, X: Tensor, Y: Tensor, q: int) -> Tensor:
        """Return up to ``q`` candidate points, in physical units."""

    def diagnostics(self) -> dict[str, np.ndarray]:
        """Anything worth recording from the last proposal. Empty by default."""
        return {}

    def _scale(self, unit: Tensor) -> Tensor:
        low, high = self.bounds
        return low + unit.to(self.bounds) * (high - low)


# --------------------------------------------------------------------- baselines


class Sobol(Strategy):
    """Quasi-random space-filling search.

    The honest baseline. Sobol beats uniform random at covering a box, so
    outperforming it is a real result in a way that outperforming uniform random
    is not.

    Uses its own scrambled engine, offset from the seed that generated the shared
    initial design, so the proposals are not a continuation of those same points.
    """

    name = "sobol"

    def __init__(self, bounds: Tensor, *, seed: int) -> None:
        super().__init__(bounds, seed=seed)
        self._engine = SobolEngine(dimension=self.dim, scramble=True, seed=seed + 10_000)

    def propose(self, X: Tensor, Y: Tensor, q: int) -> Tensor:
        return self._scale(self._engine.draw(q))


class Grid(Strategy):
    """Two-level full factorial --- the classical design-of-experiments baseline.

    With a 64-evaluation budget in four dimensions, a uniform grid can afford
    ``64 ** (1/4) = 2.83`` points per axis. Two levels gives 2^4 = 16 points and
    uses a quarter of the budget; three gives 81 and overspends by 27%. Neither
    fits, and that is not an implementation annoyance --- it is the argument
    against grid search beyond two dimensions, so it is reported rather than
    engineered around.

    Two levels per axis places every point at a corner of the domain, which is
    exactly a 2^4 full factorial: the design a process engineer would classically
    run. That makes this a fair representative of standard practice rather than a
    strawman.
    """

    name = "grid"
    levels = 2

    def __init__(self, bounds: Tensor, *, seed: int) -> None:
        super().__init__(bounds, seed=seed)
        low, high = bounds
        axes = [torch.linspace(low[i], high[i], self.levels, dtype=bounds.dtype) for i in range(self.dim)]
        self._points = torch.cartesian_prod(*axes).to(bounds)
        self._cursor = 0

    def propose(self, X: Tensor, Y: Tensor, q: int) -> Tensor:
        batch = self._points[self._cursor : self._cursor + q]
        self._cursor += len(batch)
        return batch


# --------------------------------------------------------- model-based strategies


class _GPStrategy(Strategy):
    """Shared plumbing: fit the surrogate, optimize an acquisition function."""

    num_restarts = 10
    raw_samples = 256

    def __init__(self, bounds: Tensor, *, seed: int) -> None:
        super().__init__(bounds, seed=seed)
        self._lengthscales: Tensor | None = None

    @abstractmethod
    def _acquisition(self, model, X: Tensor):
        """Build the acquisition function for this arm."""

    @abstractmethod
    def _train_Y(self, Y: Tensor) -> Tensor:
        """Select which objective columns this arm models."""

    def propose(self, X: Tensor, Y: Tensor, q: int) -> Tensor:
        torch.manual_seed(self.seed + len(X))  # reproducible restarts
        model = fit_surrogate(X, self._train_Y(Y), self.bounds)
        self._lengthscales = lengthscales(model)
        candidates, _ = optimize_acqf(
            self._acquisition(model, X),
            bounds=self.bounds,
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            sequential=True,
        )
        return candidates.detach()

    def diagnostics(self) -> dict[str, np.ndarray]:
        if self._lengthscales is None:
            return {}
        return {"lengthscales": self._lengthscales.cpu().numpy()}


class QNEHVI(_GPStrategy):
    """Batch noisy expected hypervolume improvement --- the method under test.

    The *noisy* variant matters here: observations carry 1% measurement error, and
    qNEHVI integrates over the posterior at previously observed points rather than
    trusting them as exact. Using plain qEHVI would treat lucky measurements as
    ground truth.
    """

    name = "qnehvi"

    def _train_Y(self, Y: Tensor) -> Tensor:
        return Y

    def _acquisition(self, model, X: Tensor):
        return qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=reference_point(dtype=X.dtype, device=X.device).tolist(),
            X_baseline=X,
            prune_baseline=True,
        )


class StyOnly(_GPStrategy):
    """Single-objective BO on throughput, ignoring the E-factor entirely.

    A diagnostic as much as a baseline. Under the chosen reference point the best
    STY point alone captures only ~9.7% of the true front's hypervolume, so this
    arm should measurably trail qNEHVI. If it does not, the multi-objective
    machinery is not doing what it claims.
    """

    name = "sty_only"

    def _train_Y(self, Y: Tensor) -> Tensor:
        return Y[:, :1]

    def _acquisition(self, model, X: Tensor):
        return qLogNoisyExpectedImprovement(model=model, X_baseline=X, prune_baseline=True)


STRATEGIES: dict[str, type[Strategy]] = {
    "qnehvi": QNEHVI,
    "sobol": Sobol,
    "sty_only": StyOnly,
    "grid": Grid,
}


def build_strategy(name: str, bounds: Tensor, *, seed: int) -> Strategy:
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None
    return cls(bounds, seed=seed)
