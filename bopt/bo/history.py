"""What a run records, and how it is stored.

The central discipline here is keeping ``Y_noisy`` and ``Y_true`` separate and
never letting the second reach a strategy. ``Y_noisy`` is what the experimenter
measured; ``Y_true`` is the noiseless value at the same point and exists purely so
that reported hypervolume reflects *where the method actually went* rather than
which measurements happened to be lucky.

Without that separation, "best observed" is optimistically biased --- a method is
rewarded for sampling in high-variance regions and getting a favourable draw ---
and results can appear to beat their own ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bopt.bo.reference import hypervolume_fraction

__all__ = ["IterationRecord", "RunRecord"]


@dataclass
class IterationRecord:
    """One proposal round."""

    iteration: int
    n_evaluated: int  #: cumulative, so arms with different q remain comparable
    hv_fraction: float  #: from Y_true, as a fraction of the true front's hypervolume
    propose_seconds: float
    lengthscales: np.ndarray | None = None


@dataclass
class RunRecord:
    """One strategy, one seed, start to finish."""

    strategy: str
    seed: int
    budget: int
    q: int
    noise_level: float
    X: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Y_noisy: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Y_true: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    iterations: list[IterationRecord] = field(default_factory=list)
    exhausted: bool = False  #: strategy ran out of proposals before the budget did

    # ------------------------------------------------------------------ derived

    @property
    def n_evaluated(self) -> int:
        return len(self.X)

    @property
    def unused_budget(self) -> int:
        """Evaluations the strategy could not use. Non-zero only for Grid."""
        return self.budget - self.n_evaluated

    @property
    def final_hv_fraction(self) -> float:
        return self.iterations[-1].hv_fraction if self.iterations else 0.0

    def trace(self) -> tuple[np.ndarray, np.ndarray]:
        """``(n_evaluated, hv_fraction)`` --- the curve every plot is built from.

        Indexed by evaluation count rather than iteration so that arms with
        different batch sizes, and arms that terminate early, line up on a common
        x-axis.
        """
        return (
            np.array([it.n_evaluated for it in self.iterations]),
            np.array([it.hv_fraction for it in self.iterations]),
        )

    def hv_fraction_at(self, budget: int) -> float:
        """Hypervolume fraction using only the first ``budget`` evaluations.

        Lets a run be compared at a smaller budget after the fact, and lets an
        exhausted arm be scored fairly: Grid stops at 16 points, and its curve is
        then flat to the end of the axis because nothing further was learned.
        """
        return hypervolume_fraction(self.Y_true[:budget])

    # ------------------------------------------------------------ serialization

    def to_arrays(self) -> dict[str, np.ndarray]:
        n_eval, hv = self.trace()
        arrays = {
            "strategy": np.array(self.strategy),
            "seed": np.array(self.seed),
            "budget": np.array(self.budget),
            "q": np.array(self.q),
            "noise_level": np.array(self.noise_level),
            "exhausted": np.array(self.exhausted),
            "X": self.X,
            "Y_noisy": self.Y_noisy,
            "Y_true": self.Y_true,
            "trace_n": n_eval,
            "trace_hv": hv,
            "propose_seconds": np.array([it.propose_seconds for it in self.iterations]),
        }
        scales = [it.lengthscales for it in self.iterations if it.lengthscales is not None]
        if scales:
            arrays["lengthscales"] = np.stack(scales)
        return arrays

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.to_arrays())
