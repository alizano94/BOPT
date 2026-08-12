"""The optimization loop.

Deliberately strategy-agnostic: it knows only ``Strategy.propose``. Every arm ---
including the baselines --- goes through this identical code path, so the initial
design, the budget accounting and the metric are the same by construction rather
than by careful duplication.

Two oracles are used. The **noisy** one produces what a strategy sees; the
**clean** one produces the values used for reporting and is never passed to a
strategy. They cannot be collapsed into one: noise enters at the concentration
level and propagates non-linearly through two different objective formulas, so
the noisy observation is not the clean value plus a perturbation. Two calls cost
10 ms against a proposal step measured in seconds.
"""

from __future__ import annotations

import time

import torch
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from bopt.adapters import BoTorchProblem
from bopt.bo.history import IterationRecord, RunRecord
from bopt.bo.reference import hypervolume_fraction
from bopt.bo.strategies import Strategy, build_strategy
from bopt.oracles import SnarOracle

__all__ = ["initial_design", "run"]

DEFAULT_N_INIT = 16
DEFAULT_BUDGET = 64
DEFAULT_Q = 4
DEFAULT_NOISE = 1.0


def initial_design(bounds: Tensor, n: int, seed: int) -> Tensor:
    """Shared starting points for every arm at a given seed.

    Sharing these is what makes the comparison about *proposal rules*. If each arm
    started somewhere different, part of any measured gap would be initial-design
    luck rather than method quality.
    """
    return draw_sobol_samples(bounds=bounds, n=n, q=1, seed=seed).squeeze(1)


def run(
    strategy: str | Strategy,
    *,
    seed: int,
    budget: int = DEFAULT_BUDGET,
    n_init: int = DEFAULT_N_INIT,
    q: int = DEFAULT_Q,
    noise_level: float = DEFAULT_NOISE,
) -> RunRecord:
    """Run one strategy at one seed and return its record.

    Parameters
    ----------
    strategy
        A name from :data:`~bopt.bo.strategies.STRATEGIES`, or an instance.
    seed
        Controls the initial design, the noise stream and acquisition restarts.
        The same seed gives every arm the same starting points.
    budget
        Total oracle evaluations, initial design included, so arms with different
        batch sizes are compared on equal terms.
    """
    if n_init > budget:
        raise ValueError(f"n_init ({n_init}) exceeds budget ({budget})")

    noisy = BoTorchProblem(SnarOracle(noise_level=noise_level, seed=seed))
    clean = BoTorchProblem(SnarOracle())
    bounds = clean.bounds

    if isinstance(strategy, str):
        strategy = build_strategy(strategy, bounds, seed=seed)

    X = initial_design(bounds, n_init, seed)
    Y_noisy = noisy.evaluate(X)
    Y_true = clean.evaluate(X)

    record = RunRecord(
        strategy=strategy.name,
        seed=seed,
        budget=budget,
        q=q,
        noise_level=noise_level,
    )
    record.iterations.append(
        IterationRecord(
            iteration=0,
            n_evaluated=len(X),
            hv_fraction=hypervolume_fraction(Y_true),
            propose_seconds=0.0,
        )
    )

    iteration = 0
    while len(X) < budget:
        iteration += 1
        q_effective = min(q, budget - len(X))

        started = time.perf_counter()
        candidates = strategy.propose(X, Y_noisy, q_effective)
        elapsed = time.perf_counter() - started

        if candidates is None or len(candidates) == 0:
            # The strategy has nothing left to offer. Only Grid does this, and
            # the budget it leaves on the table is recorded rather than hidden.
            record.exhausted = True
            break

        candidates = torch.as_tensor(candidates, dtype=X.dtype, device=X.device)[:q_effective]
        X = torch.cat([X, candidates])
        Y_noisy = torch.cat([Y_noisy, noisy.evaluate(candidates)])
        Y_true = torch.cat([Y_true, clean.evaluate(candidates)])

        record.iterations.append(
            IterationRecord(
                iteration=iteration,
                n_evaluated=len(X),
                hv_fraction=hypervolume_fraction(Y_true),
                propose_seconds=elapsed,
                lengthscales=strategy.diagnostics().get("lengthscales"),
            )
        )

    record.X = X.cpu().numpy()
    record.Y_noisy = Y_noisy.cpu().numpy()
    record.Y_true = Y_true.cpu().numpy()
    return record
