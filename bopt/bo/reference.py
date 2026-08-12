"""The hypervolume reference point and the metric derived from it.

Both constants below are **stated, not computed at runtime**. Hypervolume is only
comparable across methods and seeds if the reference point is fixed in advance, so
re-running the sweep must not be able to silently move the metric.

Provenance: ``scripts/sweep_pareto.py --grid 25``, run 2026-08-11. Full reasoning
in ``docs/findings-degeneracy.md`` §6.
"""

from __future__ import annotations

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from torch import Tensor

__all__ = [
    "REFERENCE_POINT",
    "TRUE_FRONT_HYPERVOLUME",
    "reference_point",
    "hypervolume",
    "hypervolume_fraction",
]

#: Reference point in **all-maximize** space, i.e. ``(sty, -e_factor)``.
#:
#: Chosen as the true front's nadir pushed out by 10% of the front's span on each
#: axis. That rule balances the axes by construction --- ``span / (1.1 * span)`` is
#: 1/1.1 regardless of scale --- so both objectives occupy 90.9% of their
#: hypervolume axis. Under the problem's declared bounds ``(0, 500)`` instead, the
#: E-factor would occupy 0.3% and qNEHVI would be single-objective STY
#: optimization in all but name.
#:
#: This encodes knowledge of where the front is. That is standard in
#: multi-objective benchmarking, since comparison requires a fixed reference, but
#: it means hypervolume here is a *search and comparison signal*, not a physical
#: quantity, and is not comparable to values computed against a different
#: reference.
REFERENCE_POINT: tuple[float, float] = (1985.4606, -9.8860)

#: Hypervolume of the brute-forced front under :data:`REFERENCE_POINT`.
#:
#: Used to normalize, so a trace of 1.0 means "recovered the whole front".
#:
#: For scale: the single best-STY point achieves 1,248.26, or **9.67%** of this.
#: Recovering the front is worth roughly 10x more than finding peak throughput
#: alone, which is why the single-objective baseline is a genuine comparison here
#: rather than a formality.
TRUE_FRONT_HYPERVOLUME: float = 12911.7096


def reference_point(
    dtype: torch.dtype = torch.double, device: str | torch.device = "cpu"
) -> Tensor:
    """:data:`REFERENCE_POINT` as a tensor."""
    return torch.tensor(REFERENCE_POINT, dtype=dtype, device=device)


def hypervolume(Y: Tensor | np.ndarray) -> float:
    """Hypervolume dominated by ``Y`` relative to :data:`REFERENCE_POINT`.

    Parameters
    ----------
    Y
        ``(n, 2)`` outcomes in **all-maximize** space, exactly as
        :meth:`~bopt.adapters.botorch.BoTorchProblem.evaluate` returns them.
        Points dominated by others, or by the reference point itself, contribute
        nothing --- no filtering is required beforehand.
    """
    Y = torch.as_tensor(Y, dtype=torch.double)
    if Y.ndim != 2 or Y.shape[-1] != len(REFERENCE_POINT):
        raise ValueError(f"expected (n, {len(REFERENCE_POINT)}) outcomes, got {tuple(Y.shape)}")
    if len(Y) == 0:
        return 0.0
    partitioning = DominatedPartitioning(ref_point=reference_point(), Y=Y)
    return float(partitioning.compute_hypervolume().item())


def hypervolume_fraction(Y: Tensor | np.ndarray) -> float:
    """:func:`hypervolume` as a fraction of the true front's.

    Values slightly above 1.0 are expected and legitimate: the reference front
    came from a 25-point-per-dimension grid, so a continuous optimizer can land
    between grid nodes and dominate it. Treat 1.0 as "matched the grid front",
    not as a ceiling.
    """
    return hypervolume(Y) / TRUE_FRONT_HYPERVOLUME
