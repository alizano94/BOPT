"""Translate an :class:`~bopt.oracles.base.Oracle` into BoTorch's conventions.

BoTorch assumes three things an oracle has no reason to know about:

* everything is a ``torch`` tensor, in double precision;
* every objective is **maximized**;
* candidates arrive with arbitrary leading batch dimensions.

This module is the single place those assumptions are met. In particular it is
**the only place a sign is flipped**. Downstream --- surrogate, acquisition,
hypervolume, reference point --- lives entirely in "all-maximize" space, and
:meth:`BoTorchProblem.to_physical` converts back for reporting and plots.

Concentrating the flip here is a deliberate defence. The alternative, keeping
physical signs and passing weights into each acquisition function, spreads the
convention across every call site and makes each new acquisition a fresh
opportunity to forget it --- a mistake that produces plausible numbers rather than
an error.

What this module does *not* do
------------------------------
Input normalization and outcome standardization live on the **model**, via
BoTorch's :class:`~botorch.models.transforms.input.Normalize` and
:class:`~botorch.models.transforms.outcome.Standardize`. That is not a stylistic
preference: ``Standardize`` un-transforms the *posterior*, mean and variance
together, which a hand-rolled version gets subtly wrong in a way that never
raises and merely degrades every acquisition value.

The reference point is also absent by design. It is a methodological choice that
changes reported hypervolume, so it belongs with the optimization code, not with
the problem or its translation. See ``docs/findings-degeneracy.md`` §6.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from bopt.oracles.base import Oracle

__all__ = ["BoTorchProblem"]


class BoTorchProblem:
    """Wraps an :class:`Oracle` so BoTorch can call it.

    Parameters
    ----------
    oracle
        The problem to wrap. Its ``evaluate`` is called with physical values and
        returns physical values with true signs.
    dtype
        Defaults to ``torch.double``. BoTorch's GP fitting and acquisition
        optimization are numerically fragile in single precision, and the library
        itself warns when handed float32.
    device
        Defaults to CPU, and that is usually right here even when a GPU exists:
        the oracle is CPU-bound ``scipy``, and GP fits on tens to hundreds of
        points are small enough that kernel-launch overhead outweighs any gain.
    """

    def __init__(
        self,
        oracle: Oracle,
        *,
        dtype: torch.dtype = torch.double,
        device: str | torch.device = "cpu",
    ) -> None:
        self.oracle = oracle
        self.dtype = dtype
        self.device = torch.device(device)

        # +1 where the physical objective is already maximized, -1 where it is
        # minimized and must be negated. The only sign convention in the project.
        self._signs = torch.tensor(
            [1.0 if o.maximize else -1.0 for o in oracle.objectives],
            dtype=dtype,
            device=self.device,
        )

    # ------------------------------------------------------------------ metadata

    @property
    def dim(self) -> int:
        return self.oracle.dim

    @property
    def num_objectives(self) -> int:
        return self.oracle.n_objectives

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.oracle.parameter_names

    @property
    def objective_names(self) -> tuple[str, ...]:
        return self.oracle.objective_names

    @property
    def bounds(self) -> Tensor:
        """``(2, dim)`` tensor of physical bounds, as ``optimize_acqf`` expects."""
        return torch.as_tensor(self.oracle.bounds, dtype=self.dtype, device=self.device)

    # ---------------------------------------------------------------- evaluation

    def evaluate(self, X: Tensor) -> Tensor:
        """Evaluate at physical inputs, returning **all-maximize** outputs.

        Parameters
        ----------
        X
            ``(..., dim)``. Arbitrary leading batch dimensions are allowed, since
            BoTorch routinely passes ``b x q x d``; they are preserved in the
            output as ``(..., num_objectives)``.

        Notes
        -----
        The oracle is not differentiable, so ``X`` is detached before conversion.
        Gradients never flow through a real experiment either.
        """
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        if X.shape[-1] != self.dim:
            raise ValueError(
                f"X has {X.shape[-1]} columns, expected {self.dim} "
                f"({', '.join(self.parameter_names)})"
            )
        batch_shape = X.shape[:-1]

        flat = X.detach().reshape(-1, self.dim).cpu().numpy()
        Y = np.asarray(self.oracle.evaluate(flat), dtype=np.float64)

        out = torch.as_tensor(Y, dtype=self.dtype, device=self.device)
        return (out * self._signs).reshape(*batch_shape, self.num_objectives)

    __call__ = evaluate

    # ------------------------------------------------------------- sign handling

    def to_physical(self, Y: Tensor) -> Tensor:
        """All-maximize outputs back to physical values, for reporting and plots."""
        Y = torch.as_tensor(Y, dtype=self.dtype, device=self.device)
        return Y * self._signs

    def from_physical(self, Y: Tensor) -> Tensor:
        """Physical values into all-maximize space.

        Its own inverse --- the signs are all +/-1 --- but naming both directions
        keeps call sites readable about which space they are in.
        """
        return self.to_physical(Y)

    def __repr__(self) -> str:
        objs = ", ".join(
            f"{'+' if s > 0 else '-'}{n}" for s, n in zip(self._signs, self.objective_names)
        )
        return (
            f"{type(self).__name__}({type(self.oracle).__name__}, "
            f"dim={self.dim}, maximize=[{objs}], dtype={self.dtype})"
        )
