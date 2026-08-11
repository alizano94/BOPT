"""Problem definitions for black-box optimization.

An :class:`Oracle` describes *the world*: what can be varied, what comes back, and
in what units. It deliberately knows nothing about Bayesian Optimization --- no
surrogates, no acquisition functions, no hypervolume reference points, no torch.

That one-directional dependency is the point. Anything methodological (how we
*search* the world) lives downstream in ``bopt.bo`` and ``bopt.adapters``; if a
choice would change your reported results, it does not belong in this module.

Conventions
-----------
* ``evaluate`` takes and returns **physical values in physical units**.
* Objectives keep their **true direction**: ``e_factor`` is a quantity you want
  small, and it is reported as a positive number that you want small. Sign flips
  for maximization-only optimizers happen in the adapter layer, in one place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

__all__ = ["Parameter", "Objective", "Oracle"]


@dataclass(frozen=True)
class Parameter:
    """One controllable input, with the range over which it may be set.

    Parameters
    ----------
    name, unit
        Used for error messages, plot labels and dataframe columns.
    low, high
        Inclusive bounds in physical units.
    tol
        Optional *absolute* out-of-bounds tolerance in physical units. When
        ``None`` (the usual case) the owning :class:`Oracle` derives one as a
        fraction of :attr:`span`.

        A span-relative default is not merely tidier than an absolute one: the
        dominant source of out-of-bounds slop is un-normalization from a unit
        cube, ``x = low + u * span``, whose floating-point error scales with
        ``span`` by construction. Matching the tolerance to that mechanism keeps
        one number meaningful across parameters of wildly different magnitude.
    """

    name: str
    low: float
    high: float
    unit: str
    tol: float | None = None

    def __post_init__(self) -> None:
        if not self.high > self.low:
            raise ValueError(
                f"parameter {self.name!r}: require high > low, got "
                f"low={self.low!r}, high={self.high!r}"
            )
        if self.tol is not None and self.tol < 0:
            raise ValueError(f"parameter {self.name!r}: tol must be >= 0, got {self.tol!r}")

    @property
    def span(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class Objective:
    """One measured response.

    ``maximize`` records the *physical* preference direction. Nothing in this
    module acts on it; it is metadata that the adapter layer reads when it has to
    present everything as a maximization problem.
    """

    name: str
    maximize: bool
    unit: str


class Oracle(ABC):
    """Base class for a multi-output black-box problem.

    Subclasses set the class attributes :attr:`parameters` and :attr:`objectives`
    and implement :meth:`_evaluate`. The public :meth:`evaluate` is concrete: it
    normalizes shapes and enforces bounds *before* delegating, so a subclass can
    assume its input is a clean ``(n, dim)`` array lying inside the domain.
    """

    #: Set by the subclass, in the column order that ``evaluate`` expects.
    parameters: tuple[Parameter, ...]
    #: Set by the subclass, in the column order that ``evaluate`` returns.
    objectives: tuple[Objective, ...]

    def __init__(self, *, bounds_tol: float = 1e-9) -> None:
        """
        Parameters
        ----------
        bounds_tol
            Default out-of-bounds tolerance, as a fraction of each parameter's
            span. Inputs outside the domain by less than this are silently
            clipped to the bound; anything further raises.

            The default of ``1e-9`` sits about seven orders of magnitude above
            float64 epsilon: loose enough to absorb any realistic slop from an
            acquisition optimizer, tight enough that a genuine bug cannot hide
            inside it. Individual parameters may override it via
            :attr:`Parameter.tol`.
        """
        for attr in ("parameters", "objectives"):
            value = getattr(self, attr, None)
            if not value:
                raise TypeError(
                    f"{type(self).__name__} must define a non-empty {attr!r} class attribute"
                )
        if bounds_tol < 0:
            raise ValueError(f"bounds_tol must be >= 0, got {bounds_tol!r}")
        self.bounds_tol = bounds_tol

    # ------------------------------------------------------------------ shape

    @cached_property
    def bounds(self) -> FloatArray:
        """``(2, dim)`` array of ``[low; high]`` in physical units."""
        return np.array(
            [[p.low for p in self.parameters], [p.high for p in self.parameters]],
            dtype=float,
        )

    @cached_property
    def tolerances(self) -> FloatArray:
        """``(dim,)`` array of absolute out-of-bounds tolerances, one per parameter."""
        return np.array(
            [p.tol if p.tol is not None else self.bounds_tol * p.span for p in self.parameters],
            dtype=float,
        )

    @property
    def dim(self) -> int:
        return len(self.parameters)

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def objective_names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.objectives)

    # ------------------------------------------------------------- evaluation

    def evaluate(self, X: npt.ArrayLike) -> FloatArray:
        """Evaluate the problem at one or more points.

        Parameters
        ----------
        X
            ``(n, dim)`` array of physical inputs, in :attr:`parameters` order.
            A 1-D array of length ``dim`` is accepted as a single point.

        Returns
        -------
        ``(n, n_objectives)`` array of physical outputs, in :attr:`objectives`
        order, with true units and true signs.
        """
        return self._evaluate(self._validate(X))

    __call__ = evaluate

    @abstractmethod
    def _evaluate(self, X: FloatArray) -> FloatArray:
        """Compute outputs for a validated in-bounds ``(n, dim)`` array.

        Implementations may assume the input is float64, two-dimensional, has
        exactly :attr:`dim` columns, and lies inside :attr:`bounds`.
        """

    # ------------------------------------------------------------- validation

    def _validate(self, X: npt.ArrayLike) -> FloatArray:
        """Coerce to ``(n, dim)`` float64 and enforce bounds.

        Points outside the domain by no more than :attr:`tolerances` are clipped
        onto the bound, so downstream physics only ever sees values inside the
        declared range. Larger violations raise, naming every offending
        parameter and by how much it overran.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            # A flat array of length `dim` is unambiguously a single point.
            X = X[None, :]
        if X.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got {X.ndim}-D with shape {X.shape}")
        if X.shape[1] != self.dim:
            raise ValueError(
                f"X has {X.shape[1]} columns, expected {self.dim} "
                f"({', '.join(self.parameter_names)})"
            )
        if not np.all(np.isfinite(X)):
            bad = np.argwhere(~np.isfinite(X))
            row, col = bad[0]
            raise ValueError(
                f"X contains non-finite values, first at row {row}, "
                f"parameter {self.parameters[col].name!r}"
            )

        low, high = self.bounds
        tol = self.tolerances
        under = low - X  # positive where X is below the lower bound
        over = X - high  # positive where X is above the upper bound
        violation = np.maximum(under, over)  # per-element overrun, <= 0 when inside

        if np.any(violation > tol):
            raise ValueError(self._bounds_error(X, violation, tol))

        return np.clip(X, low, high)

    def _bounds_error(self, X: FloatArray, violation: FloatArray, tol: FloatArray) -> str:
        """Build a message naming each parameter that overran, and by how much."""
        rows, cols = np.nonzero(violation > tol)
        lines = [f"{len(rows)} value(s) outside the domain of {type(self).__name__}:"]
        for row, col in list(zip(rows, cols))[:10]:
            p = self.parameters[col]
            lines.append(
                f"  row {row}, {p.name}={X[row, col]!r} {p.unit} "
                f"outside [{p.low}, {p.high}] by {violation[row, col]:.3e} "
                f"(tolerance {tol[col]:.3e})"
            )
        if len(rows) > 10:
            lines.append(f"  ... and {len(rows) - 10} more")
        return "\n".join(lines)

    # ------------------------------------------------------------------ repr

    def __repr__(self) -> str:
        params = ", ".join(f"{p.name}[{p.low},{p.high}]" for p in self.parameters)
        objs = ", ".join(f"{'max' if o.maximize else 'min'} {o.name}" for o in self.objectives)
        return f"{type(self).__name__}({params} -> {objs})"
