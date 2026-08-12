"""Gaussian process surrogate construction.

One place where the model is built, so every strategy that uses a GP uses the
same one and differences between arms are attributable to the acquisition
function rather than to modelling choices.
"""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

__all__ = ["fit_surrogate", "lengthscales"]


def fit_surrogate(X: Tensor, Y: Tensor, bounds: Tensor) -> SingleTaskGP:
    """Fit an independent GP per objective.

    Parameters
    ----------
    X
        ``(n, d)`` in **physical** units. The model normalizes internally.
    Y
        ``(n, m)`` in **all-maximize** space.
    bounds
        ``(2, d)`` physical domain bounds.

    Notes
    -----
    ``Normalize`` is given explicit domain bounds rather than being allowed to
    infer them from data. Data-inferred bounds shift as points accumulate, which
    silently changes the meaning of the fitted lengthscales between iterations and
    makes them useless as a diagnostic.

    ``Standardize`` matters more than it looks: it un-transforms the *posterior* ---
    mean and variance together. Standardizing by hand and forgetting to
    un-transform the variance produces no error, just uniformly wrong acquisition
    values.

    Both transforms live on the model rather than in the adapter so that
    ``optimize_acqf`` can work in physical units and candidates need no manual
    un-scaling, which is a common source of off-by-a-domain bugs.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"expected 2-D X and Y, got {tuple(X.shape)} and {tuple(Y.shape)}")
    if len(X) != len(Y):
        raise ValueError(f"X has {len(X)} rows but Y has {len(Y)}")

    model = SingleTaskGP(
        X,
        Y,
        input_transform=Normalize(d=X.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=Y.shape[-1]),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def lengthscales(model: SingleTaskGP) -> Tensor | None:
    """ARD lengthscales as ``(m, d)``, or ``None`` if the kernel has none.

    Recorded every iteration because it is a free sensitivity analysis: a long
    lengthscale means the model has decided an input barely matters. Given that
    ``conc_dfnb`` pins to its upper bound across the whole Pareto front, watching
    whether the GP discovers that is a direct check on whether it is learning the
    structure we already know is there.
    """
    kernel = getattr(model, "covar_module", None)
    raw = getattr(kernel, "lengthscale", None)
    if raw is None:
        base = getattr(kernel, "base_kernel", None)
        raw = getattr(base, "lengthscale", None)
    if raw is None:
        return None
    return raw.detach().reshape(-1, raw.shape[-1]).to(torch.double)
