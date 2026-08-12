"""Surrogates, acquisition functions and the optimization loop.

This is where methodology lives. Anything that would change a reported number ---
the hypervolume reference point, the normalization scheme, the acquisition
choice --- belongs here and not in ``bopt.oracles``, which describes the world and
knows nothing about how it is searched.
"""

from bopt.bo.history import IterationRecord, RunRecord
from bopt.bo.loop import initial_design, run
from bopt.bo.reference import (
    REFERENCE_POINT,
    TRUE_FRONT_HYPERVOLUME,
    hypervolume,
    hypervolume_fraction,
    reference_point,
)
from bopt.bo.strategies import STRATEGIES, Strategy, build_strategy
from bopt.bo.surrogate import fit_surrogate, lengthscales

__all__ = [
    "run",
    "initial_design",
    "RunRecord",
    "IterationRecord",
    "Strategy",
    "STRATEGIES",
    "build_strategy",
    "fit_surrogate",
    "lengthscales",
    "REFERENCE_POINT",
    "TRUE_FRONT_HYPERVOLUME",
    "reference_point",
    "hypervolume",
    "hypervolume_fraction",
]
