"""Problem definitions --- the ground-truth models that Bayesian Optimization queries."""

from bopt.oracles.base import Objective, Oracle, Parameter

__all__ = ["Oracle", "Parameter", "Objective"]
