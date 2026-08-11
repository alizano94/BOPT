"""Bayesian Optimization with Gaussian Process surrogates, on a mechanistic oracle.

Layout
------
``bopt.oracles``   problem definitions --- the world, in physical units
``bopt.adapters``  translation to optimizer conventions (torch, unit cube, sign flips)
``bopt.bo``        surrogates, acquisition functions, the optimization loop
``bopt.viz``       plotting

Dependencies run one way only: ``oracles`` imports nothing from the rest.
"""

__all__: list[str] = []
