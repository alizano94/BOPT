"""Compare the port against 10,000 evaluations from real Summit.

Summit cannot be installed alongside BoTorch (Python <3.11 versus >=3.11), so the
reference implementation is unavailable at test time. Instead this compares
against ``data/golden/summit_snar_nsga2.csv``, captured from Summit's own
repository. See ``data/golden/README.md`` for provenance.

Two properties of that fixture shape every assertion here:

1. **It uses the pre-2020-06-20 E-factor**, which counted the product as its own
   waste, so ``E_old == E_current + 1`` exactly. Diagnosed from the residuals
   before being confirmed against Summit's git history.
2. **Its values are numerically imprecise.** Summit integrates at scipy's default
   ``rtol=1e-3``; measured against a converged solution the stored numbers carry
   a median error of 2e-3 and a maximum of 70%, while our port at *identical*
   tolerances is accurate to 6e-4.

Point 2 is why nothing here asserts pointwise equality. The thresholds are set by
the reference data's own uncertainty, not by our tolerance for error --- demanding
tighter agreement than the fixture can support would be testing scipy's 2020
step-size heuristics, not our port.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from bopt.oracles import SnarOracle
from bopt.oracles.snar import SUMMIT_ATOL, SUMMIT_RTOL

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "golden" / "summit_snar_nsga2.csv"

#: The pre-2020-06-20 E-factor included the product in the waste sum.
LEGACY_E_FACTOR_OFFSET = 1.0


@pytest.fixture(scope="module")
def summit():
    """(X, Y) from the fixture: 4 decision variables, then sty and e_factor."""
    with FIXTURE.open() as handle:
        rows = list(csv.DictReader(handle))
    X = np.array(
        [[float(r[c]) for c in ("tau", "equiv_pldn", "conc_dfnb", "temperature")] for r in rows]
    )
    Y = np.array([[float(r["sty"]), float(r["e_factor"])] for r in rows])
    return X, Y


@pytest.fixture(scope="module")
def ported(summit):
    """Our port at Summit's own integration tolerances, for a like-for-like diff."""
    X, _ = summit
    return SnarOracle(rtol=SUMMIT_RTOL, atol=SUMMIT_ATOL).evaluate(X)


def _relative_error(ours, theirs):
    return np.abs(ours - theirs) / np.abs(theirs)


# ---------------------------------------------------------------------- fixture


def test_fixture_is_present_and_the_expected_size(summit):
    X, Y = summit
    assert X.shape == (10_000, 4)
    assert Y.shape == (10_000, 2)
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(Y))


def test_fixture_inputs_lie_inside_the_declared_domain(summit):
    """If Summit's own search stayed in bounds, our Parameter ranges match theirs."""
    X, _ = summit
    low, high = SnarOracle().bounds
    assert np.all(X >= low - 1e-9) and np.all(X <= high + 1e-9)


# ----------------------------------------------------------------- space-time yield


def test_sty_agrees_with_summit(summit, ported):
    _, Y = summit
    error = _relative_error(ported[:, 0], Y[:, 0])
    assert np.median(error) < 3e-3
    assert np.percentile(error, 90) < 2e-2
    assert np.mean(error < 1e-2) > 0.85


# ------------------------------------------------------------------------ E-factor


def test_e_factor_agrees_once_the_legacy_offset_is_applied(summit, ported):
    _, Y = summit
    error = _relative_error(ported[:, 1] + LEGACY_E_FACTOR_OFFSET, Y[:, 1])
    assert np.median(error) < 3e-3
    assert np.percentile(error, 90) < 2e-2
    assert np.mean(error < 1e-2) > 0.90


def test_the_legacy_offset_is_necessary(summit, ported):
    """Without the +1 the agreement is ~100x worse.

    Pins the archaeology: the fixture really was produced by the older formula,
    and the offset is not a fudge factor chosen to make a test pass.
    """
    _, Y = summit
    with_offset = np.median(_relative_error(ported[:, 1] + LEGACY_E_FACTOR_OFFSET, Y[:, 1]))
    without = np.median(_relative_error(ported[:, 1], Y[:, 1]))
    assert without > 50 * with_offset


def test_the_recovered_offset_is_one(summit, ported):
    """E_old = (solvent + waste + product)/product = E_current + 1, exactly.

    Recovering the offset empirically must give 1, not some fitted value --- that
    is what distinguishes a diagnosis from a fudge factor.

    The tolerance is 0.03 rather than something tighter because the offset is
    exact only in the *formulas*. Estimating it from data means differencing two
    E-factors near 8-10 that each carry ~1e-3 relative error, which propagates to
    roughly +/-0.01. Observed: 0.992. Demanding better would be asserting that
    the fixture is more precise than it is, which is the trap this module's
    docstring warns about --- and 0.03 still separates an offset of 1 from 0 or 2
    by a wide margin.
    """
    _, Y = summit
    offset = np.median(Y[:, 1] - ported[:, 1])
    assert offset == pytest.approx(1.0, abs=0.03)


# -------------------------------------------------------- the fixture's own error


def test_our_port_is_more_accurate_than_the_reference_data(summit, ported):
    """Justifies the loose thresholds above, and is worth asserting in its own right.

    At identical tolerance settings our port sits far closer to the converged
    solution than Summit's stored values do. The residual disagreement is
    therefore imprecision in the fixture, not in the port --- so tightening these
    tests would be testing scipy's 2020 step-size heuristics.
    """
    X, Y = summit
    subset = slice(0, 3000)  # converged integration is ~11x slower; a sample suffices
    truth = SnarOracle().evaluate(X[subset])[:, 0]

    ours = _relative_error(ported[subset, 0], truth)
    theirs = _relative_error(Y[subset, 0], truth)

    assert np.median(ours) < np.median(theirs)
    assert ours.max() < 1e-2
    assert theirs.max() > 0.1


# ------------------------------------------------------------ structural agreement


def test_pareto_structure_matches_summits_own_data(summit, ported):
    """The headline finding, checked against the reference implementation.

    The SnAr objectives are nearly independent: across Summit's own front the
    E-factor spans barely 15% while STY spans ~4x, and conc_dfnb pins to its
    upper bound throughout. Because ``x -> x + 1`` is strictly monotone, the
    non-dominated *set* is identical under both E-factor definitions, so this
    comparison is unaffected by the legacy offset.
    """
    X, Y = summit

    def front(objectives):
        Z = np.column_stack([objectives[:, 0], -objectives[:, 1]])
        order = np.lexsort((-Z[:, 1], -Z[:, 0]))
        mask = np.zeros(len(Z), dtype=bool)
        best = -np.inf
        for i in order:
            if Z[i, 1] > best:
                mask[i] = True
                best = Z[i, 1]
        return mask

    theirs, ours = front(Y), front(ported)

    for mask, objectives in ((theirs, Y), (ours, ported)):
        f = objectives[mask]
        assert f[:, 0].max() / f[:, 0].min() > 3.0, "STY should span several-fold"
        assert f[:, 1].max() / f[:, 1].min() < 1.3, "E-factor should barely move"
        assert np.all(X[mask][:, 2] > 0.49), "conc_dfnb should pin to its upper bound"
