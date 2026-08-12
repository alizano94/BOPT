"""Tests for the optimization layer.

Weighted towards the guarantees that make the benchmark *fair*, rather than
towards whether BoTorch works:

* every arm starts from the same points at a given seed, so a measured gap is
  about proposal rules and not initial-design luck;
* the clean oracle's values never reach a strategy, so no arm can be rewarded for
  lucky measurements;
* budget accounting is identical across arms, including one that terminates early.

GP-based arms are exercised sparingly and at tiny budgets --- a qNEHVI proposal
costs seconds, and this suite should stay runnable.
"""

import numpy as np
import pytest
import torch

from bopt.adapters import BoTorchProblem
from bopt.bo import (
    REFERENCE_POINT,
    TRUE_FRONT_HYPERVOLUME,
    RunRecord,
    build_strategy,
    fit_surrogate,
    hypervolume,
    hypervolume_fraction,
    initial_design,
    lengthscales,
    reference_point,
    run,
)
from bopt.oracles import SnarOracle


@pytest.fixture(scope="module")
def problem():
    return BoTorchProblem(SnarOracle())


@pytest.fixture(scope="module")
def bounds(problem):
    return problem.bounds


# ------------------------------------------------------------------- reference


def test_reference_point_is_pinned():
    """Hypervolume is only comparable if this never moves. Pinned deliberately."""
    assert REFERENCE_POINT == (1985.4606, -9.8860)
    assert TRUE_FRONT_HYPERVOLUME == pytest.approx(12911.7096)


def test_reference_point_tensor_is_double(problem):
    assert reference_point().dtype == torch.double


def test_empty_set_has_no_hypervolume():
    assert hypervolume(torch.empty(0, 2, dtype=torch.double)) == 0.0


def test_points_dominated_by_the_reference_contribute_nothing():
    below = torch.tensor([[0.0, -500.0]], dtype=torch.double)
    assert hypervolume(below) == 0.0


def test_hypervolume_is_monotone_under_adding_points():
    a = torch.tensor([[8000.0, -9.0]], dtype=torch.double)
    b = torch.tensor([[8000.0, -9.0], [11000.0, -9.7]], dtype=torch.double)
    assert hypervolume(b) >= hypervolume(a)


def test_the_best_sty_point_captures_about_a_tenth_of_the_front():
    """Records the number that revised prediction #2 in the findings doc: peak
    throughput alone is worth ~9.7% of the front, so the single-objective arm is
    a genuine comparison rather than a formality."""
    best_sty = torch.tensor([[11560.2759, -9.7556]], dtype=torch.double)
    assert hypervolume_fraction(best_sty) == pytest.approx(0.0967, abs=2e-3)


def test_hypervolume_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="expected"):
        hypervolume(torch.zeros(4, 3, dtype=torch.double))


# ------------------------------------------------------------------- surrogate


def test_surrogate_fits_and_exposes_ard_lengthscales(problem, bounds):
    torch.manual_seed(0)
    X = initial_design(bounds, 10, seed=0)
    Y = problem.evaluate(X)
    model = fit_surrogate(X, Y, bounds)

    scales = lengthscales(model)
    assert scales is not None
    assert scales.shape == (problem.num_objectives, problem.dim), "one lengthscale per input, per objective"
    assert torch.all(scales > 0)


def test_surrogate_rejects_mismatched_rows(bounds):
    with pytest.raises(ValueError, match="rows"):
        fit_surrogate(torch.zeros(5, 4, dtype=torch.double), torch.zeros(3, 2, dtype=torch.double), bounds)


# ------------------------------------------------------------------ strategies


@pytest.mark.parametrize("name", ["sobol", "grid"])
def test_cheap_strategies_propose_the_requested_shape(name, bounds):
    strategy = build_strategy(name, bounds, seed=0)
    out = strategy.propose(torch.empty(0, 4, dtype=torch.double), torch.empty(0, 2), q=4)
    assert out.shape == (4, 4)
    assert torch.all(out >= bounds[0]) and torch.all(out <= bounds[1])


def test_sobol_is_reproducible_and_not_constant(bounds):
    a = build_strategy("sobol", bounds, seed=3).propose(None, None, q=6)
    b = build_strategy("sobol", bounds, seed=3).propose(None, None, q=6)
    c = build_strategy("sobol", bounds, seed=4).propose(None, None, q=6)
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)


def test_grid_is_a_two_level_full_factorial_and_then_exhausts(bounds):
    grid = build_strategy("grid", bounds, seed=0)
    points = grid.propose(None, None, q=64)
    assert len(points) == 16, "2^4 corners"
    # every coordinate sits on a bound -- that is what a 2-level factorial is
    on_bound = (points == bounds[0]) | (points == bounds[1])
    assert torch.all(on_bound)
    assert len(grid.propose(None, None, q=4)) == 0, "nothing left to offer"


def test_unknown_strategy_names_are_rejected(bounds):
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy("nope", bounds, seed=0)


# ------------------------------------------------------------- fairness of setup


def test_every_arm_starts_from_identical_points(bounds):
    """The guarantee that makes the comparison about proposal rules."""
    records = [run(name, seed=7, budget=12, n_init=8, q=4) for name in ("sobol", "grid")]
    np.testing.assert_array_equal(records[0].X[:8], records[1].X[:8])


def test_initial_design_depends_on_the_seed(bounds):
    assert not torch.allclose(initial_design(bounds, 8, seed=0), initial_design(bounds, 8, seed=1))


def test_initial_design_lies_inside_the_domain(bounds):
    X = initial_design(bounds, 32, seed=0)
    assert torch.all(X >= bounds[0]) and torch.all(X <= bounds[1])


# --------------------------------------------------------- the noise protocol


def test_noisy_and_clean_observations_differ_when_noise_is_on():
    """Proves the two-oracle protocol is actually live.

    If these ever coincided, reported hypervolume would be computed from the same
    noisy numbers the strategy optimized against, and 'best observed' would be
    optimistically biased.
    """
    record = run("sobol", seed=0, budget=12, n_init=12, q=4, noise_level=1.0)
    assert not np.allclose(record.Y_noisy, record.Y_true)
    assert np.allclose(record.Y_noisy, record.Y_true, rtol=0.15), "1% noise, not chaos"


def test_noise_free_runs_have_identical_observations():
    record = run("sobol", seed=0, budget=12, n_init=12, q=4, noise_level=0.0)
    np.testing.assert_allclose(record.Y_noisy, record.Y_true)


def test_reported_hypervolume_comes_from_the_clean_values():
    record = run("sobol", seed=1, budget=12, n_init=12, q=4, noise_level=1.0)
    assert record.final_hv_fraction == pytest.approx(hypervolume_fraction(record.Y_true))
    assert record.final_hv_fraction != pytest.approx(hypervolume_fraction(record.Y_noisy))


# -------------------------------------------------------------- budget and loop


def test_budget_is_respected_exactly(bounds):
    record = run("sobol", seed=0, budget=14, n_init=6, q=4)
    assert record.n_evaluated == 14, "final batch is truncated to land on the budget"
    assert record.unused_budget == 0


def test_an_exhausted_strategy_stops_early_and_records_the_shortfall():
    record = run("grid", seed=0, budget=64, n_init=16, q=4)
    assert record.exhausted
    assert record.n_evaluated == 32  # 16 shared initial + 16 factorial points
    assert record.unused_budget == 32


def test_n_init_larger_than_budget_is_rejected():
    with pytest.raises(ValueError, match="exceeds budget"):
        run("sobol", seed=0, budget=4, n_init=8)


def test_trace_is_indexed_by_evaluation_count(bounds):
    record = run("sobol", seed=0, budget=20, n_init=8, q=4)
    n, hv = record.trace()
    assert list(n) == [8, 12, 16, 20]
    assert np.all(np.diff(hv) >= -1e-12), "hypervolume can only grow as points accumulate"


def test_hypervolume_fraction_at_a_smaller_budget(bounds):
    record = run("sobol", seed=0, budget=20, n_init=8, q=4)
    assert record.hv_fraction_at(8) == pytest.approx(record.trace()[1][0])
    assert record.hv_fraction_at(20) == pytest.approx(record.final_hv_fraction)


# ---------------------------------------------------------------- serialization


def test_records_round_trip_through_npz(tmp_path):
    record = run("sobol", seed=0, budget=12, n_init=8, q=4)
    path = tmp_path / "run.npz"
    record.save(path)

    loaded = np.load(path, allow_pickle=False)
    assert str(loaded["strategy"]) == "sobol"
    assert int(loaded["seed"]) == 0
    np.testing.assert_allclose(loaded["X"], record.X)
    np.testing.assert_allclose(loaded["Y_true"], record.Y_true)
    assert loaded["trace_hv"].shape == loaded["trace_n"].shape


def test_empty_record_has_sane_defaults():
    record = RunRecord(strategy="x", seed=0, budget=10, q=4, noise_level=0.0)
    assert record.final_hv_fraction == 0.0
    assert record.n_evaluated == 0


# ------------------------------------------------------- the model-based arms


@pytest.mark.slow
def test_gp_arms_beat_random_at_a_small_budget():
    """Sanity check on the whole stack, at the smallest budget that separates.

    Marked slow: each qNEHVI proposal takes seconds.
    """
    kwargs = dict(seed=0, budget=20, n_init=12, q=4)
    sobol = run("sobol", **kwargs).final_hv_fraction
    qnehvi = run("qnehvi", **kwargs).final_hv_fraction
    assert qnehvi > sobol


@pytest.mark.slow
def test_single_objective_arm_models_only_throughput(bounds):
    record = run("sty_only", seed=0, budget=16, n_init=12, q=4)
    assert record.n_evaluated == 16
    scales = [it.lengthscales for it in record.iterations if it.lengthscales is not None]
    assert scales and scales[-1].shape == (1, 4), "one objective modelled, not two"
