"""Tests for the BoTorch adapter.

The adapter's whole job is convention translation, so the tests are about
conventions: does the sign flip happen exactly once and in the right place, do
batch shapes survive, is everything double precision.

The sign tests carry the most weight. A sign error here would not raise --- it
would produce a BO run that diligently *minimizes* throughput while reporting
plausible-looking hypervolume, and nothing downstream could detect it.
"""

import numpy as np
import pytest
import torch

from bopt.adapters import BoTorchProblem
from bopt.oracles import Objective, Oracle, Parameter, SnarOracle

MILD = (1.0, 2.0, 0.3, 60.0)


class Doubler(Oracle):
    """Trivial oracle with one maximized and one minimized output.

    Deterministic and closed-form, so the sign and shape logic can be checked
    without any dependence on the reaction model.
    """

    parameters = (Parameter("a", 0.0, 1.0, "-"), Parameter("b", 0.0, 10.0, "-"))
    objectives = (
        Objective("up", maximize=True, unit="-"),
        Objective("down", maximize=False, unit="-"),
    )

    def _evaluate(self, X):
        return np.column_stack([X[:, 0] * 2.0, X[:, 1] * 3.0])


@pytest.fixture
def toy():
    return BoTorchProblem(Doubler())


@pytest.fixture(scope="module")
def snar():
    return BoTorchProblem(SnarOracle())


# ------------------------------------------------------------------- metadata


def test_bounds_are_a_two_by_dim_double_tensor(toy):
    b = toy.bounds
    assert b.shape == (2, 2)
    assert b.dtype == torch.double
    torch.testing.assert_close(b[0], torch.tensor([0.0, 0.0], dtype=torch.double))
    torch.testing.assert_close(b[1], torch.tensor([1.0, 10.0], dtype=torch.double))


def test_metadata_passes_through(snar):
    assert snar.dim == 4
    assert snar.num_objectives == 2
    assert snar.parameter_names == ("tau", "equiv_pldn", "conc_dfnb", "temperature")
    assert snar.objective_names == ("sty", "e_factor")


def test_repr_shows_the_sign_convention(snar):
    assert "+sty" in repr(snar) and "-e_factor" in repr(snar)


# ---------------------------------------------------------------- sign handling


def test_minimized_objective_is_negated_and_maximized_one_is_not(toy):
    Y = toy.evaluate(torch.tensor([[0.5, 2.0]], dtype=torch.double))
    assert Y[0, 0].item() == pytest.approx(1.0)  # up: 0.5*2, unchanged
    assert Y[0, 1].item() == pytest.approx(-6.0)  # down: 2*3, negated


def test_to_physical_inverts_the_sign_flip(toy):
    X = torch.tensor([[0.3, 4.0], [0.9, 7.0]], dtype=torch.double)
    physical = toy.to_physical(toy.evaluate(X))
    expected = torch.tensor([[0.6, 12.0], [1.8, 21.0]], dtype=torch.double)
    torch.testing.assert_close(physical, expected)


def test_round_trip_reproduces_the_oracle_exactly(snar):
    """The adapter must be a pure convention change, not a numerical one."""
    X = np.array([MILD, (0.5, 1.0, 0.5, 120.0), (2.0, 5.0, 0.5, 30.0)])
    direct = snar.oracle.evaluate(X)
    through = snar.to_physical(snar.evaluate(torch.as_tensor(X, dtype=torch.double)))
    np.testing.assert_allclose(through.numpy(), direct, rtol=0, atol=0)


def test_sign_flip_is_an_involution(toy):
    Y = torch.tensor([[3.0, -4.0]], dtype=torch.double)
    torch.testing.assert_close(toy.to_physical(toy.from_physical(Y)), Y)


def test_maximizing_the_adapted_output_minimizes_the_physical_one(snar):
    """The property that actually matters, stated end to end.

    A point that is better in all-maximize space must have a *lower* physical
    E-factor. If this inverts, BO optimizes the wrong direction while every
    intermediate number still looks reasonable.
    """
    X = torch.tensor([[1.0, 1.2, 0.5, 60.0], [2.0, 5.0, 0.5, 120.0]], dtype=torch.double)
    Y = snar.evaluate(X)
    physical = snar.to_physical(Y)
    better = int(Y[:, 1].argmax())
    assert physical[better, 1] == physical[:, 1].min()


# -------------------------------------------------------------------- shapes


def test_single_point_as_one_dimensional_input(toy):
    assert toy.evaluate(torch.tensor([0.5, 2.0], dtype=torch.double)).shape == (2,)


def test_n_by_d_input(toy):
    assert toy.evaluate(torch.rand(7, 2, dtype=torch.double) * 0.5).shape == (7, 2)


def test_leading_batch_dimensions_are_preserved(toy):
    """BoTorch routinely passes b x q x d during acquisition optimization."""
    X = torch.rand(3, 5, 2, dtype=torch.double) * 0.5
    assert toy.evaluate(X).shape == (3, 5, 2)


def test_wrong_column_count_is_rejected(toy):
    with pytest.raises(ValueError, match="expected 2"):
        toy.evaluate(torch.rand(4, 3, dtype=torch.double))


def test_numpy_and_lists_are_accepted(toy):
    torch.testing.assert_close(toy.evaluate([[0.5, 2.0]]), toy.evaluate(np.array([[0.5, 2.0]])))


# --------------------------------------------------------------------- dtype


def test_output_is_double_even_from_float32_input(toy):
    """BoTorch is numerically fragile in single precision and warns about it."""
    Y = toy.evaluate(torch.tensor([[0.5, 2.0]], dtype=torch.float32))
    assert Y.dtype == torch.double


def test_bounds_validation_is_still_enforced_by_the_oracle(snar):
    with pytest.raises(ValueError, match="temperature"):
        snar.evaluate(torch.tensor([[1.0, 2.0, 0.3, 500.0]], dtype=torch.double))


# ------------------------------------------------------- does it actually plug in


def test_output_can_be_fitted_by_a_botorch_gp(snar):
    """End-to-end proof that the adapter produces model-ingestible tensors.

    Uses the transforms this project relies on --- ``Normalize`` with explicit
    domain bounds (rather than data-inferred) and ``Standardize`` --- so the test
    exercises the real intended wiring, not a simplified stand-in.
    """
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.manual_seed(0)
    low, high = snar.bounds
    X = low + torch.rand(12, snar.dim, dtype=torch.double) * (high - low)
    Y = snar.evaluate(X)

    model = SingleTaskGP(
        X,
        Y,
        input_transform=Normalize(d=snar.dim, bounds=snar.bounds),
        outcome_transform=Standardize(m=snar.num_objectives),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

    test_X = low + torch.rand(5, snar.dim, dtype=torch.double) * (high - low)
    posterior = model.posterior(test_X)
    assert posterior.mean.shape == (5, snar.num_objectives)
    assert torch.all(posterior.variance > 0)
