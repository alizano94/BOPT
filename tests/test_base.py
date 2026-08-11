"""Tests for the Oracle base class.

``base.py`` is pure deterministic logic --- no physics, no randomness, no
integration --- which makes it both the cheapest thing in the project to test
exhaustively and the most valuable. It sits underneath everything else, so a bug
here surfaces later disguised as a physics bug or a BO bug, in the wrong file.

Bias throughout: test the failures that would *look like success*. A crash finds
itself; a silently-skipped clip or a NaN that slips through the bounds check does
not.
"""

import numpy as np
import pytest

from bopt.oracles import Objective, Oracle, Parameter


class Spy(Oracle):
    """Minimal concrete Oracle that records what the base class handed it.

    The two spans differ by 90x on purpose: it makes the span-relative tolerance
    behaviour observable rather than a matter of trust.
    """

    parameters = (
        Parameter("alpha", 0.0, 1.0, "-"),  # span 1
        Parameter("temperature", 30.0, 120.0, "degC"),  # span 90
    )
    objectives = (
        Objective("yield_", maximize=True, unit="%"),
        Objective("cost", maximize=False, unit="USD"),
    )

    def _evaluate(self, X):
        self.received = X
        return np.zeros((len(X), self.n_objectives))


@pytest.fixture
def spy():
    return Spy()


# --------------------------------------------------------------------- metadata


def test_metadata_is_derived_from_parameters(spy):
    assert spy.dim == 2
    assert spy.n_objectives == 2
    assert spy.parameter_names == ("alpha", "temperature")
    assert spy.objective_names == ("yield_", "cost")


def test_bounds_matrix_shape_and_content(spy):
    assert spy.bounds.shape == (2, spy.dim)
    np.testing.assert_array_equal(spy.bounds[0], [0.0, 30.0])
    np.testing.assert_array_equal(spy.bounds[1], [1.0, 120.0])


def test_repr_shows_ranges_and_directions(spy):
    text = repr(spy)
    assert "alpha[0.0,1.0]" in text
    assert "max yield_" in text and "min cost" in text


# ------------------------------------------------------------ input validation


def test_one_dimensional_input_is_promoted_to_a_single_row(spy):
    spy.evaluate([0.5, 60.0])
    assert spy.received.shape == (1, 2)


def test_integer_input_is_coerced_to_float(spy):
    """Catches a missing dtype=float: ints surviving would later do integer division."""
    spy.evaluate(np.array([[0, 60]]))
    assert spy.received.dtype == np.float64


def test_wrong_number_of_columns_names_the_expected_parameters(spy):
    with pytest.raises(ValueError, match="expected 2"):
        spy.evaluate([[0.5, 60.0, 1.0]])


def test_three_dimensional_input_is_rejected(spy):
    with pytest.raises(ValueError, match="1-D or 2-D"):
        spy.evaluate(np.zeros((2, 2, 2)))


def test_call_is_an_alias_for_evaluate(spy):
    out = spy([[0.5, 60.0]])
    assert out.shape == (1, 2)


def test_evaluate_returns_one_row_per_input_row(spy):
    out = spy.evaluate([[0.1, 40.0], [0.2, 50.0], [0.3, 60.0]])
    assert out.shape == (3, 2)


# -------------------------------------------------------------- bounds: inside


def test_points_exactly_on_the_bounds_are_accepted(spy):
    """Bounds are inclusive. Optimizers land exactly on them constantly, so this
    is a common case rather than an exotic one --- and '<' where '<=' was meant
    would make the declared domain unreachable at its edges."""
    spy.evaluate([[0.0, 30.0]])
    spy.evaluate([[1.0, 120.0]])


def test_interior_points_pass_through_unmodified(spy):
    X = [[0.5, 60.0]]
    spy.evaluate(X)
    np.testing.assert_array_equal(spy.received, X)


# ------------------------------------------------------------- bounds: clipping


def test_slight_overrun_is_clipped_to_the_exact_bound(spy):
    """The single most important assertion in this file.

    Within tolerance we clip and proceed. If the clip silently did not happen,
    nothing would raise and the numbers would still look plausible --- while the
    physics quietly received out-of-domain input. A passing-looking failure.
    """
    spy.evaluate([[1.0 + 1e-11, 60.0]])  # tolerance for alpha is 1e-9
    assert spy.received[0, 0] == 1.0


def test_slight_underrun_is_clipped_to_the_exact_bound(spy):
    spy.evaluate([[-1e-11, 60.0]])
    assert spy.received[0, 0] == 0.0


# ----------------------------------------------------------- bounds: violations


def test_violation_beyond_tolerance_raises(spy):
    with pytest.raises(ValueError, match="outside the domain"):
        spy.evaluate([[1.0 + 1e-6, 60.0]])


def test_error_message_names_the_offending_parameter(spy):
    """An error saying only 'out of bounds' is nearly worthless inside a 4-D BO
    loop three hours in. Message quality is debuggability, so it is tested."""
    with pytest.raises(ValueError, match="temperature"):
        spy.evaluate([[0.5, 200.0]])


def test_error_message_reports_every_violation(spy):
    with pytest.raises(ValueError) as excinfo:
        spy.evaluate([[5.0, 200.0], [0.5, 60.0], [-3.0, 60.0]])
    message = str(excinfo.value)
    assert "3 value(s)" in message
    assert "alpha" in message and "temperature" in message


# ------------------------------------------------------------------- tolerances


def test_tolerance_is_proportional_to_span(spy):
    """Pins down a design decision we deliberated: span-relative, not absolute
    and not value-relative. A comment could drift out of true; this cannot."""
    assert spy.tolerances[1] == pytest.approx(90.0 * spy.tolerances[0])


def test_default_tolerance_is_the_configured_fraction_of_span(spy):
    assert spy.tolerances[0] == pytest.approx(1e-9 * 1.0)
    assert spy.tolerances[1] == pytest.approx(1e-9 * 90.0)


def test_bounds_tol_is_tunable_per_instance():
    loose = Spy(bounds_tol=1e-3)
    loose.evaluate([[1.0 + 1e-4, 60.0]])  # would raise at the default tolerance
    assert loose.received[0, 0] == 1.0


def test_per_parameter_tol_overrides_the_default():
    class Overridden(Spy):
        parameters = (
            Parameter("alpha", 0.0, 1.0, "-", tol=0.5),  # absolute, deliberately huge
            Parameter("temperature", 30.0, 120.0, "degC"),
        )

    oracle = Overridden()
    assert oracle.tolerances[0] == 0.5
    assert oracle.tolerances[1] == pytest.approx(1e-9 * 90.0)  # untouched
    oracle.evaluate([[1.4, 60.0]])
    assert oracle.received[0, 0] == 1.0


# ---------------------------------------------------------------- non-finite


def test_nan_is_rejected(spy):
    """The guard that would otherwise be missing.

    Every comparison against NaN is False, including ``nan > tol``. Without an
    explicit finite check a NaN passes the bounds test, survives the clip,
    reaches the integrator, and returns a NaN objective that poisons the GP fit
    --- with the symptom appearing hundreds of lines from the cause.
    """
    with pytest.raises(ValueError, match="non-finite"):
        spy.evaluate([[np.nan, 60.0]])


def test_inf_is_rejected(spy):
    with pytest.raises(ValueError, match="non-finite"):
        spy.evaluate([[np.inf, 60.0]])


def test_non_finite_error_names_the_parameter(spy):
    with pytest.raises(ValueError, match="temperature"):
        spy.evaluate([[0.5, np.nan]])


# ------------------------------------------------------- construction contracts


def test_parameter_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="high > low"):
        Parameter("bad", 1.0, 0.0, "-")


def test_parameter_rejects_negative_tolerance():
    with pytest.raises(ValueError, match="tol must be"):
        Parameter("bad", 0.0, 1.0, "-", tol=-1.0)


def test_negative_bounds_tol_is_rejected():
    with pytest.raises(ValueError, match="bounds_tol"):
        Spy(bounds_tol=-1.0)


def test_subclass_without_parameters_fails_at_construction():
    """Loud at construction beats an AttributeError deep inside a BO loop."""

    class Incomplete(Oracle):
        objectives = (Objective("y", maximize=True, unit="-"),)

        def _evaluate(self, X):
            return np.zeros((len(X), 1))

    with pytest.raises(TypeError, match="parameters"):
        Incomplete()


def test_subclass_without_evaluate_cannot_be_instantiated():
    """Enforced by ABC itself, at instantiation --- the reason we chose ABC over
    Protocol. Worth an explicit test so the guarantee is visible."""

    class NoEvaluate(Oracle):
        parameters = (Parameter("a", 0.0, 1.0, "-"),)
        objectives = (Objective("y", maximize=True, unit="-"),)

    with pytest.raises(TypeError, match="abstract"):
        NoEvaluate()
