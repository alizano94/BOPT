"""Tests for the ported SnAr reaction model.

The port has no reference implementation available in this environment (Summit
cannot coexist with BoTorch), so correctness has to come from properties the
model must satisfy regardless of what upstream did:

* **conservation** --- summing the rate laws must cancel exactly. This catches any
  sign or coefficient error in the transcription without needing an oracle to
  compare against.
* **closed-form identities** --- STY and the E-factor were derived symbolically in
  ``docs/snar-benchmark.md``; the code must agree with the algebra.
* **physical monotonicity** --- higher activation energy must mean a steeper
  response to temperature, since that is what the whole trade-off rests on.

A separate golden-fixture test compares against real Summit output captured in a
disposable Python 3.10 environment. These tests stand on their own without it.
"""

import numpy as np
import pytest

from bopt.oracles import IntegrationError, SnarOracle
from bopt.oracles import snar as snar_module
from bopt.oracles.snar import (
    AMINE,
    BIS_ADDUCT,
    DEFAULT_RTOL,
    ISOMER,
    MOLAR_MASS,
    PRODUCT,
    SUBSTRATE,
    SUMMIT_ATOL,
    SUMMIT_RTOL,
    _DEPLETION_FRACTION,
    _K_REF,
    _RATE_UNIT_CONVERSION,
    _SUMMIT_KELVIN_OFFSET,
)

# Reference conditions where the substrate is never driven to exhaustion, so the
# depletion clamp stays out of the way and conservation should be exact.
MILD = (1.0, 2.0, 0.3, 60.0)

# The opposite corner: hot, amine-rich and long. The substrate is consumed
# entirely, the clamp fires, and the product is eaten by over-reaction.
HARSH = (2.0, 5.0, 0.5, 120.0)


@pytest.fixture
def oracle():
    return SnarOracle()


# ------------------------------------------------------------------- kinetics


def test_rate_constants_at_reference_temperature_are_just_the_unit_conversion():
    """At T_ref the exponential is exactly 1, so only the 1e-2 M^-1 s^-1 ->
    M^-1 min^-1 conversion remains. Pins both the reference values and the 0.6."""
    k = SnarOracle.rate_constants(90.0)
    np.testing.assert_allclose(k, _RATE_UNIT_CONVERSION * _K_REF, rtol=1e-15)


def test_desired_route_is_the_fastest_at_reference():
    k_a, k_b, k_c, k_d = SnarOracle.rate_constants(90.0)
    assert k_a > 20 * k_b, "desired route should dominate the isomer route ~21x"
    assert k_a > k_b > k_d > k_c


def test_all_rate_constants_increase_with_temperature():
    cold = SnarOracle.rate_constants(30.0)
    hot = SnarOracle.rate_constants(120.0)
    assert np.all(hot > cold)


def test_higher_activation_energy_means_steeper_temperature_response():
    """The central physical claim of the whole problem.

    Activation energies are ordered k_a < k_b < k_c < k_d, so their sensitivity
    to temperature must be ordered the same way. This is *why* heating the
    reactor costs selectivity: the desired route responds least.
    """
    sensitivity = SnarOracle.rate_constants(120.0) / SnarOracle.rate_constants(30.0)
    assert sensitivity[0] < sensitivity[1] < sensitivity[2] < sensitivity[3]


def test_over_reaction_gains_ground_as_the_reactor_is_heated():
    cold = SnarOracle.rate_constants(30.0)
    hot = SnarOracle.rate_constants(120.0)
    assert hot[2] / hot[0] > cold[2] / cold[0]  # k_c / k_a
    assert hot[3] / hot[0] > cold[3] / cold[0]  # k_d / k_a


def test_regioselectivity_degrades_less_than_over_reaction():
    """Separates the two ways heat costs you selectivity.

    k_a and k_b differ by only 2 kJ/mol, while k_c and k_d sit 5.6 and 11.5 above
    k_a, so the isomer split degrades far more slowly than the over-reaction. It
    is not negligible though: over the full range the ratio still falls ~17%.
    An earlier draft of the docs claimed ~5%, which was the 90->120 figure
    mislabelled as 30->120; this test pins the correct one.
    """
    cold = SnarOracle.rate_constants(30.0)
    hot = SnarOracle.rate_constants(120.0)
    isomer = (hot[1] / hot[0]) / (cold[1] / cold[0])
    over_product = (hot[2] / hot[0]) / (cold[2] / cold[0])
    over_isomer = (hot[3] / hot[0]) / (cold[3] / cold[0])

    assert isomer < over_product < over_isomer
    assert isomer == pytest.approx(1.20, abs=0.02)
    assert (hot[0] / hot[1]) / (cold[0] / cold[1]) == pytest.approx(0.834, abs=0.01)


# ------------------------------------------------- conservation (the RHS alone)


@pytest.mark.parametrize("temperature", [30.0, 75.0, 120.0])
def test_rate_laws_conserve_the_aromatic_core(temperature):
    """Every aromatic species is one ring: substrate, both mono-adducts and the
    bis-adduct must sum to a constant. A sign error anywhere breaks this."""
    rng = np.random.default_rng(0)
    k = SnarOracle.rate_constants(temperature)
    C_initial = np.array([0.5, 2.5, 0.0, 0.0, 0.0])
    for _ in range(20):
        C = rng.uniform(0.01, 1.0, size=5)  # all well above the clamp threshold
        r = SnarOracle._rhs(0.0, C.copy(), k, C_initial)
        assert r[[SUBSTRATE, PRODUCT, ISOMER, BIS_ADDUCT]].sum() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("temperature", [30.0, 75.0, 120.0])
def test_rate_laws_conserve_amine(temperature):
    """Each mono-adduct locks up one pyrrolidine and the bis-adduct two."""
    rng = np.random.default_rng(1)
    k = SnarOracle.rate_constants(temperature)
    C_initial = np.array([0.5, 2.5, 0.0, 0.0, 0.0])
    for _ in range(20):
        C = rng.uniform(0.01, 1.0, size=5)
        r = SnarOracle._rhs(0.0, C.copy(), k, C_initial)
        bound = r[AMINE] + r[PRODUCT] + r[ISOMER] + 2 * r[BIS_ADDUCT]
        assert bound == pytest.approx(0.0, abs=1e-12)


def test_bis_adduct_only_ever_accumulates():
    k = SnarOracle.rate_constants(90.0)
    C_initial = np.array([0.5, 2.5, 0.0, 0.0, 0.0])
    C = np.array([0.2, 0.9, 0.15, 0.05, 0.01])
    r = SnarOracle._rhs(0.0, C.copy(), k, C_initial)
    assert r[BIS_ADDUCT] > 0
    assert r[SUBSTRATE] < 0


# ------------------------------------------ conservation (through integration)


def test_conservation_holds_through_integration_when_the_clamp_is_idle(oracle):
    C_initial = oracle.initial_concentrations(MILD[1], MILD[2])
    C = oracle._integrate(*MILD).y[:, -1]
    core = C[[SUBSTRATE, PRODUCT, ISOMER, BIS_ADDUCT]].sum()
    core_0 = C_initial[[SUBSTRATE, PRODUCT, ISOMER, BIS_ADDUCT]].sum()
    assert core == pytest.approx(core_0, rel=1e-10)


def test_depletion_clamp_breaks_conservation_by_about_its_own_threshold():
    """Documents the cost of a defect we chose to preserve.

    The clamp zeroes a reactant once it falls below 1e-6 of its initial value, so
    at conditions that exhaust the substrate it destroys mass --- but only about
    that much. Asserting both bounds means the test fails if the clamp is
    silently removed *or* if it starts doing something larger than advertised.
    """
    oracle = SnarOracle()
    C_initial = oracle.initial_concentrations(HARSH[1], HARSH[2])
    C = oracle._integrate(*HARSH).y[:, -1]
    core = C[[SUBSTRATE, PRODUCT, ISOMER, BIS_ADDUCT]].sum()
    core_0 = C_initial[[SUBSTRATE, PRODUCT, ISOMER, BIS_ADDUCT]].sum()
    drift = abs(core - core_0) / core_0
    assert 1e-9 < drift < 2 * _DEPLETION_FRACTION


# ----------------------------------------------------------- initial conditions


def test_only_the_two_reactants_are_present_at_the_inlet():
    C = SnarOracle.initial_concentrations(equiv_pldn=3.0, conc_dfnb=0.2)
    assert C[SUBSTRATE] == 0.2
    assert C[AMINE] == pytest.approx(0.6)
    assert C[[PRODUCT, ISOMER, BIS_ADDUCT]].sum() == 0.0


# -------------------------------------------------------------------- objectives


@pytest.mark.parametrize("conditions", [MILD, HARSH, (0.5, 1.0, 0.5, 120.0)])
def test_sty_matches_the_closed_form(oracle, conditions):
    """STY = 60 * M_product * C_product / tau, derived in docs/snar-benchmark.md."""
    tau = conditions[0]
    C = oracle._integrate(*conditions).y[:, -1]
    expected = 60.0 * MOLAR_MASS[PRODUCT] * C[PRODUCT] / tau
    sty, _ = oracle._objectives(C, tau)
    assert sty == pytest.approx(expected, rel=1e-12)


def test_e_factor_does_not_depend_on_residence_time(oracle):
    """Volumetric flow cancels exactly between numerator and denominator, so at a
    fixed outlet composition the E-factor is invariant to tau. Pins the algebraic
    result that the docs rely on."""
    C = oracle._integrate(*MILD).y[:, -1]
    values = [oracle._objectives(C, tau)[1] for tau in (0.5, 1.0, 1.5, 2.0)]
    assert all(v == pytest.approx(values[0], rel=1e-12) for v in values)


def test_e_factor_is_dominated_by_solvent(oracle):
    """Ethanol contributes 789 g/L against tens of g/L of organics, so the metric
    should sit close to 3.75 / C_product."""
    C = oracle._integrate(*MILD).y[:, -1]
    _, e_factor = oracle._objectives(C, MILD[0])
    assert e_factor == pytest.approx(3.75 / C[PRODUCT], rel=0.15)


def test_no_product_gives_the_floor_and_the_ceiling():
    sty, e_factor = SnarOracle._objectives(np.zeros(5), tau=1.0)
    assert sty == snar_module._STY_FLOOR
    assert e_factor == snar_module._E_FACTOR_CEILING


def test_e_factor_is_capped():
    C = np.zeros(5)
    C[PRODUCT] = 1e-7  # non-zero but negligible
    C[SUBSTRATE] = 0.5
    _, e_factor = SnarOracle._objectives(C, tau=1.0)
    assert e_factor == snar_module._E_FACTOR_CEILING


def test_over_reaction_destroys_throughput(oracle):
    """The harsh corner should be far worse than the mild one on *both* axes ---
    excess amine converts product into bis-adduct."""
    (sty_mild, e_mild), (sty_harsh, e_harsh) = oracle.evaluate([MILD, HARSH])
    assert sty_harsh < sty_mild
    assert e_harsh > e_mild


# ------------------------------------------------------------------------ noise


def test_default_oracle_is_deterministic(oracle):
    a = oracle.evaluate([MILD])
    b = oracle.evaluate([MILD])
    np.testing.assert_array_equal(a, b)


def test_same_seed_reproduces_the_noise_stream():
    a = SnarOracle(noise_level=5.0, seed=42).evaluate([MILD])
    b = SnarOracle(noise_level=5.0, seed=42).evaluate([MILD])
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_noise():
    a = SnarOracle(noise_level=5.0, seed=1).evaluate([MILD])
    b = SnarOracle(noise_level=5.0, seed=2).evaluate([MILD])
    assert not np.allclose(a, b)


def test_noise_without_a_seed_warns():
    """An irreproducible noisy run is the sort of thing discovered a week later,
    when a result cannot be reproduced. Cheap to catch at construction."""
    with pytest.warns(UserWarning, match="reproducible"):
        SnarOracle(noise_level=1.0)


def test_noiseless_oracle_does_not_warn(recwarn):
    SnarOracle()
    assert len(recwarn) == 0


def test_negative_noise_level_is_rejected():
    with pytest.raises(ValueError, match="noise_level"):
        SnarOracle(noise_level=-1.0)


def test_noise_never_produces_negative_concentrations():
    oracle = SnarOracle(noise_level=500.0, seed=0)  # absurd, to force the clamp
    C = np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-9])
    assert np.all(oracle._apply_noise(C) >= 0)


# ------------------------------------------------------------ integration failure


def test_integration_failure_raises_and_carries_the_partial_state(oracle, monkeypatch):
    """Upstream never checks res.success and would use whatever is in the array.

    The partial state is attached to the exception rather than returned, so it
    can be inspected but not silently consumed.
    """

    class FailedSolve:
        success = False
        status = -1
        message = "step size became too small"
        y = np.arange(15, dtype=float).reshape(5, 3)

    monkeypatch.setattr(snar_module, "solve_ivp", lambda *a, **kw: FailedSolve())

    with pytest.raises(IntegrationError) as excinfo:
        oracle.evaluate([MILD])

    error = excinfo.value
    assert error.conditions["temperature"] == MILD[3]
    assert error.status == -1
    assert "step size" in error.solver_message
    np.testing.assert_array_equal(error.partial, [2.0, 5.0, 8.0, 11.0, 14.0])


# --------------------------------------------------------------------- contract


def test_evaluate_returns_one_row_of_two_objectives_per_input(oracle):
    Y = oracle.evaluate([MILD, HARSH, (0.5, 1.0, 0.5, 120.0)])
    assert Y.shape == (3, 2)
    assert np.all(np.isfinite(Y))


def test_parameter_and_objective_order_is_as_documented(oracle):
    assert oracle.parameter_names == ("tau", "equiv_pldn", "conc_dfnb", "temperature")
    assert oracle.objective_names == ("sty", "e_factor")
    assert [o.maximize for o in oracle.objectives] == [True, False]


def test_bounds_validation_is_inherited(oracle):
    with pytest.raises(ValueError, match="temperature"):
        oracle.evaluate([[1.0, 2.0, 0.3, 500.0]])


# ------------------------------------------------------------------- trajectory


def test_trajectory_shapes_and_endpoints(oracle):
    t, C = oracle.trajectory(*MILD, n_points=50)
    assert t.shape == (50,)
    assert C.shape == (5, 50)
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(MILD[0])


def test_trajectory_starts_at_the_inlet_composition(oracle):
    _, C = oracle.trajectory(*MILD)
    expected = oracle.initial_concentrations(MILD[1], MILD[2])
    np.testing.assert_allclose(C[:, 0], expected, atol=1e-9)


def test_product_peaks_inside_the_reactor_at_harsh_conditions(oracle):
    """The product is itself a substrate for a second attack, so it is *not*
    stable in the reactor. This is what makes a finite residence time optimal
    rather than 'run it to completion'.
    """
    _, C = oracle.trajectory(*HARSH, n_points=400)
    product = C[PRODUCT]
    peak = int(product.argmax())
    assert 0 < peak < len(product) - 1, "product should rise and then fall"
    assert product[-1] < 0.5 * product[peak]


# ------------------------------------------------------- preserved-defect pins


def test_summit_kelvin_offset_is_preserved():
    """273.71 is wrong --- absolute zero is 273.15 --- and is kept deliberately so
    results stay comparable to published Summit numbers. Pinned so that a
    well-meaning 'fix' has to be a conscious act that breaks a test.
    """
    assert _SUMMIT_KELVIN_OFFSET == 273.71


def test_summit_tolerances_are_available_and_differ_from_the_defaults():
    """The default is tight because an oracle serving as ground truth should not
    carry 1e-3 relative error; upstream's values remain reachable so the
    golden-fixture comparison can be exact."""
    assert SUMMIT_RTOL == 1e-3 and SUMMIT_ATOL == 1e-6
    assert DEFAULT_RTOL < SUMMIT_RTOL

    faithful = SnarOracle(rtol=SUMMIT_RTOL, atol=SUMMIT_ATOL)
    assert np.all(np.isfinite(faithful.evaluate([MILD])))


def test_loose_and_tight_tolerances_agree_to_a_few_digits(oracle):
    """Sanity bound on how much upstream's loose defaults actually cost. The
    convergence study quantifies this properly; this only guards against them
    being wildly different."""
    loose = SnarOracle(rtol=SUMMIT_RTOL, atol=SUMMIT_ATOL).evaluate([MILD])
    tight = oracle.evaluate([MILD])
    np.testing.assert_allclose(loose, tight, rtol=1e-3)
