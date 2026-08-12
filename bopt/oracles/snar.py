"""Nucleophilic aromatic substitution (S_NAr) in a plug-flow reactor.

A port of ``summit/benchmarks/snar.py`` from Summit 0.8.9, with kinetics from
C. A. Hone et al., *React. Chem. Eng.* 2017, **2**, 103-108
(https://doi.org/10.1039/C6RE00109B).

Ported rather than imported because Summit requires Python <3.11 while BoTorch
requires >=3.11; the two can never share an environment. The reaction model
depends only on numpy and scipy, so it transfers directly.

Chemistry
---------
2,4-difluoronitrobenzene is attacked by pyrrolidine, which displaces a fluoride.
Substitution can occur at either of two positions, giving the desired product or
an unwanted regioisomer; either mono-adduct can then be attacked a second time to
give a bis-adduct. Four second-order reactions in total::

      k_a (Ea 33.3)
    ---------------> [2] DESIRED --.
   /                                \\ k_c (Ea 38.9)
  [0] + [1]                          >--> [4] bis-adduct
   \\                                / k_d (Ea 44.8)
    ---------------> [3] isomer  --'
      k_b (Ea 35.3)

The desired route has the *lowest* activation energy of the four, so heating the
reactor accelerates everything but accelerates the parasitic routes faster.
Throughput is therefore bought with selectivity, which is the origin of the
trade-off between the two objectives.

Fidelity policy
---------------
The upstream source contains several defects. The rule applied here is:

    preserve anything that changes the objective value;
    fix anything that only affects reproducibility or clarity.

Preserving them keeps results comparable to published Summit numbers, and the
golden-fixture test pins the agreement empirically rather than by argument. Every
preserved defect below is a named constant or a commented block, so it is a
visible decision rather than an inherited bug.

Preserved: the wrong Kelvin offset (:data:`_SUMMIT_KELVIN_OFFSET`); the E-factor
charging the *entire* flow as ethanol solvent; the output clipping; the
multiplicative noise model; the reactant-depletion clamp inside the right-hand
side, including its mutation of the integrator's state array.

Fixed: the random number generator is seedable; the docstring names the right
compound.

Departed from deliberately: integration tolerances default to far tighter values
than Summit's, because an oracle serving as ground truth should not carry 1e-3
relative error. Pass ``rtol=SUMMIT_RTOL, atol=SUMMIT_ATOL`` to reproduce upstream
exactly. Integration failures raise instead of silently returning whatever is in
the solver's array.

See ``docs/snar-benchmark.md`` for the full derivation of both objectives.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from bopt.oracles.base import FloatArray, Objective, Oracle, Parameter

__all__ = ["SnarOracle", "IntegrationError", "SUMMIT_RTOL", "SUMMIT_ATOL"]


# --------------------------------------------------------------------- species

SUBSTRATE = 0  # 2,4-difluoronitrobenzene
AMINE = 1  # pyrrolidine
PRODUCT = 2  # desired mono-adduct
ISOMER = 3  # unwanted regioisomer of the mono-adduct
BIS_ADDUCT = 4  # doubly substituted, from over-reaction

#: Molar masses in g/mol, indexed by species. PRODUCT and ISOMER are isomers and
#: therefore share a mass; that degeneracy is chemistry, not a typo.
MOLAR_MASS = np.array([159.09, 71.12, 210.21, 210.21, 261.33])

# -------------------------------------------------------------------- kinetics

#: Reference rate constants at :data:`_T_REF_CELSIUS`, in 1e-2 / (M s), ordered
#: [k_a, k_b, k_c, k_d].
_K_REF = np.array([57.9, 2.70, 0.865, 1.63])

#: Activation energies in kJ/mol, same order. Note k_a -- the desired route --
#: has the lowest, which is what makes heat cost selectivity.
_E_ACTIVATION = np.array([33.3, 35.3, 38.9, 44.8])

#: Converts 1e-2 / (M s) to 1 / (M min):  1e-2 * 60 = 0.6.
_RATE_UNIT_CONVERSION = 0.6

_GAS_CONSTANT = 8.314 / 1000  # kJ / (K mol)
_T_REF_CELSIUS = 90.0

#: Summit converts Celsius to Kelvin with 273.71 rather than 273.15. Preserved
#: because it changes objective values. It is applied to both the temperature and
#: the reference temperature, so it partly cancels and the effect on rate
#: constants is well under a percent -- but it is wrong, and deliberately visible
#: here rather than buried in an expression.
_SUMMIT_KELVIN_OFFSET = 273.71  # sic

_T_REF_KELVIN = _T_REF_CELSIUS + _SUMMIT_KELVIN_OFFSET

#: Below this fraction of its initial value, a reactant is forced to zero inside
#: the right-hand side. See :meth:`SnarOracle._rhs`.
_DEPLETION_FRACTION = 1e-6

# --------------------------------------------------------------------- reactor

_REACTOR_VOLUME_ML = 5.0
_ETHANOL_DENSITY_G_PER_ML = 0.789

# -------------------------------------------------------------------- clipping

_STY_FLOOR = 1e-6
_E_FACTOR_CEILING = 1e3

# ----------------------------------------------------------------- integration

#: scipy's ``solve_ivp`` defaults, which Summit uses by not overriding them.
#: Pinned explicitly so a future change to scipy's defaults cannot silently move
#: what "reproduces Summit" means.
SUMMIT_RTOL = 1e-3
SUMMIT_ATOL = 1e-6

DEFAULT_RTOL = 1e-10
DEFAULT_ATOL = 1e-12


class IntegrationError(RuntimeError):
    """Raised when ``solve_ivp`` fails to converge.

    Upstream never checks ``res.success`` and would silently use whatever is in
    the solver's array. Failing loudly is a correctness fix, not a change to the
    objective: it only differs from upstream in cases where upstream produced
    garbage.

    The partial state is attached rather than returned, so that inspecting it is
    possible but ignoring it is not.

    Attributes
    ----------
    conditions
        The four decision variables that triggered the failure.
    partial
        Concentrations at the last successful step, or ``None`` if the solver
        produced nothing at all.
    status, solver_message
        Diagnostics straight from ``scipy``.
    """

    def __init__(self, message, *, conditions, partial, status, solver_message):
        super().__init__(message)
        self.conditions = conditions
        self.partial = partial
        self.status = status
        self.solver_message = solver_message


class SnarOracle(Oracle):
    """S_NAr in a plug-flow reactor: 4 continuous knobs, 2 competing objectives.

    Although the ODE is integrated over ``t``, the reactor is a plug-flow tube and
    that axis is physically *distance along the tube*: a fluid element entering
    one end has reacted for ``tau`` minutes by the time it leaves. This is why
    ``tau`` is simultaneously "reaction time" and "1 / throughput", and hence why
    the two objectives conflict.

    Parameters
    ----------
    noise_level
        Standard deviation of multiplicative measurement noise on the outlet
        concentrations, **as a percentage of each signal**. Zero (the default)
        makes the oracle deterministic.

        Noise is applied to concentrations *before* the objectives are computed,
        so it propagates through two different non-linear formulas and correlates
        them. It cannot be factored out into a wrapper without changing its
        meaning.
    seed
        Seeds the noise generator. Only relevant when ``noise_level > 0``.
    rtol, atol
        Integration tolerances. Defaults are far tighter than Summit's; pass
        ``SUMMIT_RTOL`` / ``SUMMIT_ATOL`` to reproduce upstream exactly.
    method
        Any ``solve_ivp`` method. ``"RK45"`` matches upstream.
    bounds_tol
        See :class:`~bopt.oracles.base.Oracle`.
    """

    parameters = (
        Parameter("tau", 0.5, 2.0, "min"),
        Parameter("equiv_pldn", 1.0, 5.0, "-"),
        Parameter("conc_dfnb", 0.1, 0.5, "M"),
        Parameter("temperature", 30.0, 120.0, "degC"),
    )
    objectives = (
        Objective("sty", maximize=True, unit="kg/m^3/h"),
        Objective("e_factor", maximize=False, unit="kg waste / kg product"),
    )

    def __init__(
        self,
        *,
        noise_level: float = 0.0,
        seed: int | None = None,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        method: str = "RK45",
        bounds_tol: float = 1e-9,
    ) -> None:
        super().__init__(bounds_tol=bounds_tol)
        if noise_level < 0:
            raise ValueError(f"noise_level must be >= 0, got {noise_level!r}")
        if noise_level > 0 and seed is None:
            warnings.warn(
                "SnarOracle has noise_level > 0 but no seed; this run will not be "
                "reproducible. Pass seed=<int> to fix the noise stream.",
                stacklevel=2,
            )
        self.noise_level = noise_level
        self.seed = seed
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ kinetics

    @staticmethod
    def rate_constants(temperature: float) -> FloatArray:
        """Arrhenius rate constants ``[k_a, k_b, k_c, k_d]`` in 1/(M min).

        ``k = 0.6 * k_ref * exp(-Ea/R * (1/T - 1/T_ref))``, the reference form of
        the Arrhenius equation. A *larger* activation energy means a steeper
        response to temperature, so the ordering of ``_E_ACTIVATION`` is what
        determines how selectivity degrades as the reactor is heated.
        """
        T = temperature + _SUMMIT_KELVIN_OFFSET
        return _RATE_UNIT_CONVERSION * _K_REF * np.exp(
            -_E_ACTIVATION / _GAS_CONSTANT * (1.0 / T - 1.0 / _T_REF_KELVIN)
        )

    @staticmethod
    def _rhs(t: float, C: FloatArray, k: FloatArray, C_initial: FloatArray) -> FloatArray:
        """Right-hand side of the five-species system. All steps are second order.

        ``k`` is precomputed once per integration rather than per call: it depends
        only on temperature, which is constant during a run, so hoisting it is a
        pure speedup that cannot change results.

        .. warning::
           The depletion clamp below **mutates** ``C``, which ``solve_ivp`` may
           pass as its own internal state array. It also makes the derivative
           discontinuous, which adaptive steppers handle poorly, and it breaks
           the conservation laws that otherwise hold exactly. All of this is
           upstream behaviour, preserved because it changes objective values; the
           golden-fixture test is what makes preserving it safe.
        """
        k_a, k_b, k_c, k_d = k

        for i in (SUBSTRATE, AMINE):
            C[i] = 0.0 if C[i] < _DEPLETION_FRACTION * C_initial[i] else C[i]

        consumption = (k_a + k_b) * C[SUBSTRATE] * C[AMINE]
        over_product = k_c * C[AMINE] * C[PRODUCT]
        over_isomer = k_d * C[AMINE] * C[ISOMER]

        r = np.empty(5)
        r[SUBSTRATE] = -consumption
        r[AMINE] = -consumption - over_product - over_isomer
        r[PRODUCT] = k_a * C[SUBSTRATE] * C[AMINE] - over_product
        r[ISOMER] = k_b * C[SUBSTRATE] * C[AMINE] - over_isomer
        r[BIS_ADDUCT] = over_product + over_isomer
        return r

    # --------------------------------------------------------------- integration

    @staticmethod
    def initial_concentrations(equiv_pldn: float, conc_dfnb: float) -> FloatArray:
        """Reactor inlet composition in M. Only the two reactants are present."""
        C = np.zeros(5)
        C[SUBSTRATE] = conc_dfnb
        C[AMINE] = equiv_pldn * conc_dfnb
        return C

    def _integrate(
        self,
        tau: float,
        equiv_pldn: float,
        conc_dfnb: float,
        temperature: float,
        *,
        dense_output: bool = False,
    ):
        C_initial = self.initial_concentrations(equiv_pldn, conc_dfnb)
        k = self.rate_constants(temperature)
        res = solve_ivp(
            self._rhs,
            (0.0, tau),
            C_initial,
            args=(k, C_initial),
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
            dense_output=dense_output,
        )
        if not res.success:
            raise IntegrationError(
                f"solve_ivp failed at tau={tau!r}, equiv_pldn={equiv_pldn!r}, "
                f"conc_dfnb={conc_dfnb!r}, temperature={temperature!r}: {res.message}",
                conditions={
                    "tau": tau,
                    "equiv_pldn": equiv_pldn,
                    "conc_dfnb": conc_dfnb,
                    "temperature": temperature,
                },
                partial=res.y[:, -1] if res.y.size else None,
                status=res.status,
                solver_message=res.message,
            )
        return res

    # --------------------------------------------------------------------- noise

    def _apply_noise(self, C: FloatArray) -> FloatArray:
        """Multiplicative noise on concentrations, as a percentage of each signal.

        Skipped entirely when disabled. That is unobservable rather than merely
        cheap: with ``noise_level == 0`` upstream's draw is multiplied by zero,
        and the generator is used nowhere else.
        """
        if self.noise_level == 0:
            return C
        C = C + C * self._rng.normal(scale=self.noise_level, size=C.size) / 100.0
        C[C < 0] = 0.0  # noise can push a near-zero concentration negative
        return C

    # ---------------------------------------------------------------- objectives

    @staticmethod
    def _objectives(C_final: FloatArray, tau: float) -> tuple[float, float]:
        """Space-time yield and E-factor from the outlet composition.

        **STY** ``= 60 * M_product * C_product / tau`` in kg/m^3/h -- product mass
        per reactor volume per hour. The ``1/tau`` is where the pressure for short
        residence times comes from.

        **E-factor** ``= (mass of everything else) / (mass of product)``, the
        Sheldon waste metric. Volumetric flow cancels exactly between numerator
        and denominator, so the E-factor depends only on outlet *composition*, not
        on how fast the reactor is run. Since ethanol contributes 789 g/L against
        perhaps 60 g/L of dissolved organics, it dominates, and the metric is
        approximately ``3.75 / C_product``.

        Note the solvent term charges the *entire* volumetric flow as ethanol,
        even though the reagent streams are not pure solvent. Upstream computes
        the true ethanol flow and then never uses it; waste is overstated as a
        result. Preserved -- it is part of the published objective.
        """
        q_tot = _REACTOR_VOLUME_ML / tau  # mL/min

        product_mass_flow = 1e-3 * MOLAR_MASS[PRODUCT] * C_final[PRODUCT] * q_tot  # g/min

        sty = 6e4 / 1000 * MOLAR_MASS[PRODUCT] * C_final[PRODUCT] * q_tot / _REACTOR_VOLUME_ML
        sty = max(sty, _STY_FLOOR)

        if np.isclose(C_final[PRODUCT], 0.0):
            e_factor = _E_FACTOR_CEILING
        else:
            other = np.delete(MOLAR_MASS * C_final, PRODUCT).sum()
            waste_mass_flow = q_tot * _ETHANOL_DENSITY_G_PER_ML + 1e-3 * other * q_tot
            e_factor = min(waste_mass_flow / product_mass_flow, _E_FACTOR_CEILING)

        return float(sty), float(e_factor)

    # ------------------------------------------------------------------ contract

    def _evaluate(self, X: FloatArray) -> FloatArray:
        out = np.empty((len(X), self.n_objectives))
        for i, (tau, equiv_pldn, conc_dfnb, temperature) in enumerate(X):
            res = self._integrate(tau, equiv_pldn, conc_dfnb, temperature)
            C_final = self._apply_noise(res.y[:, -1])
            out[i] = self._objectives(C_final, tau)
        return out

    # ---------------------------------------------------------------- inspection

    def trajectory(
        self,
        tau: float,
        equiv_pldn: float,
        conc_dfnb: float,
        temperature: float,
        n_points: int = 200,
    ) -> tuple[FloatArray, FloatArray]:
        """Concentration profile along the reactor, for plotting and intuition.

        Returns ``(t, C)`` with ``t`` of shape ``(n_points,)`` and ``C`` of shape
        ``(5, n_points)``. Watching :data:`PRODUCT` rise and then fall as
        :data:`BIS_ADDUCT` grows is the clearest way to see why a finite residence
        time is optimal. Noise is not applied here --- this is the model, not a
        measurement.
        """
        res = self._integrate(tau, equiv_pldn, conc_dfnb, temperature, dense_output=True)
        t = np.linspace(0.0, tau, n_points)
        return t, res.sol(t)
