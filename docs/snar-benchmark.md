# The SnAr Benchmark — Chemistry, Kinetics, and Objectives

Reference documentation for the ground-truth ("real world") model chosen for this
Bayesian Optimization project.

**Source:** [`summit/benchmarks/snar.py`](https://github.com/sustainable-processes/summit/blob/main/summit/benchmarks/snar.py)
(Summit v0.8.9, 184 lines, depends only on `numpy` and `scipy.integrate.solve_ivp`)

**Kinetics from:** C. A. Hone et al., *React. Chem. Eng.*, 2017, **2**, 103–108.
[DOI: 10.1039/C6RE00109B](https://doi.org/10.1039/C6RE00109B)

---

## Why this problem

A Bayesian Optimization oracle must be queryable at **arbitrary points** the acquisition
function proposes — it is doing interpolation across a whole design space, not prediction
on a held-out test set. That rules out most large industrial datasets, which are
observational logs from processes deliberately held near setpoint and therefore cover only
a narrow tube of the input space.

This benchmark avoids the problem entirely by being **mechanistic**: a system of ODEs
integrated per query. Consequences:

| Property | Value here |
|---|---|
| Queries | Unlimited, cheap, exact |
| Ground truth | Known — the model *is* the truth, no emulator error |
| True Pareto front | Brute-forceable on a 4-D grid, so regret is exact |
| Noise | A controllable knob (`noise_level`), off by default |
| Readability | ~120 lines of ODE you can derive and defend line by line |
| Trade-off | Emerges from the kinetics, not asserted by fiat |

---

## 1. The reaction

**S_NAr** — nucleophilic aromatic substitution.

An aromatic ring carrying a strongly electron-withdrawing nitro group and two fluorines is
attacked by an amine (pyrrolidine). The amine displaces a fluoride. The nitro group is what
makes the reaction work: it pulls electron density out of the ring, stabilising the
negatively-charged Meisenheimer intermediate.

The five species are never named in the source, only weighed. The molecular weights
identify them:

| `M[i]` | g/mol | Species | Verification |
|---|---|---|---|
| 0 | 159.09 | 2,4-difluoronitrobenzene (substrate) | C₆H₃F₂NO₂ = 159.09 ✓ |
| 1 | 71.12 | pyrrolidine (nucleophile) | C₄H₉N = 71.12 ✓ |
| 2 | 210.21 | mono-substituted product — **desired** | 159.09 + 71.12 − HF(20.01) = 210.20 ✓ |
| 3 | 210.21 | the other regioisomer — **unwanted** | same formula, substitution at the other F |
| 4 | 261.33 | bis-adduct (both F replaced) — **waste** | 210.21 + 71.12 − 20.01 = 261.32 ✓ |

Species 2 and 3 are **isomers**: identical mass, different structure. There are two
fluorines — one *ortho* to the nitro group, one *para*. Substitution at one gives product,
at the other gives waste. This is a **regioselectivity** problem: no amount of "more
reaction" fixes it, because selectivity is set by the *ratio* of two rate constants.

> ⚠️ The source docstring says `2,4 dinitrofluorobenzene`. That compound is 186.10 g/mol.
> The mass in the code (159.09) says otherwise — it is 2,4-**difluoronitro**benzene.
> Typo in Summit's source.

---

## 2. The reaction network

```
              k_a  (Ea 33.3 kJ/mol)
      ┌──────────────────────────────► [2] DESIRED ──┐
      │                                              │ k_c (Ea 38.9)
 [0] + [1]                                           ├──────────► [4] bis-adduct
      │                                              │ k_d (Ea 44.8)
      └──────────────────────────────► [3] isomer  ──┘
              k_b  (Ea 35.3 kJ/mol)
```

Two mechanisms operate simultaneously, and they are different in kind:

- **Parallel competition** (`k_a` vs `k_b`) — the substrate branches to either the desired
  product or the wrong isomer. This sets *selectivity*.
- **Series over-reaction** (`k_c`, `k_d`) — the product is *itself* a substrate for a
  second attack. **The product is not stable in the reactor.** Leave it too long and it is
  consumed.

The series step is what makes this an optimization problem rather than a
"run it to completion" problem. There is a finite-time optimum; both too little and too
much reaction are bad.

### Rate laws

All four steps are second-order (bimolecular — one collision each):

```
d[0]/dt = −(k_a + k_b)·C₀C₁
d[1]/dt = −(k_a + k_b)·C₀C₁ − k_c·C₁C₂ − k_d·C₁C₃     ← amine consumed by all four steps
d[2]/dt = +k_a·C₀C₁ − k_c·C₁C₂                          ← made, then destroyed
d[3]/dt = +k_b·C₀C₁ − k_d·C₁C₃
d[4]/dt = +k_c·C₁C₂ + k_d·C₁C₃                          ← accumulates, never leaves
```

`solve_ivp` integrates this from `t = 0` to `t = τ`. The state at τ is the reactor outlet.

---

## 3. Temperature dependence

```
k = 0.6 · k_ref · exp( −Ea/R · (1/T − 1/T_ref) )        T_ref = 90 °C
```

Arrhenius in reference form. The `0.6` converts units from 10⁻² M⁻¹s⁻¹ to M⁻¹min⁻¹
(×10⁻² then ×60).

**Rate constants at the 90 °C reference** (after the 0.6 conversion):

| Constant | Ea (kJ/mol) | k at 90 °C (M⁻¹min⁻¹) | Role |
|---|---|---|---|
| k_a | 33.3 | **34.7** | → desired product |
| k_b | 35.3 | 1.62 | → wrong isomer |
| k_c | 38.9 | 0.519 | desired → bis-adduct |
| k_d | 44.8 | 0.978 | isomer → bis-adduct |

The desired path is ~21× faster than the wrong isomer. The chemistry is intrinsically
well-behaved.

### The key insight about temperature

In the Arrhenius exponential, a **larger Ea means a steeper response to heat**. The desired
reaction has the **lowest** Ea of the four (33.3). Therefore heating accelerates everything,
but accelerates the three parasitic reactions **faster**.

Working the exponents over the full 30 → 120 °C range:

| Quantity | Change over 30→120 °C | Interpretation |
|---|---|---|
| k_a (absolute) | ~**20×** | heat buys enormous rate |
| k_c / k_a | ↑ **1.7×** | over-reaction gains ground |
| k_d / k_a | ↑ **2.8×** | over-reaction gains ground |
| k_a / k_b | ↓ **17%** | regioselectivity degrades least of the three |

**Temperature is a rate-vs-selectivity dial, and the selectivity loss is concentrated in
the over-reaction, not the isomer split.**

---

## 4. The reactor

Continuous flow, fixed volume **V = 5 mL**, with `q_tot = V/τ`.

Residence time τ is not set directly — you set a **pump speed**, and τ follows:

| τ (min) | q_tot (mL/min) |
|---|---|
| 0.5 | 10.0 |
| 2.0 | 2.5 |

Short τ = high throughput, but less time to react.

---

## 5. Decision variables

| Variable | Meaning | Bounds | Units |
|---|---|---|---|
| `tau` | residence time | [0.5, 2] | min |
| `equiv_pldn` | equivalents of pyrrolidine relative to substrate | [1.0, 5] | — |
| `conc_dfnb` | substrate concentration at reactor inlet (after mixing) | [0.1, 0.5] | M |
| `temperature` | reactor temperature | [30, 120] | °C |

Initial conditions: `C₀ = conc_dfnb`, `C₁ = equiv_pldn × conc_dfnb`, `C₂ = C₃ = C₄ = 0`.

---

## 6. Objectives

| Objective | Direction | Declared bounds |
|---|---|---|
| `sty` — space-time yield (kg m⁻³ h⁻¹) | **maximize** | [0, 13000] |
| `e_factor` — kg waste per kg product | **minimize** | [0, 500] |

### Space-time yield

Working the units through source line 137:

$$\text{STY} = 60 \cdot M_2 \cdot C_2 / \tau \approx 12{,}613 \cdot C_2 / \tau$$

(C₂ in M, τ in min.) Product mass per reactor volume per hour — a pure productivity metric.

> **Sanity check:** at C₂ = 0.5 M and τ = 0.5 min this gives 12,613, and the code declares
> `sty` bounds of `[0, 13000]`. The derivation is confirmed.

### E-factor

The Sheldon green-chemistry waste metric. Source line 146:

$$E = \frac{q_{tot}\rho_{eth} + 10^{-3}\sum_{i \neq 2} M_i C_i q_{tot}}{10^{-3} M_2 C_2 q_{tot}}$$

**`q_tot` cancels completely**, giving:

$$E = \frac{789 + \sum_{i\neq 2} M_i C_i}{M_2 C_2}$$

(ρ_eth = 0.789 g/mL = 789 g/L.)

Two consequences:

1. **The E-factor does not depend on flow rate at all** — only on the *composition* at the
   outlet.
2. **The solvent dominates.** Ethanol contributes 789 g/L while dissolved organics
   contribute perhaps 60 g/L, so E ≈ 3.75 / C₂:

   | C₂ (M) | E-factor (approx) |
   |---|---|
   | 0.10 | ~38 |
   | 0.45 | ~8 |

---

## 7. Where the trade-off actually lives

> **This section was rewritten after the brute-force sweep.** Its original version made
> two predictions from the algebra above. One was confirmed; the other was **backwards**.
> Both are kept below, because being able to see which kind of reasoning held up is more
> useful than a clean-looking narrative. Full analysis in
> [`findings-degeneracy.md`](findings-degeneracy.md).

### Confirmed: `conc_dfnb` pins to its upper bound

Both objectives want product concentration C₂ as high as possible, so `conc_dfnb` was
predicted to sit at 0.5 M for both. It does — on **100% of the front**, in this port and
in Summit's own data independently. On the Pareto set the problem is effectively
three-dimensional.

### Refuted: "fast and dirty vs. slow and clean"

The original claim was that throughput is bought with heat and paid for in selectivity.
The sweep says otherwise. At the high-throughput end (τ=0.5, equiv=5, conc=0.5):

| Temperature | STY | E-factor |
|---|---|---|
| 30 °C | **11,560** | **9.76** |
| 60 °C | 9,132 | 12.59 |
| 90 °C | 6,496 | 18.06 |
| 120 °C | 3,556 | 33.73 |

Heat is **monotonically bad on both objectives** in this regime, and the maximum-STY point
on the entire grid sits at the *coldest* temperature available.

The flaw in the original reasoning was assuming that a short residence time forces you to
buy rate with temperature. It does not. **Reagent excess and temperature are substitutes
for rate, and only one of them costs selectivity.** At 5 equivalents the amine sits at
2.5 M, enough for near-complete conversion in 30 seconds even at 30 °C; heat then adds
nothing but over-reaction, which destroys product (↓ STY) *and* generates waste (↑ E).

Temperature remains a genuine rate-vs-selectivity dial at *low* equivalents, which is where
the rest of the front lives. It is simply not the dominant axis.

### And the trade-off is weak

| Objective | Range on the front | Fraction of the full grid range |
|---|---|---|
| STY | 4.05× | 75.5% |
| E-factor | **1.15×** | **0.16%** |

Because `conc_dfnb` pins at its bound and `E ≈ 3.75 / C_product`, the E-factor has almost
nowhere left to move. This benchmark is close to single-objective in practice — a
conclusion reproduced independently from Summit's own published data. See
[`findings-degeneracy.md`](findings-degeneracy.md).

> See [`pareto-fronts.md`](pareto-fronts.md) for what a Pareto front is, why scalarizing
> these two objectives would be backwards, and how hypervolume turns the front into a
> search signal BoTorch can optimize.

---

## 8. Source quirks to handle when porting

| # | Issue | Location | Impact |
|---|---|---|---|
| 1 | `T_ref = 90 + 273.71` — absolute zero is 273.15 | lines 155–156 | 0.56 K off, applied consistently to both terms, so effect is tiny — but wrong |
| 2 | `q_1`, `q_2`, `q_eth` computed and **never used** | lines 119–121 | E-factor charges `q_tot · ρ_eth`, i.e. treats the *entire* flow as pure ethanol even though reagent streams are not. Waste is somewhat overstated. |
| 3 | `np.random.default_rng()` is **unseeded** | line 52 | Noise is irreproducible as written. Must be fixed for reproducible experiments. |
| 4 | Noise is **multiplicative on concentrations**, applied *before* STY/E-factor | lines 128–133 | Propagates non-linearly into both objectives and **correlates them** (both read the same perturbed C's). More realistic than output noise — and more interesting. `scale = noise_level/100`, i.e. a percentage. |
| 5 | STY floored at `1e-6`; E-factor capped at `1e3`, hard-set to `1e3` when no product forms | lines 138–148 | Creates flat plateaus in dead regions of the space. Will affect GP fitting — know about these before being confused by them. |
| 6 | Docstring names the wrong compound | docstring | Cosmetic; see §1 note. |

---

## 9. Environment constraint

Neither of the two relevant research packages can run alongside BoTorch:

| Package | Requires | Blocking pins |
|---|---|---|
| `summit` 0.8.9 (Feb 2023) | Python **>=3.8, <3.11** | `torch<2.0`, `numpy<2.0`, `GPy`, `gpyopt` (archived), `numba<0.56` |
| `olymp` 0.0.1b0 (Oct 2020) | — | **TensorFlow 1.15**, which itself caps at Python 3.7 |
| `botorch` 0.18.1 | Python **>=3.11** | — |

Project environment is Python 3.12.3. The conflict is structural, not a version bump.

**Therefore:** the benchmark is **ported**, not imported. Since `snar.py` depends only on
`numpy` and `scipy` — everything else is Summit's `Domain`/`Experiment` plumbing — the ODE
and objective calculations transfer directly. This is also the better outcome for a project
whose goal is understanding every line: a dependency on a dead 2020 package that hides the
oracle behind TensorFlow 1.15 would work against that.

---

## Appendix — related datasets

- **Olympus** ships an experimental `snar` dataset (66 points, `e_factor` objective) on the
  same chemistry — usable as a reality check against the mechanistic model.
- All 43 Olympus datasets are plain `config.json` + `data.csv` + `description.txt` files in
  the repo and can be fetched with `curl`, no install required.
- Structural note on Olympus: its **continuous** parameter spaces are all
  **single-objective**, and its **multi-objective** datasets (`lnp3`, `dye_lasers`,
  `redoxmers`) all have **categorical** parameter spaces. It contains no continuous
  multi-objective formulation problem.
- **`lnp3`** (lipid nanoparticles → drug loading / encapsulation efficiency / particle
  diameter) is 4×3×4×4×4 = 768 rows — a **complete factorial enumeration**, so its oracle
  is a lookup table with zero interpolation error and an exactly-known Pareto front. Held
  as a candidate second chapter.
