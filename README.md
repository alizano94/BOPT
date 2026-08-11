# bopt

Bayesian Optimization with Gaussian Process surrogates, driven against a mechanistic
chemistry model standing in for a real plant.

A study project: the goal is a BO loop whose every component is understood and defensible,
not a wrapper around someone else's solver.

---

## The premise

A process plant wants to develop a new product. Each experiment is slow and expensive, so
the question is not "what is the optimum" but **"which experiment should I run next?"**

That is the problem Bayesian Optimization solves: fit a cheap probabilistic surrogate to
what you have measured so far, then use its *uncertainty* — not just its predictions — to
choose the next experiment. This repository simulates that loop end to end, using a
mechanistic reaction model as a stand-in for the plant so that ground truth is knowable and
progress can be measured exactly.

## The problem being optimized

A **nucleophilic aromatic substitution (S_NAr)** in a continuous plug-flow reactor, taken
from [Summit's benchmark suite](https://github.com/sustainable-processes/summit) with
kinetics from [Hone et al., *React. Chem. Eng.* 2017](https://doi.org/10.1039/C6RE00109B).

| Knob | Range | Unit |
|---|---|---|
| residence time | 0.5 – 2 | min |
| pyrrolidine equivalents | 1 – 5 | — |
| substrate concentration | 0.1 – 0.5 | M |
| temperature | 30 – 120 | °C |

| Objective | Direction | Meaning |
|---|---|---|
| space-time yield | maximize | product mass per reactor volume per hour |
| E-factor | minimize | kg waste per kg product |

The two objectives genuinely conflict, and the conflict is **derivable rather than
asserted**: the desired reaction has the lowest activation energy of the four competing
pathways, so heating the reactor accelerates everything but accelerates the byproduct
routes faster. Throughput is bought with selectivity. The result is a Pareto front —
fast-and-dirty at one end, slow-and-clean at the other — rather than a single best recipe.

## Why this problem, and not a dataset

A BO oracle must answer queries at **arbitrary points** the acquisition function proposes.
It is therefore doing interpolation across an entire design space, not prediction on a
held-out test set. That rules out most large industrial datasets: a well-controlled plant
generates enormous volumes of data inside a narrow operating band, so an oracle fitted to it
extrapolates — confidently and wrongly — the moment BO probes outside the hull.

A mechanistic model sidesteps this entirely:

- **unlimited, cheap, exact queries** at any point in the domain
- **no emulator error** — the model *is* the ground truth
- the **true Pareto front is brute-forceable**, so regret can be measured rather than estimated
- **noise is a knob**, off by default, rather than an unknown baked into the data
- roughly 120 lines of readable ODE, defensible line by line

The reasoning behind rejecting the alternatives (including a 737k-row plant dataset) is
recorded in [`docs/snar-benchmark.md`](docs/snar-benchmark.md).

## Layout

```
bopt/
├── bopt/
│   ├── oracles/      problem definitions — the world, in physical units
│   ├── adapters/     translation to optimizer conventions (torch, unit cube, sign flips)
│   ├── bo/           surrogates, acquisition functions, the optimization loop
│   └── viz/          plotting
├── tests/
├── scripts/          reproducible analyses (sweeps, golden-fixture generation)
├── data/             fixtures a clone cannot regenerate — committed deliberately
├── artifacts/        regenerable output — not committed
└── docs/             an Obsidian vault; see below
```

Dependencies run one way only: `oracles` imports nothing from the rest of the package and
knows nothing about Bayesian Optimization. Anything that would change a reported result —
a hypervolume reference point, a normalization scheme — lives downstream, never in the
problem definition.

## Setup

Requires Python ≥ 3.11 (BoTorch's floor).

```bash
python -m venv ~/envs/bopt
~/envs/bopt/bin/pip install numpy scipy pytest        # oracle + tests
~/envs/bopt/bin/pip install torch botorch gpytorch    # optimization layer (later)
```

No install step for this package: it uses a flat layout and is imported from the repo root.
`pytest.ini` sets `pythonpath = .` so the test suite can find it.

```bash
~/envs/bopt/bin/python -m pytest
```

> **Note on Summit.** The benchmark's source cannot be installed alongside BoTorch —
> Summit requires Python `<3.11`, BoTorch requires `≥3.11`. The reaction model is therefore
> **ported**, not imported, and validated against real Summit output captured once in a
> disposable Python 3.10 environment and committed as a fixture. See
> [`docs/snar-benchmark.md`](docs/snar-benchmark.md) §9.

## Status

| | |
|---|---|
| ✅ | Problem selected and characterized on paper |
| ✅ | `Oracle` base class — bounds, tolerances, validation contracts (28 tests) |
| ⬜ | `SnarOracle` — the ported reaction model |
| ⬜ | Validation: mass balance, integration convergence, diff against Summit |
| ⬜ | Brute-force Pareto front |
| ⬜ | BoTorch adapter — unit cube, sign conventions |
| ⬜ | GP surrogates and acquisition functions |
| ⬜ | BO vs. random vs. grid, over many seeds |

## Documentation

- [`docs/snar-benchmark.md`](docs/snar-benchmark.md) — the chemistry, the kinetics, the
  objectives derived term by term, known defects in the reference source, and why the
  alternatives were rejected
- [`docs/pareto-fronts.md`](docs/pareto-fronts.md) — dominance, Pareto sets and fronts, and
  how hypervolume turns a front into something an acquisition function can optimize
