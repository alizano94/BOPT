# Finding: the SnAr benchmark's objectives are nearly independent

**Status:** confirmed against two independent implementations.
**Date:** 2026-08-11.
**Reproduce:** `python -m scripts.sweep_pareto --grid 25`

The SnAr benchmark is presented as a bi-objective problem — maximize space-time
yield, minimize E-factor. Brute-forcing the domain shows that in practice it is
close to single-objective: **the E-factor varies by only 15% across the entire
Pareto front, while STY varies four-fold.**

See [`snar-benchmark.md`](snar-benchmark.md) for the problem and
[`pareto-fronts.md`](pareto-fronts.md) for the concepts used here.

---

## 1. What was run

A regular 25⁴ grid over the four decision variables — 390,625 points — evaluated
at converged integration tolerances (`rtol=1e-10`), 242 s on 14 cores. The
non-dominated set was extracted exactly. Because the oracle is mechanistic and
cheap, this is the true front to grid resolution, not an estimate.

## 2. The result

| Objective | Range on the front | As a fraction of the full grid range |
|---|---|---|
| STY | 2,856 → 11,560 (**4.05×**) | 75.5% |
| E-factor | 8.45 → 9.76 (**1.15×**) | **0.16%** |

52 of 390,625 points are non-dominated (0.013%). In normalized objective space
the front is very nearly a horizontal line: you can quadruple throughput for 15%
more waste, and no setting trades meaningfully in the other direction.

### Why

The E-factor is dominated by solvent. From the derivation in
[`snar-benchmark.md` §6](snar-benchmark.md#6-objectives), volumetric flow cancels
and ethanol contributes 789 g/L against tens of g/L of organics, so

```
E  ~=  3.75 / C_product
```

The best achievable E-factor at each substrate concentration tracks that estimate
to within ~12% across the whole range:

| `conc_dfnb` | best E achieved | `789 / (210.21 · c)` |
|---|---|---|
| 0.1 M | 41.97 | 37.53 |
| 0.3 M | 14.03 | 12.51 |
| 0.5 M | 8.45 | 7.51 |

So the E-factor is effectively a function of **one input** — and that input pins
to its upper bound on **100%** of the front. With `conc_dfnb` fixed at 0.5 M, the
E-factor has almost nowhere left to move.

This is the same objection raised against the iron-ore flotation dataset during
problem selection: two objectives only produce a front if they genuinely conflict.
It should have been predicted from the `E ≈ 3.75 / C_product` algebra rather than
discovered by brute force.

## 3. Corroboration

Summit's repository ships `pareto_front_snar.csv` — 10,000 evaluations from an
NSGA-II run against the reference implementation. Extracting its non-dominated
set independently reproduces the structure:

| | STY span | E-factor span | `conc_dfnb` at upper bound |
|---|---|---|---|
| This port, 25⁴ grid | 4.05× | 1.15× | 100% |
| Summit's own data, 10k NSGA-II | 3.91× | 1.14× | 100% |

Two implementations, two search strategies, two different E-factor definitions
(see §5), one conclusion. **The degeneracy is a property of the benchmark, not of
this port.**

## 4. The mechanism is not the one predicted

An earlier draft of [`snar-benchmark.md` §7](snar-benchmark.md) predicted "fast
and dirty vs. slow and clean" — high throughput bought with heat, at the cost of
selectivity. **That is backwards.** At the high-throughput end of the front
(τ=0.5, equiv=5, conc=0.5):

| Temperature | STY | E-factor |
|---|---|---|
| 30 °C | **11,560** | **9.76** |
| 60 °C | 9,132 | 12.59 |
| 90 °C | 6,496 | 18.06 |
| 120 °C | 3,556 | 33.73 |

Heat is **monotonically bad on both objectives** here. The maximum-STY point on
the entire grid sits at the *coldest* available temperature.

The reason: **reagent excess and temperature are substitutes for reaction rate,
and only one of them costs selectivity.** At 5 equivalents the amine concentration
is 2.5 M, which drives near-complete conversion in 30 seconds even at 30 °C.
Heat then adds nothing but over-reaction — which destroys product (lowering STY)
*and* creates waste (raising the E-factor). The original prediction assumed short
residence times force you to buy rate with heat; they do not, because
concentration buys it more cheaply.

Temperature is still a rate-vs-selectivity dial in the *low-equivalents* regime,
which is where the rest of the front lives. It is simply not the dominant axis.

## 5. A refuted hypothesis, and an archaeological one

**Refuted.** The preserved defect in the E-factor (charging the entire flow as
ethanol rather than the true ethanol fraction — quirk #2) looked like a candidate
cause, since the true ethanol fraction varies strongly with equivalents.
Correcting it makes matters *worse*: the front collapses to a **single point**
(τ=0.5, equiv=5, T=30, E=2.83), because high equivalents then displace solvent and
become good for both objectives. The bug creates what little trade-off exists.

**Confirmed.** Our port initially disagreed with Summit's CSV by ~10% on the
E-factor. The residuals showed `ours − theirs ≈ −1.00` on every row, suggesting
the reference counted the product as its own waste. Summit's git history confirms
it: versions before
[`e10d139e`](https://github.com/sustainable-processes/summit/commit/e10d139e1)
(2020-06-20) summed over `range(5)` without excluding the product. So
`E_old = E_current + 1`, exactly.

Because `x → x + 1` is strictly monotone, **Pareto dominance is invariant** and
the non-dominated set is identical under both definitions. The formula change
shifts the front but changes no conclusion — which is why §3's comparison holds
despite the two datasets using different formulas.

## 6. What this means for the project

The bi-objective framing is kept, and **the degeneracy is reported as a result
rather than engineered away**. The alternatives were rejected deliberately:

- *Reframe as constrained single-objective* — defensible, and closer to what the
  data supports, but it discards the finding instead of presenting it.
- *Restrict `conc_dfnb` to force a trade-off* — would manufacture the desired
  shape by altering a published benchmark.

Consequences to keep in view when the optimization machinery lands:

1. **Hypervolume improvement will be almost entirely STY.** Expect the E-factor
   contribution to be near-invisible. That is the correct behaviour on this
   problem, not a bug in the acquisition function.
2. ~~**A single-objective STY baseline should perform nearly as well as qNEHVI.**~~
   **Revised 2026-08-11, after the reference point was fixed.** This prediction was
   written assuming the declared-bounds reference point, under which the E-factor
   contributes ~0.3% of hypervolume. Under the chosen nadir+10% reference the
   picture is different: the single best-STY point achieves only **9.67%** of the
   true front's hypervolume (1,248 of 12,912), because it sits at the *worst*
   E-factor on the front. Recovering the whole front is worth roughly **10×** more
   than finding the best throughput point alone.

   So a genuine gap between qNEHVI and single-objective STY optimization *is*
   expected here, and the baseline is a real comparison rather than a foregone
   conclusion. The degeneracy is a property of the problem; how much it matters to
   the *metric* is a property of the reference point, and these two must not be
   conflated — as they were in the original version of this prediction.
3. **The reference point matters more than usual.** With a front this flat,
   hypervolume is dominated by the reference point's distance along the E-factor
   axis. It must be fixed in advance and stated — which is done below.

### The reference point

Chosen as **the true front's nadir, pushed out by 10% of the front's span on each
axis**:

| | STY | E-factor |
|---|---|---|
| ideal (best on the front) | 11560.2759 | 8.4520 |
| nadir (worst on the front) | 2855.8984 | 9.7556 |
| front span | 8704.3775 | 1.3037 |
| **reference point** (physical) | **1985.4606** | **9.8860** |

In all-maximize space, which is where the optimization code works:
**`(1985.4606, -9.8860)`**.

Why this rule rather than the declared objective bounds `(0, 500)`:

| Reference point | Fraction of the STY axis the front uses | ... of the E-factor axis |
|---|---|---|
| declared bounds `(0, 500)` | 75.3% | **0.3%** |
| nadir + 10% of span | 90.9% | **90.9%** |

The rule *balances the axes by construction* — `span / (1.1 · span) = 1/1.1` on
every axis regardless of scale — which is exactly what makes the E-factor visible
to hypervolume at all. Under the declared bounds the E-factor would contribute
about 0.3% of hypervolume and qNEHVI would be single-objective STY optimization in
all but name.

**The cost, stated plainly: this reference point encodes knowledge of where the
front is.** That is standard practice in multi-objective benchmarking, since
comparing hypervolume across methods and seeds requires a fixed reference, but it
means hypervolume here is a *search and comparison signal*, not a physical
quantity, and it is not comparable to hypervolume values computed by anyone using
a different reference. Numbers above come from the 25⁴ sweep
(`scripts/sweep_pareto.py`, 2026-08-11) and are hard-coded rather than recomputed
at runtime, so that a re-run of the sweep cannot silently move the metric.

## 7. Caveats

- The front is exact only to grid resolution (25 points per dimension). The
  qualitative conclusion is far too large to be a resolution artifact, but the
  precise endpoints are not exact.
- Corroboration in §3 uses Summit's stored NSGA-II output, whose values carry a
  median 2e-3 and worst-case 70% integration error (see
  `data/golden/README.md`). It is strong evidence about *structure*, weak
  evidence about *numbers*.
- The port has not been compared against current Summit `main` running under
  modern scipy. Every difference observed so far is explained — the legacy
  formula, and integration tolerance — but that check remains undone.
