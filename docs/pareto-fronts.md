# Pareto Fronts — Concept Notes

Reference notes on multi-objective optimization concepts used in this project.
Worked against the SnAr benchmark; see [`snar-benchmark.md`](snar-benchmark.md) for the
problem itself.

---

## 1. The problem it solves

With one objective, "best" is unambiguous — sort by the number, take the top. With two
**conflicting** objectives, that breaks down.

Three hypothetical SnAr runs:

| Run | STY (↑ better) | E-factor (↓ better) |
|---|---|---|
| **A** | 2,000 | 12 |
| **B** | 9,000 | 31 |
| **C** | 1,500 | 35 |

*(Illustrative numbers, not computed output.)*

**Is A better than B?** A wastes far less; B produces 4.5× more. There is **no answer**
without knowing how much waste you would trade for throughput — and that is a business
decision, not a chemistry one. A plant with cheap solvent disposal picks B; one facing
effluent regulations picks A.

**C is different.** A beats C on *both* axes — more product *and* less waste. Nobody would
ever choose C. That asymmetry is the entire idea.

---

## 2. Dominance

> **A dominates B** if A is at least as good as B on *every* objective, and strictly better
> on *at least one*.

In the table above, A dominates C (2000 > 1500 ✓, 12 < 35 ✓). A does **not** dominate B,
and B does not dominate A — they are **incomparable**.

| Term | Definition |
|---|---|
| **Pareto optimal** / non-dominated | A point that nothing else dominates |
| **Pareto set** | The set of such points **in design space** (τ, equivalents, concentration, temperature) |
| **Pareto front** | Their image **in objective space** (the STY / E-factor pairs) |

Dominance is a **partial order**, not a total one. That is the formal reason "best" stops
being a single thing.

---

## 3. The picture

For SnAr — maximize STY (rightward), minimize E-factor (downward) — the ideal corner is
bottom-right:

```
  E-factor
  (↓ better)
    40 │   ○         ○                              ● D  ← fast & dirty
       │        ○              ○              ●
    30 │              ○                  ● C
       │      ○                    ●
    20 │            ○         ● B          ○
       │                 ●
    10 │           ● A                 ○
       │   ← nothing exists down here
     0 └──────────────────────────────────────────────
       0     2k     4k     6k     8k    10k    12k
                    STY (→ better)

       ● = Pareto optimal      ○ = dominated
```

The front is the **lower-right boundary** of everything achievable. Dominated points sit up
and to the left of it — worse on at least one axis, better on none.

### Two things to read off this

**The front is a curve, not a point.** Its output is not one recipe but a **menu**. The
shape of the curve answers "what does one more unit of throughput cost me in waste?" That
slope is the **exchange rate**, and it is usually not constant — near A, buying throughput
is cheap; near D, you pay a lot of waste for very little extra product. *Knowing where the
curve bends is often the most actionable result of the whole study.*

**The empty region matters.** Nothing exists below-left of the front. That region is
physically unreachable — the kinetics forbid it. A Pareto front is as much a statement
about what is **impossible** as about what is optimal.

---

## 4. Why this is the right frame for SnAr

From [`snar-benchmark.md` §7](snar-benchmark.md#7-where-the-trade-off-actually-lives):
STY wants τ = 0.5 min with hard pushes on temperature and excess amine; E-factor wants long
τ, gentle temperature, near-stoichiometric amine. These genuinely fight.

Scalarizing them into something like `0.7·STY − 0.3·E` would force you to invent the
weights **before** you know the exchange rate — which is backwards, since the exchange rate
is exactly what the study is meant to discover.

> **Corollary — how to spot a fake multi-objective problem.** Two objectives only produce a
> front if they actually conflict. The iron-ore flotation dataset was rejected during
> problem selection partly because `%silica` is essentially the complement of `%iron`: the
> two "objectives" are one objective, and the front collapses to a single point.

---

## 5. Connecting to BoTorch — hypervolume

**The practical problem:** Expected Improvement needs a scalar to improve *on*. With a
front, "current best" is an entire set.

**The standard fix — hypervolume.** Choose a deliberately bad **reference point** (e.g.
STY = 0, E-factor = 500, the declared bound), then measure the area dominated by your front
and bounded by that reference point:

```
  E-factor
    500 ┤─────────────────────────── reference point ──┐
        │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
        │░░░░░░░░░░░ dominated region ░░░░░░░░░░░░░░░░░│
     30 │░░░░░░░░░░░░░░░░░░░░░░●───────────────────────┘
        │░░░░░░░░░░░░░░●───────┘
     10 │░░░░░░░░●─────┘
      0 └────────────────────────────────────────
```

That area is **one number** measuring the quality of an entire set. It rises when you find
a genuinely new trade-off and stays flat when you find a dominated point.

This makes the BO question well-posed again: *which experiment maximizes the expected
increase in this number?*

| Acquisition | What it is |
|---|---|
| **EHVI** | Expected Hypervolume Improvement |
| **qEHVI** | Batch version (propose `q` experiments at once) |
| **qNEHVI** | Batch + **noise-tolerant** — the one to actually call in BoTorch |

> ⚠️ **Hypervolume is a search signal, not a result.** It depends on the reference point you
> choose and on how the axes are scaled, so it is not a physically meaningful quantity on
> its own. Report the front; use hypervolume to find it.

---

## 6. Two common misreadings

1. **The front need not be convex, or even connected.** It can have gaps and kinks. Given
   the output clipping documented in
   [`snar-benchmark.md` §8, quirk #5](snar-benchmark.md#8-source-quirks-to-handle-when-porting)
   (STY floored at 1e-6, E-factor capped at 1e3), this front may well have flat regions.

2. **Pareto optimality says nothing about whether a point is a *good* choice.** It only says
   nothing else beats it outright. A point can be Pareto optimal and still be operationally
   useless — the front includes the extremes, and the extremes are often absurd in practice
   (maximum throughput at ruinous waste, or near-zero waste at throughput too low to be
   worth running the plant).
