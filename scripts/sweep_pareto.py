#!/usr/bin/env python
"""Brute-force the SnAr domain on a regular grid and extract the true Pareto front.

Two purposes:

1. **Ground truth for regret.** Because the oracle is mechanistic and cheap, the
   Pareto front can be computed by exhaustion rather than estimated. Every later
   claim about how well Bayesian Optimization performed is measured against this.
2. **Structure.** ``docs/snar-benchmark.md`` §7 makes predictions from algebra
   alone --- that ``conc_dfnb`` pins to its upper bound, and that the trade-off is
   "fast and dirty vs. slow and clean". A grid says whether that is true.

A grid rather than a quasi-random sample because the second purpose needs
axis-aligned slices. The front it produces is therefore an approximation limited
by grid resolution, which is fine: it is a reference, not a claim of optimality.

Usage
-----
    python -m scripts.sweep_pareto --grid 25 --workers 14

Run with ``-m`` from the repo root, not as ``python scripts/sweep_pareto.py``.
Executing a file directly puts its own directory on ``sys.path``, so ``import
bopt`` fails; ``-m`` puts the working directory there instead. This is the same
flat-layout friction that ``pytest.ini``'s ``pythonpath = .`` settles for tests.

Output lands in ``artifacts/`` and is not committed --- it is fully regenerable
from this script.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from bopt.oracles import SnarOracle

_ORACLE: SnarOracle | None = None


def _init_worker(oracle_kwargs: dict) -> None:
    """Build one oracle per process. Cheaper than pickling it for every chunk."""
    global _ORACLE
    _ORACLE = SnarOracle(**oracle_kwargs)


def _evaluate_chunk(X: np.ndarray) -> np.ndarray:
    assert _ORACLE is not None
    return _ORACLE.evaluate(X)


def pareto_mask(Y: np.ndarray, maximize: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows.

    Flips minimized columns so everything is a maximization, then uses the
    standard 2-D sweep: sort by the first objective descending, walk the list
    once, and keep a point only if it beats the best second objective seen so
    far. O(n log n) rather than the O(n^2) of pairwise comparison, which matters
    at 400k points.

    Ties are handled by the secondary sort key: among points with equal first
    objective, the best second objective comes first and the rest are dominated.
    Exact duplicates do not dominate one another, and only the first survives.
    """
    Z = np.where(maximize, Y, -Y)
    order = np.lexsort((-Z[:, 1], -Z[:, 0]))
    mask = np.zeros(len(Z), dtype=bool)
    best = -np.inf
    for i in order:
        if Z[i, 1] > best:
            mask[i] = True
            best = Z[i, 1]
    return mask


def build_grid(oracle: SnarOracle, n: int) -> np.ndarray:
    axes = [np.linspace(p.low, p.high, n) for p in oracle.parameters]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, oracle.dim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=25, help="points per dimension")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("artifacts/pareto_sweep.npz"))
    args = parser.parse_args()

    oracle = SnarOracle()
    X = build_grid(oracle, args.grid)
    print(f"grid {args.grid}^{oracle.dim} = {len(X):,} points, {args.workers} workers")

    chunks = np.array_split(X, max(1, len(X) // args.chunk))
    started = time.perf_counter()
    with mp.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=({"rtol": oracle.rtol, "atol": oracle.atol},),
    ) as pool:
        results = []
        for i, Y in enumerate(pool.imap(_evaluate_chunk, chunks, chunksize=1), start=1):
            results.append(Y)
            if i % 20 == 0 or i == len(chunks):
                done = sum(len(r) for r in results)
                rate = done / (time.perf_counter() - started)
                print(
                    f"  {done:,}/{len(X):,}  ({done / len(X):5.1%})  "
                    f"{rate:,.0f} eval/s  eta {(len(X) - done) / rate:5.1f}s",
                    end="\r",
                )
    Y = np.vstack(results)
    elapsed = time.perf_counter() - started
    print(f"\nevaluated in {elapsed:.1f}s ({len(X) / elapsed:,.0f} eval/s)")

    maximize = np.array([o.maximize for o in oracle.objectives])
    mask = pareto_mask(Y, maximize)
    print(f"Pareto-optimal: {mask.sum():,} of {len(X):,} ({mask.sum() / len(X):.3%})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        Y=Y,
        pareto=mask,
        grid=args.grid,
        parameter_names=np.array(oracle.parameter_names),
        objective_names=np.array(oracle.objective_names),
    )
    print(f"saved -> {args.out}")

    _report(oracle, X, Y, mask)


def _report(oracle, X, Y, mask) -> None:
    front_X, front_Y = X[mask], Y[mask]
    names = oracle.parameter_names

    print("\n" + "=" * 74)
    print("OBJECTIVE RANGES")
    for j, o in enumerate(oracle.objectives):
        print(
            f"  {o.name:9s} whole grid [{Y[:, j].min():10.2f}, {Y[:, j].max():10.2f}]   "
            f"on front [{front_Y[:, j].min():10.2f}, {front_Y[:, j].max():10.2f}]"
        )

    print("\nPARAMETER VALUES ON THE FRONT")
    print(f"  {'parameter':14s} {'min':>8s} {'max':>8s} {'mean':>8s}   {'at bound?':>22s}")
    for i, name in enumerate(names):
        p = oracle.parameters[i]
        col = front_X[:, i]
        at_hi = np.isclose(col, p.high).mean()
        at_lo = np.isclose(col, p.low).mean()
        note = f"{at_lo:5.1%} at low, {at_hi:5.1%} at high"
        print(f"  {name:14s} {col.min():8.3f} {col.max():8.3f} {col.mean():8.3f}   {note:>22s}")

    print("\nTHE FRONT, SAMPLED FROM HIGH-THROUGHPUT TO LOW-WASTE")
    order = np.argsort(-front_Y[:, 0])
    step = max(1, len(order) // 12)
    print(f"  {'STY':>10s} {'E-factor':>9s}   " + "".join(f"{n:>13s}" for n in names))
    for idx in order[::step]:
        row = "".join(f"{v:13.3f}" for v in front_X[idx])
        print(f"  {front_Y[idx, 0]:10.1f} {front_Y[idx, 1]:9.2f}   {row}")

    print("\nSINGLE-OBJECTIVE OPTIMA")
    for j, o in enumerate(oracle.objectives):
        best = np.argmax(Y[:, j]) if o.maximize else np.argmin(Y[:, j])
        params = ", ".join(f"{n}={v:.3f}" for n, v in zip(names, X[best]))
        print(f"  best {o.name:9s} = {Y[best, j]:10.2f}   at {params}")
        print(f"  {'':14s}   other objective = {Y[best, 1 - j]:.2f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
