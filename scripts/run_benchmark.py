#!/usr/bin/env python
"""Run every strategy at every seed, in parallel, and save the records.

Usage
-----
    python -m scripts.run_benchmark --seeds 20 --workers 14

Run with ``-m`` from the repo root; see ``scripts/sweep_pareto.py`` for why.

Parallelism
-----------
Each worker pins ``torch.set_num_threads(1)``. Measured in isolation this costs a
single run about 2% --- GP fits at these sizes are far too small for threading to
help --- and it avoids oversubscribing 16 cores with 14 processes x 8 threads.

It also improves reproducibility: threaded BLAS reductions are not bit
deterministic, and hypervolume was observed to differ in the 4th decimal between
thread counts on an otherwise identical run.

.. warning::
   **Parallel speedup is nowhere near linear.** A single qNEHVI run at budget 64
   takes ~230 s alone, which suggested ~8 min for 80 runs on 14 workers. The
   actual sweep took far longer: 14 concurrent processes contend for memory
   bandwidth and cache, and per-run wall time inflates several-fold. Measuring one
   run in isolation and multiplying is not a valid estimate for this workload ---
   budget roughly 45-60 min, and measure contention directly if it matters.

Output
------
One ``.npz`` per (strategy, seed) under ``artifacts/runs/``. Nothing here is
committed --- it is all regenerable from this script.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

DEFAULT_STRATEGIES = ("qnehvi", "sty_only", "sobol", "grid")


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")


def _run_one(task):
    """Execute one (strategy, seed) job. Returns a record or an error marker.

    Failures are caught rather than allowed to kill the pool: a numerically
    awkward seed should cost one cell of the results table, not the whole sweep.
    """
    strategy, seed, config = task
    from bopt.bo import run

    started = time.perf_counter()
    try:
        record = run(strategy, seed=seed, **config)
        return {"ok": True, "record": record, "seconds": time.perf_counter() - started}
    except Exception:
        return {
            "ok": False,
            "strategy": strategy,
            "seed": seed,
            "error": traceback.format_exc(limit=5),
            "seconds": time.perf_counter() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--n-init", type=int, default=16)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--out", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()

    config = dict(budget=args.budget, n_init=args.n_init, q=args.q, noise_level=args.noise)
    tasks = [(s, seed, config) for s in args.strategies for seed in range(args.seeds)]

    print(
        f"{len(tasks)} runs = {len(args.strategies)} strategies x {args.seeds} seeds\n"
        f"budget={args.budget} (n_init={args.n_init}, q={args.q}) noise={args.noise}%\n"
        f"{args.workers} workers, 1 torch thread each\n"
    )
    args.out.mkdir(parents=True, exist_ok=True)

    # Longest jobs first, so the tail of the sweep is not one slow run finishing
    # alone while every other worker idles.
    order = {"qnehvi": 0, "sty_only": 1, "sobol": 2, "grid": 3}
    tasks.sort(key=lambda t: order.get(t[0], 9))

    results, failures = [], []
    started = time.perf_counter()
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        for i, outcome in enumerate(pool.imap_unordered(_run_one, tasks), start=1):
            elapsed = time.perf_counter() - started
            if outcome["ok"]:
                record = outcome["record"]
                record.save(args.out / f"{record.strategy}__seed{record.seed:02d}.npz")
                results.append(record)
                label = f"{record.strategy:9s} seed {record.seed:2d}  hv {record.final_hv_fraction:.4f}"
            else:
                failures.append(outcome)
                label = f"{outcome['strategy']:9s} seed {outcome['seed']:2d}  FAILED"
            eta = elapsed / i * (len(tasks) - i)
            # flush explicitly: when stdout is redirected to a file Python block
            # buffers it, so a long sweep's log stays empty until it finishes ---
            # which is exactly when you no longer need the progress report.
            print(
                f"  [{i:3d}/{len(tasks)}] {label}  ({outcome['seconds']:5.1f}s)  eta {eta / 60:4.1f}m",
                flush=True,
            )

    print(f"\ncompleted in {(time.perf_counter() - started) / 60:.1f} min -> {args.out}")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures:
            print(f"  {f['strategy']} seed {f['seed']}:\n{f['error']}")

    _summarize(results, args.budget)


def _summarize(records, budget: int) -> None:
    if not records:
        return
    print("\n" + "=" * 78)
    print(f"{'strategy':10s} {'n':>4s} {'unused':>7s} {'HV fraction: mean':>18s} {'sd':>8s} "
          f"{'min':>8s} {'max':>8s}")
    print("-" * 78)
    by_strategy: dict[str, list] = {}
    for r in records:
        by_strategy.setdefault(r.strategy, []).append(r)
    for name, group in sorted(by_strategy.items(), key=lambda kv: -np.mean([r.final_hv_fraction for r in kv[1]])):
        hv = np.array([r.final_hv_fraction for r in group])
        used = int(np.mean([r.n_evaluated for r in group]))
        unused = int(np.mean([r.unused_budget for r in group]))
        print(
            f"{name:10s} {used:4d} {unused:7d} {hv.mean():18.4f} {hv.std():8.4f} "
            f"{hv.min():8.4f} {hv.max():8.4f}"
        )
    print("=" * 78)
    print("HV fraction is hypervolume relative to the brute-forced front under the")
    print("fixed reference point; values slightly above 1.0 are legitimate, since a")
    print("continuous optimizer can land between the reference grid's nodes.")


if __name__ == "__main__":
    main()
