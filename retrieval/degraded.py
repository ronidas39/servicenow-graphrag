"""Damage the CMDB on purpose, and measure what the answers do.

Part 0 tells a reader not to build this if their relationship data is known to be wrong,
and gives 17.89% stale as the number measured here. That advice is worthless without a
second number: **how wrong do the answers get?** Without it "far higher" is a feeling.

So this removes a fraction of the dependency edges and re-asks the question the article
opens with. Nothing here involves a retriever or a model. It measures the sensitivity of
the ANSWER to the quality of the data underneath it, which is a property of the estate and
not of anything built on top.

⛔ EDGES ARE REMOVED, NOT REWIRED. A real stale CMDB mostly fails by omission: somebody
decommissioned a thing and nobody removed the row, or added a dependency and nobody
recorded it. Randomly repointing edges would model a different and rarer failure, and it
would make the damage look worse than it is.

⛔ SEEDED, so the numbers in the article can be reproduced exactly.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import pathlib
import random
import statistics
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "questions"))

from gold import IMPACT, World, _load  # noqa: E402

# ⛔ MANY SEEDS, BECAUSE ONE IS AN ANECDOTE. The first version of this ran a single seed
# and published 13 silently-wrong answers at 5% damage. Across 25 seeds the mean is closer
# to 18 and the range runs 10 to 25, so the published figure sat near the bottom of its own
# distribution. Worse, the article drew its closing argument from the count FALLING between
# 20% and 30% damage, and that fall happens in only about half of seeds. The claim is true
# at 50%, which is the row the first version did not print.
SEEDS = list(range(20260909, 20260909 + 25))
FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]

# ⛔ EVERY ELIGIBLE SERVICE, NOT THE BUSIEST FORTY. Taking the top 40 by blast radius gave
# a sample with a mean radius of 12.6 against 6.7 for the eligible population, so the
# published curve described the most connected services in the estate and called them
# "40 production services". The whole thing runs in seconds; there is no reason to sample.
MIN_BLAST = 3


def blast(supports: dict[str, set[str]], key: str, hops: int = 4) -> set[str]:
    seen, frontier = set(), {key}
    for _ in range(hops):
        nxt: set[str] = set()
        for k in frontier:
            nxt |= supports.get(k, set())
        nxt -= seen | {key}
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen


def supports_from(rows) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["type_name"] in IMPACT:
            out[r["child_key"]].add(r["parent_key"])
    return out


def main() -> None:
    world = World()
    rows = [r for r in _load("relationships") if r["type_name"] in IMPACT]
    truth = supports_from(rows)

    candidates = sorted(
        k for k, c in world.cis.items()
        if k.startswith("svc-") and c["environment"] == "prd"
        and len(blast(truth, k)) >= MIN_BLAST)
    if len(candidates) < 5:
        raise SystemExit("not enough items with a blast radius to measure anything")

    baseline = {k: blast(truth, k) for k in candidates}
    mean_true = statistics.fmean(len(v) for v in baseline.values())
    print(f"  every production service with a blast radius of {MIN_BLAST} or more: "
          f"{len(candidates)}")
    print(f"  mean blast radius {mean_true:.1f} items")
    print(f"  {len(rows):,} impact edges, {len(SEEDS)} seeds\n")

    print(f"  {'edges removed':>14s} {'recall':>8s} {'exactly right':>11s} "
          f"{'short':>11s} {'empty':>7s}  {'short, range':>9s}")
    print("  " + "-" * 72)

    series: dict[float, list[int]] = {}
    for fraction in FRACTIONS:
        per_seed = {"unchanged": [], "wrong": [], "empty": [], "recall": []}
        for seed in SEEDS:
            rng = random.Random(seed)
            kept = [r for r in rows if rng.random() >= fraction]
            damaged = supports_from(kept)

            unchanged = wrong = empty = 0
            for key in candidates:
                got = blast(damaged, key)
                want = baseline[key]
                if got == want:
                    unchanged += 1
                elif not got:
                    # ⛔ THE COLUMN THAT WAS MISSING. The prose leaned on this category
                    # and the table had no place for it, so the rows did not sum to the
                    # sample size and a reader could not see where the difference went.
                    empty += 1
                else:
                    # ⛔ THE DANGEROUS CASE. Not an empty answer, which looks broken and
                    # makes somebody check. A SHORTER answer, which looks exactly like a
                    # correct one and gets acted on at two in the morning.
                    wrong += 1
            assert unchanged + wrong + empty == len(candidates), (
                "every service must land in exactly one column or the table lies")
            per_seed["unchanged"].append(unchanged)
            per_seed["wrong"].append(wrong)
            per_seed["empty"].append(empty)
            per_seed["recall"].append(statistics.fmean(
                len(blast(damaged, k) & baseline[k]) / len(baseline[k])
                for k in candidates))
            if fraction == 0.0:
                break            # undamaged is identical for every seed

        def pct(key, _p=per_seed):
            return statistics.fmean(_p[key]) / len(candidates) * 100

        series[fraction] = per_seed["wrong"]
        w = per_seed["wrong"]
        rng_txt = f"{min(w)}-{max(w)}" if len(w) > 1 else str(w[0])
        print(f"  {fraction:>13.0%} {statistics.fmean(per_seed['recall']):>8.2f} "
              f"{pct('unchanged'):>12.0f}% {pct('wrong'):>10.0f}% "
              f"{pct('empty'):>6.0f}%  {rng_txt:>12s}")

    # ⛔ THE CONCLUSION IS NOW CONDITIONAL ON THE NUMBERS JUST COMPUTED. The first version
    # printed "rises then falls, and that is not noise" unconditionally, so the script
    # could not report that it had not happened. It happened between 20% and 30% in about
    # half of seeds, which IS noise. It holds robustly by 50%, which the first version did
    # not print.
    peak = max(FRACTIONS, key=lambda f: statistics.fmean(series[f]))
    worst, last = statistics.fmean(series[peak]), statistics.fmean(series[FRACTIONS[-1]])
    fell = sum(1 for i in range(len(SEEDS))
               if series[FRACTIONS[-1]][i] < series[peak][i]) if len(SEEDS) > 1 else 0
    print(f"\n  short-and-plausible answers peak at {peak:.0%} damage "
          f"({worst:.1f} of {len(candidates)}) and fall to {last:.1f} by "
          f"{FRACTIONS[-1]:.0%}.")
    if fell >= 0.8 * len(SEEDS):
        print(f"  That fall holds in {fell} of {len(SEEDS)} seeds, so it is not noise.")
        print("  Mild damage produces short, plausible answers. Severe damage produces")
        print("  empty ones, and an empty answer makes somebody check. A lightly stale")
        print("  CMDB is more dangerous than an obviously broken one.")
    else:
        print(f"  ⛔ That fall holds in only {fell} of {len(SEEDS)} seeds. On this data it "
              f"is not\n     a reliable finding and must not be reported as one.")

    print("\n  Reading this table:")
    print(f"    Percentages are of all {len(candidates)} services, averaged over "
          f"{len(SEEDS)} seeds.")
    print("    'short' is the dangerous column: a SHORTER answer does not look like a")
    print("    failure. An empty result makes somebody check. A plausible short one")
    print("    gets acted on at 02:10.")


if __name__ == "__main__":
    main()
