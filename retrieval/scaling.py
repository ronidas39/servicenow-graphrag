"""How much of a retrieval result survives the corpus getting bigger.

Almost every RAG tutorial demonstrates on a few thousand chunks. This one has 82,296, and
the gap between those two numbers turned out to matter more than the choice of retriever.

The method: hold the question and its correct answers fixed, and grow the haystack around
them. Every subset contains all the gold records, so nothing is being hidden. The only
thing that changes is how many other documents the retriever has to rank them against.

⛔ SEVERAL SEEDS, BECAUSE ONE IS AN ANECDOTE. The first run of this showed keyword search
going from every correct record to none between two thousand documents and ten thousand,
which is a big enough cliff to be a fluke of which distractors that particular sample drew.
Repeating it with different samples is the difference between a finding and a coincidence.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "questions"))

from arms import BM25Only, CONTEXT_TOKEN_BUDGET, fit_to_budget  # noqa: E402
from embed import MODEL, build as build_vectors, embed_texts  # noqa: E402
from gold import World, build as build_gold  # noqa: E402
from questions import QUESTIONS  # noqa: E402
from run import build_corpus  # noqa: E402

SIZES = [2_000, 5_000, 10_000, 20_000, 40_000, 82_296]
SEEDS = [20260909, 20260910, 20260911]

# The questions with a small enough gold set to measure recall on, one per kind that has
# one, so the answer is not about a single question's quirks.
LOOK_AT = ["Q04", "Q01", "Q12"]


def main() -> None:
    # ⛔ THIS WRITES ITS NUMBERS TO A FILE AS WELL AS TO THE TERMINAL. Section 112's
    # figure used to be drawn from numbers copied out of a terminal by hand, and when
    # the embedding model changed the terminal moved and the figure did not. Anything a
    # figure reads has to be written by the thing that measured it.
    captured: dict[str, dict] = {}
    world = World()
    gold = build_gold(world)
    chunks = build_corpus(world)
    vectors = build_vectors([(c.chunk_id, c.text) for c in chunks], quiet=True)
    ids = [c.source_id for c in chunks]
    texts = [c.text for c in chunks]
    tokens = [c.tokens for c in chunks]

    for qid in LOOK_AT:
        question = next(q for q in QUESTIONS if q.id == qid)
        want = gold[qid].supporting
        gold_pos = [i for i, x in enumerate(ids) if x in want]
        qv = embed_texts([question.text])[0]

        print(f"\n  {qid} ({question.kind}, predicted to favour {question.expect_favours})"
              f", {len(want)} correct records")
        print(f"    {question.text[:72]}")
        print(f"\n    {'corpus':>8s}  {'bm25 recall':>22s}  {'vector recall':>22s}")
        captured[qid] = {"kind": question.kind, "gold": len(want),
                         "bm25": [], "vector": []}

        for size in SIZES:
            bm_runs, vec_runs = [], []
            for seed in SEEDS:
                if size >= len(chunks):
                    idx = np.arange(len(chunks))
                else:
                    rng = np.random.default_rng(seed)
                    pool = np.setdiff1d(np.arange(len(chunks)), np.array(gold_pos))
                    extra = rng.choice(pool, size - len(gold_pos), replace=False)
                    idx = np.sort(np.concatenate([np.array(gold_pos), extra]))
                sub_ids = [ids[i] for i in idx]

                # ⛔ THE SAME TOKEN BUDGET AS THE REAL RUN. Counting hits in a top 40 that
                # would never fit in the prompt measures a ranking nobody sees.
                order = np.argsort(-(vectors[idx] @ qv))[:40]
                ranked = [(sub_ids[i], texts[idx[i]]) for i in order]
                kept, _, _ = fit_to_budget(ranked, CONTEXT_TOKEN_BUDGET)
                vec_runs.append(len(set(kept) & want) / len(want))

                bm = BM25Only([(sub_ids[j], texts[idx[j]]) for j in range(len(idx))])
                got = bm.retrieve(question.text, qid).record_ids
                bm_runs.append(len(set(got) & want) / len(want))

                if size >= len(chunks):
                    break

            def show(runs):
                if len(runs) == 1:
                    return f"{runs[0]:.2f}"
                return f"{statistics.fmean(runs):.2f} ± {statistics.pstdev(runs):.2f}"

            print(f"    {size:>8,}  {show(bm_runs):>22s}  {show(vec_runs):>22s}")
            captured[qid]["bm25"].append(round(statistics.fmean(bm_runs), 4))
            captured[qid]["vector"].append(round(statistics.fmean(vec_runs), 4))

    print("\n  Every subset contains all the correct records. The only thing that grows")
    print("  is the number of other documents they have to be ranked against.")

    out = HERE.parent / "results" / "captures" / "scaling.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    payload = {
        "what": "recall against corpus size, one question held fixed, three seeds",
        "source": f"retrieval/scaling.py, embedding model {MODEL}",
        "note": ("the crossover published before this rerun came from nomic-embed-text "
                 "and does not reproduce under the model this article ships"),
        "sizes": SIZES,
        "questions": {
            qid: {**vals,
                  "vector_nomic_published":
                      existing.get("questions", {}).get(qid, {})
                      .get("vector_nomic_published")}
            for qid, vals in captured.items()
        },
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
