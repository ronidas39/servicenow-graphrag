"""Run every arm against every question and write the table Part 10 publishes.

One entry point, so the numbers in the article come from a command a reader can run
rather than from something assembled by hand in a terminal. Everything it depends on is
printed: the corpus size, the frozen question hash, the embedding model, the token budget.

⛔ THE CORPUS IS BUILT ONCE AND SHARED BY EVERY ARM. Giving one arm a different chunking
or a different budget makes the comparison a measurement of context sizes wearing the
names of retrieval strategies, which is the exact failure Part 10 exists to avoid.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "questions"))

from arms import BM25Only, CONTEXT_TOKEN_BUDGET, Hybrid, NoRetrieval, VectorOnly  # noqa: E402
from chunking import (change_chunks, ci_chunks, knowledge_chunks,  # noqa: E402
                      per_record, problem_chunks)
from evaluate import (ENUMERATION_CEILING, aggregate, report, save,  # noqa: E402
                      score_one)
from gold import World, build as build_gold  # noqa: E402
from questions import QUESTIONS, frozen_hash  # noqa: E402


def build_corpus(world: World):
    """Every record type, because what is in the index decides what is answerable.

    ⛔ THIS LESSON LANDED THREE TIMES. The corpus held only incidents, then incidents and
    items, and each time a whole class of question scored zero for every arm. Not because
    retrieval failed, but because the answer was not in the index at all.
    """
    names = {k: c["name"] for k, c in world.cis.items()}
    return (per_record(world.incidents)
            + ci_chunks(world.cis.values(), world.depends_on, world.supports, names)
            + change_chunks(world.changes, names)
            + problem_chunks(world.problems)
            + knowledge_chunks(world.knowledge))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-vector", action="store_true",
                    help="skip the arms that need embeddings")
    args = ap.parse_args()

    world = World()
    chunks = build_corpus(world)
    gold = build_gold(world)

    # Two keyings of one corpus, in one order. chunk_id is unique and is what gets
    # embedded; source_id is what a gold set names and is what gets graded.
    to_embed = [(c.chunk_id, c.text) for c in chunks]
    to_grade = [(c.source_id, c.text) for c in chunks]

    from collections import Counter
    print(f"  corpus: {len(chunks):,} documents")
    for kind, n in Counter(c.source_kind for c in chunks).most_common():
        print(f"    {kind:22s} {n:>7,}")
    hard = [g for g in gold.values() if g.gradable]
    print(f"  questions: {len(QUESTIONS)}, {len(hard)} with a mechanical answer")
    print(f"  frozen hash: {frozen_hash()}")
    print(f"  context budget: {CONTEXT_TOKEN_BUDGET:,} tokens per arm\n")

    arms = [BM25Only(to_grade), NoRetrieval(to_grade)]

    # ⛔ EVERYTHING NEEDED TO REPRODUCE THE NUMBER, RECORDED WITH THE NUMBER. The corpus
    # fingerprint is here because the chunk text silently changed once, between processes,
    # and nothing in the results said so. A published figure with no fingerprint beside it
    # cannot be checked against anything.
    import datetime
    import json as _json
    from embed import corpus_fingerprint
    manifest = _json.loads((HERE.parent / "dataset" / "manifest.json").read_text())
    hard = [g for g in gold.values() if g.gradable]
    measured = [g for g in hard
                if not g.expect_empty and g.supporting
                and len(g.supporting) <= ENUMERATION_CEILING]
    meta = {
        "run_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "corpus_documents": len(chunks),
        "corpus_fingerprint": corpus_fingerprint(to_embed),
        "dataset_seed": manifest.get("seed"),
        "dataset_built_at": manifest.get("built_at"),
        "frozen_question_hash": frozen_hash(),
        "questions_total": len(QUESTIONS),
        "questions_gradable": len(hard),
        "questions_feeding_recall": len(measured),
        "budget_tokens": CONTEXT_TOKEN_BUDGET,
        "enumeration_ceiling": ENUMERATION_CEILING,
        "no_retrieval_seed": NoRetrieval.SEED,
        "chunking": "per_record incidents, plus every other record type",
    }

    if not args.no_vector:
        from embed import MODEL, build as build_vectors, corpus_fingerprint
        print("  embeddings:")
        vectors = build_vectors(to_embed)
        fingerprint = corpus_fingerprint(to_embed)
        vector = VectorOnly(to_grade, vectors, fingerprint=fingerprint)
        bm25 = arms[0]
        arms += [vector, Hybrid(bm25, vector, chunks=to_grade)]
        meta["embedding_model"] = MODEL
        meta["embedding_dimensions"] = int(vectors.shape[1])
        print()

    # ⛔ PRINT AS IT GOES. The vector arm embeds every question, and the hybrid embeds
    # them again, so a full run is minutes of silence with buffered output. A run you
    # cannot watch is one you cannot tell from a hang.
    scores = []
    for arm in arms:
        t0 = time.time()
        for q in QUESTIONS:
            scores.append(score_one(arm.retrieve(q.text, q.id), q, gold.get(q.id),
                                    corpus_size=len(chunks)))
        print(f"    {arm.name:14s} {len(QUESTIONS)} questions in "
              f"{time.time() - t0:6.1f}s", flush=True)

    agg = aggregate(scores)
    print(report(agg))
    path = save(scores, agg, meta)
    print(f"\n  written to {path}")


if __name__ == "__main__":
    main()
