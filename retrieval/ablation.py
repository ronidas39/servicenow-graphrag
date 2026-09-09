"""Does writing the graph into the text let plain search answer a multi hop question?

This is the experiment the article turns on, and it is the one whose answer could embarrass
the thesis. `graph_denormalised` inlines everything a traversal would find: what the item
runs on, what depends on it, who owns it, what changed near it. If similarity search then
answers a question it was never supposed to answer, the honest finding is not "graphs beat
vectors". It is "the graph was needed to BUILD the index, not to query it", which is a more
useful sentence than the one this article set out to write.

⛔ ONE VARIABLE. Same questions, same gold, same arm, same token budget. The only thing that
changes is how the incident chunks were written. Anything else moving makes the result a
comparison between two things at once.

⛔ AND THE COST IS REPORTED NEXT TO THE BENEFIT. Denormalising buys reach with tokens, and a
strategy that wins by spending more context has not won, it has been given more room.
Measured on this dataset it makes an incident chunk 1.4x larger, 7.9M tokens against 11.1M
across the corpus. The token column is not decoration.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "questions"))

from arms import BM25Only, VectorOnly  # noqa: E402
from chunking import (change_chunks, ci_chunks, graph_denormalised,  # noqa: E402
                      knowledge_chunks, per_record, problem_chunks)
from evaluate import ENUMERATION_CEILING, aggregate, score_one  # noqa: E402
from gold import World, build as build_gold  # noqa: E402
from questions import MULTIHOP, QUESTIONS  # noqa: E402


def corpus(world: World, incident_chunks):
    """The same everything-else, so only the incident strategy differs."""
    names = {k: c["name"] for k, c in world.cis.items()}
    return (incident_chunks
            + ci_chunks(world.cis.values(), world.depends_on, world.supports, names)
            + change_chunks(world.changes, names)
            + problem_chunks(world.problems)
            + knowledge_chunks(world.knowledge))


def main() -> None:
    world = World()
    gold = build_gold(world)
    names = {k: c["name"] for k, c in world.cis.items()}
    ci_by_key = dict(world.cis)
    changes_by_ci: dict[str, list[dict]] = {}
    for change in world.changes:
        if change.get("ci_key"):
            changes_by_ci.setdefault(change["ci_key"], []).append(change)

    strategies = {
        "per_record": per_record(world.incidents),
        "graph_denormalised": graph_denormalised(
            world.incidents, ci_by_key, world.depends_on, world.supports, changes_by_ci),
    }

    print("  what each strategy costs, before any accuracy number:")
    print(f"    {'strategy':22s} {'chunks':>8s} {'mean tokens':>12s} {'total tokens':>14s}")
    sizes = {}
    for name, chunks in strategies.items():
        toks = [c.tokens for c in chunks]
        sizes[name] = statistics.fmean(toks)
        print(f"    {name:22s} {len(chunks):>8,} {statistics.fmean(toks):>12,.0f} "
              f"{sum(toks):>14,}")
    growth = sizes["graph_denormalised"] / sizes["per_record"]
    print(f"    denormalising makes an incident chunk {growth:.1f}x larger\n")

    # ⛔ THE VECTOR ARM IS THE POINT. BM25 needs shared words, so it can barely benefit
    # from inlined neighbours and its result settles nothing. The hypothesis is that an
    # embedding of a chunk describing its neighbourhood sits closer to a question about
    # that neighbourhood even with no shared words. That needs both corpora embedded, and
    # both are, at 78 minutes each.
    from embed import build as build_vectors, corpus_fingerprint

    results = {}
    for name, chunks in strategies.items():
        everything = corpus(world, chunks)
        grade = [(c.source_id, c.text) for c in everything]
        embed_pairs = [(c.chunk_id, c.text) for c in everything]
        vectors = build_vectors(embed_pairs, quiet=True)

        for arm_name, arm in (
            ("bm25", BM25Only(grade)),
            ("vector", VectorOnly(grade, vectors,
                                  fingerprint=corpus_fingerprint(embed_pairs))),
        ):
            scores = [score_one(arm.retrieve(q.text, q.id), q, gold.get(q.id),
                                corpus_size=len(everything))
                      for q in QUESTIONS]
            results[f"{name} / {arm_name}"] = (aggregate(scores)[arm_name], scores)

    print(f"  {'strategy / arm':32s} {'recall':>8s} {'MRR':>7s} {'tokens used':>12s}")
    for name, (agg, _) in results.items():
        print(f"    {name:32s} {agg['recall_mean']:>8.2f} {agg['mrr_mean']:>7.2f} "
              f"{agg['tokens_mean']:>12,.0f}")

    # ⛔ THE MULTI HOP QUESTIONS ARE THE WHOLE POINT. A change in the overall mean could
    # come from anywhere; the claim is specifically about questions needing a chain.
    print(f"\n  multi hop questions only, which is what the experiment is about:")
    for name, (_, scores) in results.items():
        hop = [s.recall_at_k for s in scores
               if s.kind == MULTIHOP and s.recall_at_k is not None]
        got = f"{statistics.fmean(hop):.2f} over {len(hop)}" if hop else "none measured"
        print(f"    {name:32s} {got}")

    per_q = {}
    for name, (_, scores) in results.items():
        per_q[name] = {s.question_id: s.recall_at_k for s in scores
                       if s.recall_at_k is not None}
    shared = sorted(set(per_q["per_record / vector"]) & set(per_q["graph_denormalised / vector"]))
    moved = [(q, per_q["per_record / vector"][q], per_q["graph_denormalised / vector"][q])
             for q in shared
             if abs(per_q["per_record / vector"][q] - per_q["graph_denormalised / vector"][q]) > 1e-9]
    print(f"\n  questions where the answer changed: {len(moved)} of {len(shared)}")
    for q, a, b in moved:
        print(f"    {q}  per_record {a:.2f} -> denormalised {b:.2f}"
              f"  {'BETTER' if b > a else 'WORSE'}")
    if not moved:
        print("    none. On this question set, writing the graph into the text changed "
              "nothing that BM25 could use.")

    # ⛔ THIS RESULT DOES NOT SETTLE THE QUESTION, AND SAYING SO IS PART OF REPORTING IT.
    # BM25 matches terms. Denormalising can only help it when the question happens to use
    # the same words as the inlined neighbourhood, which is a narrow case. The hypothesis
    # is about SIMILARITY search: an embedding of a chunk that describes its neighbours
    # should sit closer to a question about that neighbourhood even with no shared words.
    # That version of the experiment needs the vector arm, and until it runs the honest
    # claim is "no effect on keyword search", not "no effect".
    print("\n  The per-question comparison above is the VECTOR arm, which is where the "
          "hypothesis\n  lives. BM25 needs shared words, so it can barely benefit from "
          "inlined neighbours.")

    # ⛔ THE CONFOUND, STATED RATHER THAN HIDDEN. Both corpora contain ci_chunks, and a CI
    # chunk already writes the sentences this experiment inlines: "X depends on Y" and "if
    # X stops working, Y stops working too". So the control is not graph-free, and "no
    # effect" is equally well explained by "the graph was already in the control".
    #
    # The obvious fix is a third condition with ci_chunks removed from both arms. It
    # cannot be run. Measured: dropping ci_chunks makes the gold unreachable for five of
    # the measured questions, INCLUDING BOTH multi hop ones, because those gold sets name
    # configuration items and there would be no configuration item documents to return.
    # A condition that cannot score the questions the experiment is about is not a
    # condition.
    print("\n  ⛔ WHAT THIS EXPERIMENT CANNOT TELL YOU. Both corpora contain one document"
          "\n     per configuration item, and those already say what depends on what. So"
          "\n     'inlining the graph changed nothing' and 'the graph was already in the"
          "\n     control' predict the same result, and this design cannot separate them."
          "\n"
          "\n     Removing those documents would be the clean test. It cannot be run: it"
          "\n     makes the gold unreachable for five measured questions, including both"
          "\n     multi hop ones, because their answers ARE configuration items.")

    mrr_plain = results["per_record / vector"][0]["mrr_mean"]
    mrr_rich = results["graph_denormalised / vector"][0]["mrr_mean"]
    if mrr_rich < mrr_plain - 1e-9:
        print(f"\n  ⛔ Reciprocal rank FELL, {mrr_plain:.2f} to {mrr_rich:.2f}, while "
              f"recall held.\n     The supporting records are still found and they are "
              f"found LOWER down. Added\n     context dilutes the sentence that made the "
              f"chunk match, and at a fixed token\n     budget a lower rank is a record "
              f"that may not fit in the prompt at all.")


if __name__ == "__main__":
    main()
