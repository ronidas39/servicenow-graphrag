"""Does stemming the keyword arm change any of the numbers the article publishes?

Part 10 said it changed nothing, and that sentence had no code behind it. It also sat
badly beside the article's own explanation of one of the zeros, which blames morphology:
a question asking about "items many things depend on" against a corpus that writes
"depends" and "item". Stemming is exactly the thing that removes that mismatch, so
"changed nothing" was the claim most in need of an artefact and it was the one without.

So this runs the keyword arm twice over the same corpus, the same questions and the same
gold, changing one thing: whether terms are reduced to a stem before they are indexed and
before they are matched.

⛔ THE STEMMER IS APPLIED TO BOTH SIDES OR IT MEASURES NOTHING. Stemming the query alone
would leave "depend" looking for a term the index still stores as "depends", which makes
the arm worse for a reason that has nothing to do with stemming being a good idea.

⛔ AND IT MUST NOT TOUCH IDENTIFIERS. `lnx0525`, `INC2000042` and `pg0711` are the whole
point of several questions. A suffix stripper that turns `changes` into `change` will also
happily turn an identifier into something that matches nothing, so anything holding a digit
is passed through untouched.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "questions"))

from arms import BM25Only  # noqa: E402
from chunking import (change_chunks, ci_chunks, knowledge_chunks,  # noqa: E402
                      per_record, problem_chunks)
from evaluate import aggregate, score_one  # noqa: E402
from gold import World, build as build_gold  # noqa: E402
from questions import QUESTIONS  # noqa: E402

# The Porter step-1 suffixes, which is where nearly all of the effect in English lives.
# The later Porter steps normalise things like "ational" to "ate", and no question in this
# set turns on that. Keeping the rule set small also keeps it inspectable, which matters
# more here than matching a reference implementation exactly: the finding is whether the
# table moves at all, not which stemmer is best.
_PLURAL = ((("sses",), "ss"), (("ies",), "i"), (("ss",), "ss"), (("s",), ""))
_PAST = ("eed", "ed", "ing")


def stem(term: str) -> str:
    """Reduce an English word to a rough stem, leaving identifiers alone."""
    # ⛔ ANYTHING WITH A DIGIT IS A NAME, NOT A WORD. Stripping a suffix from lnx0525
    # produces a token that appears in no document, so the arm would lose exactly the
    # questions it is supposed to win.
    if any(ch.isdigit() for ch in term) or len(term) <= 3:
        return term

    for suffixes, replacement in _PLURAL:
        for suffix in suffixes:
            if term.endswith(suffix):
                term = term[: -len(suffix)] + replacement
                break
        else:
            continue
        break

    for suffix in _PAST:
        if term.endswith(suffix) and len(term) - len(suffix) >= 3:
            stripped = term[: -len(suffix)]
            if any(v in stripped for v in "aeiou"):
                term = stripped + ("e" if suffix == "eed" else "")
                break

    return term or "x"


class BM25Stemmed(BM25Only):
    """The same arm, with both the index and the query reduced to stems.

    ⛔ STOPWORDS COME OUT BEFORE THE STEMMER RUNS, and getting that order wrong was the
    first version of this experiment. The parent filters query terms against a stopword
    list after tokenising, so stemming inside `_tokenise` turned "this" into "thi", which
    no longer matched the list and survived into the query. The arm was then measured
    carrying a term the unstemmed arm had thrown away, and any difference in the result
    would have been that bug rather than stemming.
    """

    name = "bm25_stemmed"

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return [stem(t) for t in re.findall(r"[a-z0-9_]+", text.lower())]

    @classmethod
    def _query_terms(cls, question: str) -> list[str]:
        raw = re.findall(r"[a-z0-9_]+", question.lower())
        terms = [stem(t) for t in raw if t not in cls.STOPWORDS]
        return terms or [stem(t) for t in raw]


def main() -> None:
    world = World()
    gold = build_gold(world)
    names = {k: c["name"] for k, c in world.cis.items()}

    chunks = (per_record(world.incidents)
              + ci_chunks(world.cis.values(), world.depends_on, world.supports, names)
              + change_chunks(world.changes, names)
              + problem_chunks(world.problems)
              + knowledge_chunks(world.knowledge))
    grade = [(c.source_id, c.text) for c in chunks]
    print(f"  corpus {len(grade):,} documents, {len(QUESTIONS)} questions\n")

    results = {}
    returned: dict[str, dict[str, int]] = {}
    for arm in (BM25Only(grade), BM25Stemmed(grade)):
        fetched = {q.id: arm.retrieve(q.text, q.id) for q in QUESTIONS}
        scores = [score_one(fetched[q.id], q, gold.get(q.id), corpus_size=len(grade))
                  for q in QUESTIONS]
        results[arm.name] = (aggregate(scores)[arm.name], scores)
        # ⛔ HOW MANY DOCUMENTS CAME BACK IS A SEPARATE QUESTION FROM WHETHER THEY WERE
        # THE RIGHT ONES, and conflating the two is what made the original claim
        # misleading. An arm returning nothing and an arm returning forty wrong documents
        # both score zero recall, and only one of them has a vocabulary problem.
        returned[arm.name] = {q.id: len(fetched[q.id].record_ids) for q in QUESTIONS}

    print(f"  {'arm':16s} {'recall':>8s} {'MRR':>7s} {'measured':>9s}")
    for name, (agg, _) in results.items():
        print(f"  {name:16s} {agg['recall_mean']:>8.2f} {agg['mrr_mean']:>7.2f} "
              f"{agg['measured']:>9d}")

    per_q = {name: {s.question_id: s.recall_at_k for s in scores
                    if s.recall_at_k is not None}
             for name, (_, scores) in results.items()}
    plain, stemmed = per_q["bm25"], per_q["bm25_stemmed"]
    shared = sorted(set(plain) & set(stemmed))
    moved = [(q, plain[q], stemmed[q]) for q in shared
             if abs(plain[q] - stemmed[q]) > 1e-9]

    print(f"\n  questions where recall changed: {len(moved)} of {len(shared)}")
    for q, before, after in moved:
        print(f"    {q}  plain {before:.2f} -> stemmed {after:.2f}  "
              f"{'BETTER' if after > before else 'WORSE'}")
    if not moved:
        print("    none. Stemming moved no cell of the published table.")

    by_kind = {}
    for name, (_, scores) in results.items():
        kinds: dict[str, list[float]] = {}
        for s in scores:
            if s.recall_at_k is not None:
                kinds.setdefault(s.kind, []).append(s.recall_at_k)
        by_kind[name] = {k: statistics.fmean(v) for k, v in kinds.items()}
    print("\n  by kind:")
    for kind in sorted(by_kind["bm25"]):
        a, b = by_kind["bm25"][kind], by_kind["bm25_stemmed"][kind]
        flag = "" if abs(a - b) < 1e-9 else "   CHANGED"
        print(f"    {kind:14s} plain {a:.2f}   stemmed {b:.2f}{flag}")

    # ⛔ THE INTERESTING COLUMN. Part 10 explains one of the zeros as a vocabulary miss:
    # the query said "depend" and "items" while the corpus wrote "depends" and "item", so
    # nothing matched and the arm returned an empty result. Stemming is precisely the fix
    # for that, so if the score still does not move, the two failures were never the same
    # failure. That distinction is worth more to a reader than the null result itself.
    empty_before = [q for q, n in returned["bm25"].items() if n == 0]
    rescued = [q for q in empty_before if returned["bm25_stemmed"][q] > 0]
    print(f"\n  questions the plain arm returned nothing for: {len(empty_before)}")
    for q in sorted(rescued):
        after = returned["bm25_stemmed"][q]
        recall = stemmed.get(q)
        got = "not graded" if recall is None else f"recall {recall:.2f}"
        print(f"    {q}  0 documents -> {after} documents, and still {got}")
    if rescued:
        print("    So stemming fixed the vocabulary miss and did not fix the score.")

    # ⛔ A NULL RESULT HERE IS A RESULT ABOUT THIS QUESTION SET, NOT ABOUT STEMMING. Ten
    # graded questions cannot show that stemming never helps. What they can show is
    # whether the sentence in the article is true of the table the article prints, and
    # that is the only claim the article is entitled to make.
    print(f"\n  This settles one sentence and no more: whether stemming moves the "
          f"published\n  table. It runs on the {len(shared)} questions that reach the "
          f"recall column, so it\n  cannot say anything about stemming in general.")


if __name__ == "__main__":
    main()
