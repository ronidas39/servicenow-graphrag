"""The seven arms of the comparison, and the two controls that let it fail.

An audit's first objection was that the comparison could not fail: it ran five retrievers
that were all variations on vector search plus a graph, with nothing to check them
against. Two controls are added here for that reason.

  no_retrieval   the records handed to the model with no retrieval at all
  bm25           keyword search on its own, no vectors, no graph
  vector         similarity only, the naive baseline
  hybrid         vector and keyword together
  vector_cypher  similarity finds the entry point, then the graph is walked
  hybrid_cypher  both indexes, then the graph is walked
  text2cypher    the model writes the query

⛔ IF A CONTROL WINS, THAT IS THE FINDING AND IT GETS REPORTED. On an estate this size,
putting the whole neighbourhood in a long context may beat everything, and an article that
cannot say so is not measuring anything.

⛔ FAIRNESS IS A TOKEN BUDGET, NOT A ROW COUNT. "Same top k" is meaningless when one arm
returns a 90 token chunk and another returns a subgraph. Every arm is truncated to the
same budget, that budget is declared, and the tokens actually spent are reported next to
the accuracy. Otherwise the comparison is between context sizes wearing the names of
retrieval strategies.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

# One budget for every arm. Declared here so it appears in the article as a number
# rather than as a claim that things were kept fair.
CONTEXT_TOKEN_BUDGET = 3000


@dataclass
class Retrieved:
    """What an arm handed to the model, and what it cost to get there."""

    arm: str
    question_id: str
    # Record ids, in rank order. Graded against the gold set.
    record_ids: list[str] = field(default_factory=list)
    # The text actually placed in the prompt, after truncation to the budget.
    context: str = ""
    tokens: int = 0
    latency_ms: float = 0.0
    # Set when an arm could not run: a written query that would not parse, an index that
    # does not exist. Counted and reported rather than quietly scored as zero.
    error: str = ""
    # For the written query arm, so its reliability can be reported separately from its
    # accuracy.
    generated_query: str = ""
    retries: int = 0


class Arm(Protocol):
    name: str

    def retrieve(self, question: str, question_id: str) -> Retrieved: ...


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def fit_to_budget(pieces: list[tuple[str, str]],
                  budget: int = CONTEXT_TOKEN_BUDGET) -> tuple[list[str], str, int]:
    """Take ranked (id, text) pairs and keep as many as the budget allows.

    ⛔ TRUNCATION IS PART OF THE MEASUREMENT AND IT HAPPENS HERE, FOR EVERY ARM. If each
    retriever trimmed its own results the comparison would be between trimming policies.
    Whole pieces only: half a ticket is worse than no ticket, because a model will answer
    from the half it can see and sound just as certain.
    """
    kept_ids: list[str] = []
    kept_text: list[str] = []
    used = 0
    for rid, text in pieces:
        cost = approx_tokens(text)
        if used + cost > budget:
            continue
        kept_ids.append(rid)
        kept_text.append(text)
        used += cost
    return kept_ids, "\n\n---\n\n".join(kept_text), used


# ── control one: no retrieval at all ───────────────────────────────────────────────

class NoRetrieval:
    """Everything the budget allows, chosen without reference to the question.

    The control that most GraphRAG articles leave out, and the one that decides whether
    retrieval was needed at all. On a small estate a long context can simply hold the
    answer, and if this wins the article has to say so.
    """

    name = "no_retrieval"

    # ⛔ A SEEDED RANDOM SAMPLE, NOT THE FIRST N. Taking the corpus in order gave every
    # document near the front a free pass into the context, and a gold record that
    # happened to sit at position 14 was "found" by a control that does no retrieval at
    # all. It scored 0.17 on the semantic questions and beat every real retriever, which
    # looked like a finding and was an accident of ordering.
    #
    # A control has to be unrelated to the question AND unrelated to where a record
    # happens to live in the file. Seeded so the article's numbers reproduce.
    SEED = 20260909

    def __init__(self, corpus: list[tuple[str, str]]):
        # Deliberately not ranked. Taking the "most relevant" would make this a
        # retriever, which is the thing it exists to be a control against.
        import random
        rng = random.Random(self.SEED)
        self.corpus = list(corpus)
        rng.shuffle(self.corpus)

    def retrieve(self, question: str, question_id: str) -> Retrieved:
        t0 = time.perf_counter()
        ids, context, tokens = fit_to_budget(self.corpus)
        return Retrieved(self.name, question_id, ids, context, tokens,
                         (time.perf_counter() - t0) * 1000)


# ── control two: keyword search on its own ─────────────────────────────────────────

class BM25Only:
    """Okapi BM25 over the same chunks, with no vectors and no graph.

    Keyword search recovers a surprising amount of what people credit to embeddings,
    which is why it has to run alone rather than only inside a hybrid. It should win the
    questions about pasted stack traces and ticket numbers, and it should lose the ones
    phrased in different words from the text.
    """

    name = "bm25"

    def __init__(self, chunks: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = [c[0] for c in chunks]
        self.texts = [c[1] for c in chunks]
        self.tokenised = [self._tokenise(t) for t in self.texts]
        self.avg_len = sum(len(t) for t in self.tokenised) / max(1, len(self.tokenised))
        self.df: dict[str, int] = {}
        for toks in self.tokenised:
            for term in set(toks):
                self.df[term] = self.df.get(term, 0) + 1
        self.n = len(self.tokenised)

    # ⛔ STOPWORDS BEAT AN EXACT IDENTIFIER MATCH, MEASURED. Asked "what is the current
    # state of INC2000042", this arm returned knowledge articles and ranked the named
    # ticket 1,416th. The term inc2000042 appears in 2 of 82,296 documents and carries an
    # enormous IDF, but BM25 divides by document length, and a 24 token knowledge fragment
    # matching only "what is the of and who was it to" outscored it.
    #
    # Removing stopwords from the QUERY is the standard fix and is what every production
    # BM25 does. It is applied to the query only: taking them out of the documents would
    # change every document length and therefore the normalisation itself.
    STOPWORDS = frozenset("""
        a an and are as at be but by for from has have how i if in into is it its of on
        or that the their then there these they this to was were what when where which
        who why will with you your do does did been being had not no can could would
        should about after all any before more most other some such than
    """.split())

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        # Identifiers matter here: lnx0525 and INC2000042 are the whole point of some
        # questions, so the pattern keeps digits attached to letters.
        return re.findall(r"[a-z0-9_]+", text.lower())

    @classmethod
    def _query_terms(cls, question: str) -> list[str]:
        terms = [t for t in cls._tokenise(question) if t not in cls.STOPWORDS]
        # A question made entirely of stopwords would otherwise retrieve nothing at all.
        return terms or cls._tokenise(question)

    def retrieve(self, question: str, question_id: str, top: int = 40) -> Retrieved:
        import math
        t0 = time.perf_counter()
        q_terms = self._query_terms(question)
        scores: list[tuple[float, int]] = []
        for i, toks in enumerate(self.tokenised):
            if not toks:
                continue
            length = len(toks)
            score = 0.0
            counts: dict[str, int] = {}
            for term in toks:
                counts[term] = counts.get(term, 0) + 1
            for term in q_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        ranked = [(self.ids[i], self.texts[i]) for _, i in scores[:top]]
        ids, context, tokens = fit_to_budget(ranked)
        return Retrieved(self.name, question_id, ids, context, tokens,
                         (time.perf_counter() - t0) * 1000)


# ── the graph arm that needs no embeddings ─────────────────────────────────────────

class GraphOnly:
    """Traversal from an entry point named in the question.

    Not one of the five package retrievers: this is the honest floor for the graph side,
    the thing you can build with no model and no index at all. If the expensive arms do
    not beat it, that is worth knowing before anybody pays to embed sixty thousand
    records.
    """

    name = "graph_only"

    def __init__(self, session_factory: Callable, name_lookup: dict[str, str]):
        self.session_factory = session_factory
        self.name_lookup = name_lookup

    def _entry_points(self, question: str) -> list[str]:
        """Find items the question names. Longest names first, so a service is not
        missed because a host's name is a substring of it."""
        lowered = question.lower()
        return [key for name, key in
                sorted(self.name_lookup.items(), key=lambda kv: -len(kv[0]))
                if name.lower() in lowered][:3]

    def retrieve(self, question: str, question_id: str) -> Retrieved:
        """Walk out from the named item and return WHAT WAS REACHED.

        ⛔ THE FIRST VERSION RETURNED THE ENTRY POINT. It appended `(key, text)` where
        `key` was the item the question named, so asking "what does app1233 need" got
        back app1233. Every gold set here names the NEIGHBOURS, so this arm would have
        scored 0.00 on the questions it exists to win, and that would have been published
        as evidence that a traversal does not help.

        ⛔ AND IT ONLY WALKED UPWARD. `SUPPORTS` forward answers "what breaks if this
        breaks". It cannot answer "what does this need to work", which is the same graph
        read the other way. Both directions are walked, and each reached item is returned
        as its own record so it can be scored against a gold set.
        """
        t0 = time.perf_counter()
        keys = self._entry_points(question)
        if not keys:
            return Retrieved(self.name, question_id, [], "", 0,
                             (time.perf_counter() - t0) * 1000,
                             error="no item named in the question")
        rows: list[tuple[str, str]] = []
        seen: set[str] = set()
        with self.session_factory() as s:
            for key in keys:
                for rec in s.run("""
                    MATCH (start:ConfigurationItem {key:$key})
                    // upward: what stops working if start does
                    OPTIONAL MATCH up = (start)-[ru:SUPPORTS*1..4]->(above)
                      WHERE all(r IN ru WHERE r.carries_impact)
                    WITH start, collect(DISTINCT above) AS above
                    // downward: what start needs in order to work
                    OPTIONAL MATCH dn = (start)<-[rd:SUPPORTS*1..4]-(below)
                      WHERE all(r IN rd WHERE r.carries_impact)
                    WITH start, above, collect(DISTINCT below) AS below
                    UNWIND (above + below) AS reached
                    RETURN DISTINCT reached.key   AS key,
                                    reached.name  AS name,
                                    reached.sys_class_name AS cls,
                                    reached.environment    AS env
                    LIMIT 60
                """, key=key):
                    if not rec["key"] or rec["key"] in seen:
                        continue
                    seen.add(rec["key"])
                    rows.append((rec["key"],
                                 f"{rec['key']} {rec['name']} is a {rec['cls']} in the "
                                 f"{rec['env']} environment, reached from {key}."))
        ids, context, tokens = fit_to_budget(rows)
        return Retrieved(self.name, question_id, ids, context, tokens,
                         (time.perf_counter() - t0) * 1000)


# ── the arm this whole article is compared against ─────────────────────────────────

class VectorOnly:
    """Similarity search over embeddings of the same chunks, and nothing else.

    This is what most people mean by RAG, and it is the baseline the graph has to beat.
    It should win the questions phrased in different words from the text, and it should
    lose the ones that need counting, ordering in time, or a chain followed between
    records, because an index of nearest neighbours cannot express any of those.

    ⛔ THE SAME CHUNKS AND THE SAME BUDGET AS EVERY OTHER ARM. If the vector arm were
    given a different chunking or a larger context it would be a comparison between
    context sizes wearing the names of retrieval strategies, which is the failure Part 10
    exists to avoid.
    """

    name = "vector"

    def __init__(self, chunks: list[tuple[str, str]], vectors,
                 fingerprint: str | None = None) -> None:
        import numpy as np
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"{len(chunks):,} chunks against {vectors.shape[0]:,} vectors. These "
                f"must be the same corpus in the same order or every score is a lie.")
        # ⛔ MATCHING LENGTHS PROVE NOTHING. Two corpora of the same size in a different
        # order pass that check and then every question is scored against the wrong
        # document, silently and plausibly. The texts themselves are the only proof.
        if fingerprint is not None:
            from embed import corpus_fingerprint
            actual = corpus_fingerprint(chunks)
            if actual != fingerprint:
                raise ValueError(
                    "these vectors were built from different text than the chunks "
                    "handed to this arm. Re-embed rather than scoring against them.")
        self.ids = [c[0] for c in chunks]
        self.texts = [c[1] for c in chunks]
        self.vectors = vectors
        self._np = np

    def retrieve(self, question: str, question_id: str, top: int = 40) -> Retrieved:
        from embed import embed_texts
        t0 = time.perf_counter()
        q = embed_texts([question])[0]
        # Both sides are already unit length, so the dot product IS the cosine.
        scores = self.vectors @ q
        best = self._np.argpartition(-scores, min(top, len(scores) - 1))[:top]
        best = best[self._np.argsort(-scores[best])]
        ranked = [(self.ids[i], self.texts[i]) for i in best]
        ids, context, tokens = fit_to_budget(ranked)
        return Retrieved(self.name, question_id, ids, context, tokens,
                         (time.perf_counter() - t0) * 1000)


class Hybrid:
    """Keyword and similarity together, blended by reciprocal rank.

    ⛔ SCORES FROM TWO RETRIEVERS ARE NOT ON THE SAME SCALE and adding them is the most
    common way this is done wrong. A BM25 score is unbounded and depends on the corpus; a
    cosine is between minus one and one. Adding them lets whichever number happens to be
    larger decide every question.

    Reciprocal rank fusion ignores the scores and uses only the POSITIONS, which is why
    it needs no tuning and no normalisation. A document ranked third by one retriever and
    fourth by the other beats one ranked first by a single retriever and nowhere by the
    other, which is the behaviour you actually want from a hybrid.
    """

    name = "hybrid"

    # The constant from the original fusion paper. It stops rank one from dominating so
    # heavily that agreement further down never matters.
    K = 60

    def __init__(self, bm25: "BM25Only", vector: "VectorOnly",
                 chunks: list[tuple[str, str]] | None = None) -> None:
        self.bm25 = bm25
        self.vector = vector
        # The corpus is passed in rather than read out of one of the two arms. Reaching
        # into another arm's attributes ties this class to how that one happens to store
        # things today, and it cannot then be tested against a stand in.
        source = chunks if chunks is not None else list(zip(vector.ids, vector.texts))
        self.text_by_id = dict(source)

    def retrieve(self, question: str, question_id: str, top: int = 40) -> Retrieved:
        t0 = time.perf_counter()
        a = self.bm25.retrieve(question, question_id, top=top).record_ids
        b = self.vector.retrieve(question, question_id, top=top).record_ids

        # ⛔ DEDUPLICATE EACH RANKING BEFORE FUSING, KEEPING THE BEST RANK. The corpus is
        # keyed by source_id, and 301 knowledge articles split into several chunks each,
        # so one article can appear at ranks 3, 9 and 14 of the same list. Fusing per
        # occurrence gave it three additive contributions where a single chunk record got
        # one, so multi-chunk sources were silently promoted for being long.
        fused: dict[str, float] = {}
        for ranking in (a, b):
            best: dict[str, int] = {}
            for rank, rid in enumerate(ranking, start=1):
                best.setdefault(rid, rank)
            for rid, rank in best.items():
                fused[rid] = fused.get(rid, 0.0) + 1.0 / (self.K + rank)
        order = sorted(fused, key=lambda r: -fused[r])[:top]
        ranked = [(rid, self.text_by_id.get(rid, "")) for rid in order]
        ids, context, tokens = fit_to_budget(ranked)
        return Retrieved(self.name, question_id, ids, context, tokens,
                         (time.perf_counter() - t0) * 1000)
