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

    # ⛔ THIS DEFENDS AGAINST A FAILURE THAT NO LONGER FIRES ON THIS CORPUS, AND THE
    # COMMENT USED TO CLAIM OTHERWISE. It said that asking "what is the current state of
    # INC2000042" ranked the named ticket 1,416th, beaten by a short knowledge fragment
    # matching only "what is the of and who was it to", because BM25 divides by document
    # length. That was measured on an earlier version of the corpus. Re-checked on the
    # corpus that ships: the two documents holding that ticket number rank 1 and 2 whether
    # the stopwords are removed or not.
    #
    # The removal stays because the mechanism is real wherever a corpus holds short
    # documents full of common words, and it costs one set lookup per query term. It is
    # applied to the query only: taking them out of the documents would change every
    # document length and therefore the normalisation itself.
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


# ── the two arms that use the graph AND an index ───────────────────────────────────
#
# ⛔ THESE WERE NAMED IN THIS MODULE'S OWN DOCSTRING AND NEVER WRITTEN. `vector_cypher`
# and `hybrid_cypher` were listed at the top of this file as two of the seven arms, and
# Part 10 section 110 reported four of seven running with the three graph arms missing.
# The reason they were missing was never the code: the chunks had no home in the graph,
# because the graph in Neo4j was a different estate from the corpus. Once the chunks and
# their vectors are in the graph and joined to their records, both of these are short.

# The walk both of them share, from the records an index matched out to what those
# records depend on and what depends on them.
#
# ⛔ THE ENTRY POINT IS A CHUNK AND THE ANSWER IS A NEIGHBOURHOOD. Returning the matched
# chunk is what VectorOnly already does, and an arm that did that with extra steps would
# measure nothing. What is returned is what the traversal REACHED, which is what a gold
# set for a chain question names.
#
# ⛔ THE TRAVERSAL FILTERS ON carries_impact, exactly as GraphOnly does. Part 6 section 66
# measures what dropping that filter costs: the same four hops reach 3,365 items instead
# of 16, and an arm that returned the estate would score well for the wrong reason.
_WALK = """
UNWIND $seeds AS sid
// ⛔ CALL (sid) { }, NOT CALL { WITH sid }. The older form is deprecated in Cypher 5 and
// the server says so on every single call, which buried the run's own output in warnings.
CALL (sid) {
  MATCH (c:ConfigurationItem {key: sid}) RETURN c
  UNION
  MATCH (:Incident {number: sid})-[:AFFECTS]->(c:ConfigurationItem) RETURN c
  UNION
  MATCH (:Change {number: sid})-[:CHANGES]->(c:ConfigurationItem) RETURN c
}
WITH DISTINCT c
// ⛔ 4, NOT 3, AND THE DEPTH HAS TO MATCH GraphOnly OR THE TABLE IS NOT A COMPARISON.
// This said *1..3 while GraphOnly said *1..4, so two arms printed in the same table were
// walking to different depths, and the one difference between them that the article
// discusses was not the only difference between them. Three hops also cannot reach the
// storage array from the payments service, which is the sixteen item chain Part 6
// section 57b measures and the record this whole article opens with, so the arm was
// structurally unable to answer the question it exists to answer.
OPTIONAL MATCH (c)-[ru:SUPPORTS*1..4]->(above)
  WHERE all(r IN ru WHERE r.carries_impact)
WITH c, collect(DISTINCT above) AS above
OPTIONAL MATCH (c)<-[rd:SUPPORTS*1..4]-(below)
  WHERE all(r IN rd WHERE r.carries_impact)
WITH c, above, collect(DISTINCT below) AS below
UNWIND ([c] + above + below) AS reached
RETURN DISTINCT reached.key AS key, reached.name AS name,
                reached.sys_class_name AS cls, reached.environment AS env
LIMIT 80
"""


def _walk_from(session_factory, seeds: list[str]) -> list[tuple[str, str]]:
    """Run the shared walk and return (key, sentence) pairs ready for the budget."""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    if not seeds:
        return rows
    with session_factory() as s:
        for rec in s.run(_WALK, seeds=seeds):
            key = rec["key"]
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append((key, f"{key} {rec['name']} is a {rec['cls']} in the "
                              f"{rec['env']} environment."))
    return rows


class VectorCypher:
    """Similarity finds the entry point, then the graph is walked from it.

    The search runs INSIDE Neo4j against the vector index on :Chunk(embedding), rather
    than against the numpy array VectorOnly uses. That is deliberate: it is the thing
    section 98 builds, and doing it in the database is what makes the walk a single round
    trip from the matched chunk rather than a second query from Python.
    """

    name = "vector_cypher"

    def __init__(self, session_factory: Callable, index: str = "chunk_embedding",
                 seeds: int = 8) -> None:
        self.session_factory = session_factory
        self.index = index
        self.seeds = seeds

    def _seed_records(self, question: str) -> list[str]:
        from embed import embed_texts
        q = embed_texts([question])[0].astype(float).tolist()
        with self.session_factory() as s:
            rows = s.run("""
                CALL db.index.vector.queryNodes($index, $k, $q) YIELD node, score
                MATCH (node)-[:CHUNK_OF]->(rec)
                RETURN DISTINCT coalesce(rec.number, rec.key) AS sid
                ORDER BY sid
            """, index=self.index, k=self.seeds, q=q)
            return [r["sid"] for r in rows if r["sid"]]

    def retrieve(self, question: str, question_id: str, top: int = 40) -> Retrieved:
        t0 = time.perf_counter()
        seeds = self._seed_records(question)
        if not seeds:
            return Retrieved(self.name, question_id, [], "", 0,
                             (time.perf_counter() - t0) * 1000,
                             error="the vector index returned nothing")
        rows = _walk_from(self.session_factory, seeds)[:top]
        ids, context, tokens = fit_to_budget(rows)
        return Retrieved(self.name, question_id, ids, context, tokens,
                         (time.perf_counter() - t0) * 1000)


class HybridCypher:
    """Both indexes pick the entry points, then the same graph walk runs from them.

    ⛔ THE FUSION IS THE ONE Hybrid ALREADY USES, not a second implementation of it. Two
    reciprocal rank fusions in one file drift apart, and then a difference between this
    arm and `hybrid` would be a difference in the fusion rather than in the graph, which
    is the only thing this arm is supposed to add.
    """

    name = "hybrid_cypher"

    def __init__(self, hybrid: "Hybrid", session_factory: Callable,
                 seeds: int = 8) -> None:
        self.hybrid = hybrid
        self.session_factory = session_factory
        self.seeds = seeds

    def retrieve(self, question: str, question_id: str, top: int = 40) -> Retrieved:
        t0 = time.perf_counter()
        fused = self.hybrid.retrieve(question, question_id, top=top).record_ids
        rows = _walk_from(self.session_factory, fused[:self.seeds])[:top]
        ids, context, tokens = fit_to_budget(rows)
        return Retrieved(self.name, question_id, ids, context, tokens,
                         (time.perf_counter() - t0) * 1000)


# ── the arm that lets a model write the query ──────────────────────────────────────
#
# ⛔ THIS IS THE LAST OF THE EIGHT AND IT IS THE ONLY ONE THAT CAN FAIL IN A NEW WAY.
# Every other arm either returns records or returns none. This one can produce Cypher
# that does not parse, or Cypher that parses and means something else. Section 100 says
# those are counted separately from wrong answers, because failing to run is a
# reliability fact and not an accuracy one.

# ⛔ THE SCHEMA IS READ FROM THE GRAPH, NOT TYPED. A hand-written schema drifts from the
# database the moment a label is added, and the model is then confidently wrong in a way
# that looks like a model problem.
_SCHEMA_QUERY = """
CALL db.labels() YIELD label
RETURN 'label' AS kind, label AS name
UNION
CALL db.relationshipTypes() YIELD relationshipType
RETURN 'relationship' AS kind, relationshipType AS name
"""

# ⛔ DIRECTION IS SPELLED OUT IN WORDS. Part 6 section 55 is the whole reason: a model
# given only "(:ConfigurationItem)-[:SUPPORTS]->(:ConfigurationItem)" has a fifty percent
# chance of reading it the wrong way round, and a backwards traversal returns rows.
_DIRECTION = """
(child:ConfigurationItem)-[:SUPPORTS]->(parent:ConfigurationItem)
  means the CHILD supports the PARENT, so things that depend on X are found by
  (x)-[:SUPPORTS]->(dependent). Impact flows ALONG the arrow.
  SUPPORTS has a boolean property carries_impact; filter on it for blast radius.
(:Incident)-[:AFFECTS]->(:ConfigurationItem)
(:Change)-[:CHANGES]->(:ConfigurationItem)
(:Chunk)-[:CHUNK_OF]->(any record)
Keys: ConfigurationItem.key and .name, Incident.number, Change.number,
Problem.number, KnowledgeArticle.number.
"""

_EXAMPLES = """
Q: how many incidents name no configuration item at all
A: MATCH (i:Incident) WHERE NOT (i)-[:AFFECTS]->() RETURN count(i) AS n

Q: what depends on cluster-us-east-01
A: MATCH (c:ConfigurationItem {name:'cluster-us-east-01'})-[:SUPPORTS]->(d)
   RETURN d.key AS key LIMIT 60

Q: which items does app0005 need in order to work
A: MATCH (a:ConfigurationItem {name:'app0005'})<-[:SUPPORTS]-(n)
   RETURN n.key AS key LIMIT 60
"""

# ⛔ SECTION 102'S FOUR RULES. Three of them are enforced here: no writes, no
# unbounded traversal, and a real transaction timeout. The fourth, refusing a
# query that names a label or property the schema does not have, is not
# implemented and section 111c reports what that costs.
# A read-only user is the right answer in production; this refuses writes in the client
# too, because the article ships this code and a reader may not have made one yet.
_FORBIDDEN = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|"
                        r"CALL\s+\{[^}]*\b(CREATE|MERGE|DELETE)\b)", re.I)
# ⛔ `[r*1..]` IS UNBOUNDED AND THE FIRST VERSION OF THIS LET IT THROUGH. It caught `[r*]`
# and `[r*..]` and stopped there, so a lower bound with no upper bound read as bounded.
# What makes a pattern bounded is a number AFTER the dots, and nothing else.
#   *]     unbounded      *..]    unbounded      *1..]   unbounded
#   *2]    exactly two    *..4]   bounded        *1..4]  bounded
_UNBOUNDED = re.compile(r"\[[^\]]*\*\s*(?:\]|\d*\s*\.\.\s*\])")


def _extract_cypher(text: str) -> str:
    """Pull the query out of whatever the model wrapped it in."""
    fenced = re.search(r"```(?:cypher)?\s*(.+?)```", text, re.S | re.I)
    body = fenced.group(1) if fenced else text
    # Drop any prose before the first clause a query can start with.
    m = re.search(r"\b(MATCH|WITH|UNWIND|CALL|RETURN|PROFILE|EXPLAIN)\b", body, re.I)
    return (body[m.start():] if m else body).strip().rstrip(";")


class Text2Cypher:
    """The model is given the schema and writes the Cypher itself.

    ⛔ IT IS ALLOWED TO FAIL AND THE FAILURES ARE COUNTED. `Retrieved.error` carries the
    reason, and Part 10 reports those in their own column rather than folding them into
    recall. An arm that silently returned nothing on a syntax error would look like an
    arm that retrieved nothing, and those are different facts about a system.
    """

    name = "text2cypher"

    def __init__(self, session_factory: Callable, chat_url: str,
                 model: str = "chat", retries: int = 2, timeout_s: int = 30) -> None:
        self.session_factory = session_factory
        self.chat_url = chat_url.rstrip("/")
        self.model = model
        self.retries = retries
        self.timeout_s = timeout_s
        self.attempts = 0
        self._schema: str | None = None

    def schema(self) -> str:
        if self._schema is None:
            with self.session_factory() as s:
                rows = list(s.run(_SCHEMA_QUERY))
            labels = sorted(r["name"] for r in rows if r["kind"] == "label")
            rels = sorted(r["name"] for r in rows if r["kind"] == "relationship")
            self._schema = (f"Node labels: {', '.join(labels)}\n"
                            f"Relationship types: {', '.join(rels)}\n"
                            f"{_DIRECTION}")
        return self._schema

    def _ask(self, question: str, previous_error: str | None) -> str:
        import json as _json
        import urllib.request
        fix = ("\nYour previous query failed with this error, fix it:\n"
               f"{previous_error}\n" if previous_error else "")
        prompt = (f"You write Cypher for a Neo4j graph. Schema:\n{self.schema()}\n"
                  f"Examples:\n{_EXAMPLES}\n"
                  "Rules: read only, no CREATE MERGE DELETE SET, never an unbounded "
                  "variable-length pattern, always LIMIT. Return ONLY the query."
                  f"{fix}\nQ: {question}\nA:")
        req = urllib.request.Request(
            f"{self.chat_url}/v1/chat/completions",
            data=_json.dumps({"model": self.model, "temperature": 0, "max_tokens": 300,
                              "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return _json.load(r)["choices"][0]["message"]["content"]

    def retrieve(self, question: str, question_id: str, top: int = 40) -> Retrieved:
        t0 = time.perf_counter()
        error: str | None = None
        for _ in range(self.retries + 1):
            self.attempts += 1
            try:
                cypher = _extract_cypher(self._ask(question, error))
            except Exception as exc:                       # noqa: BLE001
                error = f"the model did not answer: {exc}"
                continue
            if _FORBIDDEN.search(cypher):
                error = "refused: the query writes to the database"
                continue
            if _UNBOUNDED.search(cypher):
                error = "refused: unbounded variable-length pattern"
                continue
            # ⛔ THE TIMEOUT GOES ON THE TRANSACTION, NOT ON `run`, AND GETTING THAT
            # WRONG COST TWELVE MINUTES OF A RENTED GPU SITTING IDLE. This used to say
            # `s.run(cypher, timeout=self.timeout_s)`. The driver takes unknown keyword
            # arguments as QUERY PARAMETERS, so that did not set a time limit, it quietly
            # bound `$timeout` to 30 and ran the query with no limit at all. The model
            # then wrote a three way cartesian product over 60,000 incidents and the run
            # stopped dead: no error, no timeout, one transaction still going twelve
            # minutes later, and I had to terminate it by hand from another session.
            #
            # Neither guard above could have caught it. `_FORBIDDEN` looks for writes and
            # this only reads. `_UNBOUNDED` looks for `[*]` and this has no variable
            # length pattern at all. The query is not malformed and it is not dangerous.
            # It is merely enormous, and the only thing that defends against enormous is
            # a clock. `begin_transaction(timeout=...)` is where the clock lives.
            try:
                with self.session_factory() as s:
                    with s.begin_transaction(timeout=self.timeout_s) as tx:
                        tx.run(f"EXPLAIN {cypher}").consume()
                        rows = list(tx.run(cypher))
            except Exception as exc:                       # noqa: BLE001
                error = str(exc).split("\n")[0][:160]
                continue

            # ⛔ WHATEVER IT RETURNED IS TURNED INTO RECORD IDS, OR INTO NOTHING. A
            # counting query returns a number and no ids, and that is a correct answer
            # this harness cannot grade. Reporting it as recall 0.00 would be scoring the
            # grader rather than the arm, so it is returned with the count as context.
            ranked: list[tuple[str, str]] = []
            for rec in rows[:top]:
                for value in rec.values():
                    if isinstance(value, str) and value:
                        ranked.append((value, f"{value}, returned by the written query."))
            context_extra = "" if ranked else (
                f"the query returned {len(rows)} row(s) with no record id in them: "
                + "; ".join(f"{k}={v}" for rec in rows[:3] for k, v in rec.items())[:400])
            ids, context, tokens = fit_to_budget(ranked)
            return Retrieved(self.name, question_id, ids, context or context_extra,
                             tokens, (time.perf_counter() - t0) * 1000)

        return Retrieved(self.name, question_id, [], "", 0,
                         (time.perf_counter() - t0) * 1000, error=error)
