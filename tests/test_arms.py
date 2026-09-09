"""Tests for the retrieval arms and the fairness controls.

The comparison is between retrieval strategies, so anything that lets one arm spend more
context than another turns it into a comparison of context sizes wearing strategy names.
That is what these check.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import pathlib
import sys

import pytest

R = pathlib.Path(__file__).resolve().parent.parent / "retrieval"
sys.path.insert(0, str(R))

from arms import (  # noqa: E402
    CONTEXT_TOKEN_BUDGET,
    BM25Only,
    NoRetrieval,
    approx_tokens,
    fit_to_budget,
)


@pytest.fixture
def corpus():
    return [
        ("INC001", "lnx0525: disk on the primary is nearly full. "
                   "```\n$ df -h\n/dev/nvme1n1 97%\n```"),
        ("INC002", "app1233: checkout is timing out for some customers. "
                   "Users report the page hangs."),
        ("INC003", "pg0071: replication lag climbing since this morning. "
                   "Possibly related to INC002, similar symptoms."),
        ("INC004", "win0836: connection pool timeout, could not acquire connection "
                   "within 30000ms at Pool.acquire"),
        ("KB9001", "How to handle capacity faults. Clear the old logs and resize."),
    ]


# ── the budget is the fairness mechanism ───────────────────────────────────────────

def test_no_arm_may_exceed_the_shared_budget(corpus):
    """'Same top k' is meaningless when one arm returns a 90 token chunk and another a
    subgraph. The budget is what makes the arms comparable."""
    big = [(f"X{i}", "word " * 2000) for i in range(20)]
    _ids, _text, tokens = fit_to_budget(big)
    assert tokens <= CONTEXT_TOKEN_BUDGET


def test_truncation_keeps_whole_records(corpus):
    """Half a ticket is worse than no ticket: a model answers from the half it can see
    and sounds exactly as certain."""
    ids, text, _ = fit_to_budget(corpus, budget=40)
    for rid in ids:
        original = next(t for i, t in corpus if i == rid)
        assert original in text, f"{rid} was cut in half"


def test_a_record_too_large_for_the_budget_is_skipped_not_sliced():
    huge = [("BIG", "word " * 5000), ("SMALL", "a short ticket about disk")]
    ids, text, tokens = fit_to_budget(huge, budget=50)
    assert "BIG" not in ids
    assert "SMALL" in ids
    assert tokens <= 50


def test_the_budget_is_declared_as_a_number():
    """It has to appear in the article as a figure rather than as a claim that things
    were kept fair."""
    assert isinstance(CONTEXT_TOKEN_BUDGET, int) and CONTEXT_TOKEN_BUDGET > 500


# ── the controls ───────────────────────────────────────────────────────────────────

def test_no_retrieval_does_not_look_at_the_question(corpus):
    """⛔ THE MOMENT IT RANKS BY THE QUESTION IT IS A RETRIEVER, and stops being the
    control it exists to be."""
    arm = NoRetrieval(corpus)
    a = arm.retrieve("disk full on lnx0525", "Q1")
    b = arm.retrieve("something entirely unrelated about certificates", "Q2")
    assert a.record_ids == b.record_ids, "the control changed with the question"


def test_bm25_finds_an_exact_identifier(corpus):
    """Some questions name a ticket or a host. Keyword search should win those outright,
    which is why it runs alone and not only inside a hybrid."""
    arm = BM25Only(corpus)
    got = arm.retrieve("what happened on win0836", "Q1")
    assert got.record_ids[0] == "INC004"


def test_bm25_finds_a_referenced_ticket_number(corpus):
    arm = BM25Only(corpus)
    got = arm.retrieve("which tickets reference INC002", "Q36")
    assert "INC003" in got.record_ids, "the cross reference was not found"


def test_bm25_ranks_a_pasted_block_above_prose(corpus):
    """Pasted output is pure text and no relationship helps."""
    arm = BM25Only(corpus)
    got = arm.retrieve("connection pool timeout Pool.acquire", "Q33")
    assert got.record_ids[0] == "INC004"


def test_an_arm_reports_what_it_spent(corpus):
    """Latency and tokens sit next to accuracy in the results table, or the table is
    only half the comparison."""
    got = BM25Only(corpus).retrieve("disk", "Q1")
    assert got.tokens > 0
    assert got.latency_ms >= 0
    assert got.arm == "bm25"


def test_an_arm_that_cannot_run_says_so_rather_than_scoring_zero(corpus):
    """A written query that will not parse is a reliability fact, not an accuracy one.
    Counting it as a miss hides the difference."""
    from arms import Retrieved
    failed = Retrieved("text2cypher", "Q08", error="generated query did not parse")
    assert failed.error
    assert failed.record_ids == []


def test_token_estimate_is_monotonic():
    assert approx_tokens("a" * 400) > approx_tokens("a" * 40)
    assert approx_tokens("") >= 1


# ── the significance test that could never have fired ──────────────────────────────

def _arm(scores: dict) -> dict:
    import statistics
    vals = list(scores.values())
    return {"per_question": scores,
            "recall_mean": statistics.fmean(vals),
            "recall_stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def test_the_sign_test_matches_the_arithmetic():
    """Checked against values anybody can work out by hand, because the whole point of
    writing it out instead of importing scipy is that a reader can check it."""
    from evaluate import _sign_test
    assert _sign_test(6, 0) == pytest.approx(2 / 64)
    assert _sign_test(8, 0) == pytest.approx(2 / 256)
    assert _sign_test(5, 0) == pytest.approx(2 / 32)
    assert _sign_test(0, 0) == 1.0


def test_a_lopsided_win_is_reported():
    """⛔ THE DEFECT THIS REPLACED. Recall is close to bimodal here: an arm either finds a
    question's supporting records or finds none, so per question scores are 1.00 or 0.00
    and the standard deviation sits near 0.5 no matter how the arms perform.

    The old check asked whether the gap between two MEANS exceeded that spread, so an arm
    winning 0.90 against 0.10 was reported as "no difference". The article would have been
    structurally unable to state its own finding, and the failure would have looked like
    honest caution rather than a broken test.
    """
    from evaluate import is_a_real_difference
    strong = _arm({f"Q{i}": (1.0 if i < 9 else 0.0) for i in range(10)})
    weak = _arm({f"Q{i}": (1.0 if i == 0 else 0.0) for i in range(10)})
    assert strong["recall_mean"] - weak["recall_mean"] > 0.7
    assert strong["recall_stdev"] > 0.25, "the spread that used to swallow the result"
    assert is_a_real_difference(strong, weak)


def test_two_identical_arms_are_never_a_finding():
    from evaluate import is_a_real_difference
    a = _arm({f"Q{i}": 0.5 for i in range(10)})
    assert not is_a_real_difference(a, _arm({f"Q{i}": 0.5 for i in range(10)}))


def test_a_gap_carried_by_two_questions_is_not_a_finding():
    """A large mean gap resting on two questions is exactly what a small question set
    produces by accident, and it must not be reported."""
    from evaluate import is_a_real_difference
    a = _arm({"Q1": 1.0, "Q2": 1.0, "Q3": 0.0})
    b = _arm({"Q1": 0.0, "Q2": 0.0, "Q3": 0.0})
    assert a["recall_mean"] - b["recall_mean"] > 0.6
    assert not is_a_real_difference(a, b)


def test_the_loser_is_never_declared_the_winner():
    from evaluate import is_a_real_difference
    weak = _arm({f"Q{i}": (1.0 if i == 0 else 0.0) for i in range(10)})
    strong = _arm({f"Q{i}": (1.0 if i < 9 else 0.0) for i in range(10)})
    assert not is_a_real_difference(weak, strong)


# ── the vector and hybrid arms ─────────────────────────────────────────────────────

def _fake_vectors(n, dim=8, seed=7):
    import numpy as np
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_a_vector_arm_refuses_a_corpus_of_the_wrong_size():
    from arms import VectorOnly
    chunks = [(f"D{i}", f"text {i}") for i in range(5)]
    with pytest.raises(ValueError, match="vectors"):
        VectorOnly(chunks, _fake_vectors(4))


def test_a_vector_arm_refuses_vectors_built_from_different_text():
    """⛔ MATCHING LENGTHS PROVE NOTHING. Two corpora of the same size in a different
    order pass a length check, and then every question is scored against the wrong
    document. It is silent, it is plausible, and no later step can detect it."""
    from arms import VectorOnly
    from embed import corpus_fingerprint
    original = [(f"D{i}", f"text {i}") for i in range(5)]
    shuffled = list(reversed(original))
    fp = corpus_fingerprint(original)
    VectorOnly(original, _fake_vectors(5), fingerprint=fp)      # the matching pair is fine
    with pytest.raises(ValueError, match="different text"):
        VectorOnly(shuffled, _fake_vectors(5), fingerprint=fp)


def test_the_fingerprint_ignores_ids_and_watches_text():
    """The same texts keyed two ways must share one cache entry, because this project
    keys the corpus by chunk_id to embed and by source_id to grade. A rewritten text
    must invalidate it."""
    from embed import corpus_fingerprint
    by_chunk = [("C1", "alpha"), ("C2", "beta")]
    by_source = [("INC1", "alpha"), ("INC2", "beta")]
    rewritten = [("C1", "alpha"), ("C2", "BETA")]
    assert corpus_fingerprint(by_chunk) == corpus_fingerprint(by_source)
    assert corpus_fingerprint(by_chunk) != corpus_fingerprint(rewritten)


def test_hybrid_fuses_on_rank_not_on_score():
    """⛔ ADDING TWO RETRIEVERS' SCORES IS THE COMMON WAY THIS GOES WRONG. A BM25 score
    is unbounded; a cosine is between minus one and one. Added together, whichever number
    happens to be larger decides every question. Fusion uses positions only.

    ⛔ MY FIRST VERSION OF THIS TEST ASSERTED SOMETHING FALSE. It gave one arm A, B, C and
    the other C, B, A, and expected B to win for being second on both. It does not:
    1/61 + 1/63 is 0.032266 and 2/62 is 0.032258, so the extremes edge out the middle
    because 1/x is convex. The implementation was right and the assertion was wrong.

    What fusion actually guarantees is the useful thing: a document BOTH retrievers found
    beats one that only a single retriever found, however highly that one ranked it.
    """
    from arms import Hybrid

    class Fake:
        def __init__(self, ids): self.ids = ids
        def retrieve(self, q, qid, top=40):
            from arms import Retrieved
            return Retrieved("fake", qid, list(self.ids), "", 0, 0.0)

    # X is first for one arm and absent from the other. B is merely second for both.
    h = Hybrid(Fake(["X", "B"]), Fake(["Y", "B"]), chunks=[(k, k) for k in "XYB"])
    order = h.retrieve("q", "Q1").record_ids
    assert order[0] == "B", (
        f"a document both retrievers found must outrank one only a single retriever "
        f"found, however highly. Got {order}")


def test_hybrid_keeps_a_document_only_one_arm_found():
    from arms import Hybrid

    class Fake:
        def __init__(self, ids): self.ids = ids
        def retrieve(self, q, qid, top=40):
            from arms import Retrieved
            return Retrieved("fake", qid, list(self.ids), "", 0, 0.0)

    h = Hybrid(Fake(["A", "B"]), Fake(["C", "D"]),
               chunks=[(k, k) for k in "ABCD"])
    got = set(h.retrieve("q", "Q1").record_ids)
    assert got == {"A", "B", "C", "D"}, (
        "fusion must union the two rankings, not intersect them")


def test_the_graph_arm_returns_what_it_reached_not_where_it_started():
    """⛔ IT RETURNED THE ENTRY POINT. Asked "what does app1233 need", the first version
    returned app1233. Every gold set here names the NEIGHBOURS, so this arm would have
    scored 0.00 on the questions it exists to win, and that would have been published as
    evidence that traversal does not help.

    A retrieval arm returns evidence. The item you already named is not evidence.
    """
    from arms import GraphOnly

    class FakeSession:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def run(self, cypher, **kw):
            assert kw["key"] == "app-1", kw
            return [{"key": "db-1", "name": "pg0001", "cls": "cmdb_ci_server",
                     "env": "prd"},
                    {"key": "host-1", "name": "lnx0001", "cls": "cmdb_ci_linux_server",
                     "env": "prd"}]

    arm = GraphOnly(lambda: FakeSession(), {"app0001": "app-1"})
    got = arm.retrieve("What does app0001 actually need in order to work?", "Q")
    assert got.record_ids == ["db-1", "host-1"], (
        f"expected the reached items, got {got.record_ids}")
    assert "app-1" not in got.record_ids, "the entry point is not an answer"


def test_hybrid_does_not_reward_a_source_for_having_many_chunks():
    """⛔ A KNOWLEDGE ARTICLE SPLIT INTO FIVE SECTIONS APPEARED FIVE TIMES IN ONE RANKING,
    and reciprocal rank fusion added a contribution for each. Long sources were promoted
    for being long, which is not what either retriever said about them."""
    from arms import Hybrid

    class Fake:
        def __init__(self, ids): self.ids = ids
        def retrieve(self, q, qid, top=40):
            from arms import Retrieved
            return Retrieved("fake", qid, list(self.ids), "", 0, 0.0)

    # KB appears three times in one ranking; INC once, but higher.
    h = Hybrid(Fake(["INC1", "KB1", "KB1", "KB1"]), Fake(["INC1", "KB1"]),
               chunks=[("INC1", "a"), ("KB1", "b")])
    order = h.retrieve("q", "Q1").record_ids
    assert order[0] == "INC1", (
        f"the multi-chunk source was promoted by repetition alone: {order}")
