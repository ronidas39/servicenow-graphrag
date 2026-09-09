"""Tests for the scoring that produces every number Part 10 publishes.

This file had no tests at all for most of the project, which is the wrong way round:
`arms.py` decides what comes back, and if it is wrong the numbers are wrong in an obvious
direction. `evaluate.py` decides what the numbers MEAN, and if it is wrong the numbers
look entirely reasonable. The 0.00 recall that cost this project an evening was a scoring
question, not a retrieval one.

What is checked here, and why each one is a defect that has actually happened somewhere in
this project or in its sources:

  a metric that scores a broken gold set instead of refusing to
  a summary line that asserts a conclusion the table beside it contradicts
  an average taken over rows that do not mean the same thing
  a "not measured" case printed as a measurement of zero

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field

import pytest

R = pathlib.Path(__file__).resolve().parent.parent / "retrieval"
sys.path.insert(0, str(R))

from evaluate import (  # noqa: E402
    ENUMERATION_CEILING,
    Score,
    aggregate,
    recall_and_mrr,
    report,
    score_one,
)


# Local stand-ins rather than the real dataclasses. The real ones pull in the estate,
# which takes seconds to load and would make this file test the generator by accident.
@dataclass
class FakeRetrieved:
    arm: str = "BM25Only"
    question_id: str = "Q01"
    record_ids: list[str] = field(default_factory=list)
    tokens: int = 100
    latency_ms: float = 5.0
    error: str = ""


@dataclass
class FakeQuestion:
    id: str = "Q01"
    kind: str = "lookup"
    holdout: bool = False


@dataclass
class FakeGold:
    supporting: set[str] = field(default_factory=set)
    gradable: bool = True
    expect_empty: bool = False


# ── the metric itself ────────────────────────────────────────────────────────────────

def test_recall_counts_distinct_hits_not_repeats():
    # An arm that returns the same right record three times has found one record.
    recall, _, _ = recall_and_mrr(["a", "a", "a"], {"a", "b"})
    assert recall == 0.5


def test_reciprocal_rank_is_the_first_hit_not_the_best_one():
    _, rr, _ = recall_and_mrr(["x", "y", "a", "b"], {"a", "b"})
    assert rr == pytest.approx(1 / 3)


def test_precision_is_over_what_came_back_not_over_the_gold_set():
    _, _, precision = recall_and_mrr(["a", "x", "y", "z"], {"a", "b", "c"})
    assert precision == 0.25


def test_an_empty_gold_set_raises_instead_of_scoring_zero():
    # ⛔ THE DEFECT THIS EXISTS FOR. Returning 0.0 here marks every arm as having found
    # nothing, which is indistinguishable from a real finding about retrieval, and that
    # is exactly how the first evaluation run wasted an evening.
    with pytest.raises(ValueError, match="empty gold set"):
        recall_and_mrr(["a", "b"], set())


def test_nothing_retrieved_scores_zero_rather_than_dividing_by_zero():
    assert recall_and_mrr([], {"a"}) == (0.0, 0.0, 0.0)


# ── one row of the table ─────────────────────────────────────────────────────────────

def test_an_arm_that_failed_to_run_gets_no_accuracy_columns_at_all():
    s = score_one(FakeRetrieved(error="the generated query would not parse"),
                  FakeQuestion(), FakeGold(supporting={"a"}))
    assert s.failed_to_run
    assert s.recall_at_k is None and s.mrr is None
    # Reliability is a separate fact from accuracy, and scoring the failure as 0.00 recall
    # would bury it inside an average that then reads as a retrieval result.


def test_an_ungradable_question_leaves_the_retrieval_columns_empty():
    s = score_one(FakeRetrieved(record_ids=["a"]), FakeQuestion(),
                  FakeGold(supporting=set(), gradable=False))
    assert s.recall_at_k is None and s.precision_at_k is None


def test_the_none_answer_is_right_only_when_the_arm_returned_nothing():
    empty = FakeGold(supporting=set(), expect_empty=True)
    assert score_one(FakeRetrieved(record_ids=[]), FakeQuestion(), empty).precision_at_k == 1.0
    assert score_one(FakeRetrieved(record_ids=["x"]), FakeQuestion(), empty).precision_at_k == 0.0


def test_a_gold_set_bigger_than_any_budget_is_scored_on_precision_not_recall():
    gold = FakeGold(supporting={f"r{i}" for i in range(ENUMERATION_CEILING + 1)})
    s = score_one(FakeRetrieved(record_ids=["r0", "r1", "zzz"]), FakeQuestion(), gold)
    assert s.enumeration
    assert s.recall_at_k is None, "recall on an enumeration measures the budget, not the arm"
    assert s.precision_at_k == pytest.approx(2 / 3)


def test_the_chance_baseline_is_the_share_of_the_corpus_the_answer_covers():
    gold = FakeGold(supporting={f"r{i}" for i in range(100)})
    s = score_one(FakeRetrieved(record_ids=["r0"]), FakeQuestion(), gold, corpus_size=1000)
    assert s.chance_precision == 0.1
    # Precision of 0.30 on a question whose answer is 10% of the corpus is three times
    # chance. Without this beside it, 0.30 reads as skill.


def test_a_question_right_at_the_ceiling_is_still_scored_on_recall():
    # The boundary matters: an off-by-one here silently moves questions between two
    # tables that are reported differently.
    gold = FakeGold(supporting={f"r{i}" for i in range(ENUMERATION_CEILING)})
    s = score_one(FakeRetrieved(record_ids=["r0"]), FakeQuestion(), gold)
    assert not s.enumeration and s.recall_at_k is not None


# ── the summary ──────────────────────────────────────────────────────────────────────

def _scores(arm: str, per_question: dict[str, float | None], **kw) -> list[Score]:
    return [Score(arm=arm, question_id=q, kind=kw.get("kind", "lookup"),
                  holdout=kw.get("holdout", False), recall_at_k=v, mrr=v,
                  tokens=100, latency_ms=5.0)
            for q, v in per_question.items()]


def test_the_average_skips_rows_that_have_no_recall_to_average():
    rows = _scores("A", {"Q1": 1.0, "Q2": 0.0})
    rows += [Score(arm="A", question_id="Q3", kind="lookup", holdout=False,
                   failed_to_run=True)]
    agg = aggregate(rows)["A"]
    assert agg["questions"] == 3 and agg["measured"] == 2
    assert agg["recall_mean"] == 0.5, "a failure must not be averaged in as a zero"
    assert agg["failed_to_run"] == 1


def test_holdout_questions_are_averaged_on_their_own():
    rows = _scores("A", {"Q1": 1.0, "Q2": 1.0})
    rows += _scores("A", {"Q9": 0.0}, holdout=True)
    agg = aggregate(rows)["A"]
    assert agg["recall_mean"] == pytest.approx(2 / 3)
    assert agg["holdout_recall"] == 0.0


def test_no_holdout_recall_at_all_reports_not_measured_rather_than_zero():
    # ⛔ THIS PRINTED 0.00 FOR EVERY ARM ONCE, from a single broken gold set, and it read
    # like a finding about generalisation. Silence read the same way: as "nothing to
    # report" rather than "this check did not run".
    text = report(aggregate(_scores("A", {"Q1": 1.0}) + _scores("B", {"Q1": 0.0})))
    assert "NOT MEASURED" in text
    assert "0.00" not in text.split("held out questions only")[1].split("enumeration")[0]


def test_the_verdict_line_agrees_with_the_table_above_it_when_nothing_separates():
    agg = aggregate(_scores("A", {f"Q{i}": 1.0 for i in range(8)})
                    + _scores("B", {f"Q{i}": 1.0 for i in range(8)}))
    text = report(agg)
    assert "SEPARATES" not in text
    assert "No pair above separates" in text


def test_the_verdict_line_agrees_with_the_table_above_it_when_a_pair_separates():
    # ⛔ THE DEFECT. The closing paragraph said "no pair above separates" unconditionally.
    # It was true every time it was printed, which is why it survived, and an assertion
    # that happens to agree with the data is not a measurement of it.
    strong = _scores("A", {f"Q{i}": 1.0 for i in range(8)})
    weak = _scores("B", {f"Q{i}": 0.0 for i in range(8)})
    text = report(aggregate(strong + weak))
    assert "SEPARATES" in text
    assert "No pair above separates" not in text
    assert "A over B" in text


def test_the_reachable_floor_comes_from_the_widest_comparison():
    # Eight paired questions can reach p = 2/256. Saying so tells a reader what this
    # question set could ever have shown, which is a different fact from what it did show.
    text = report(aggregate(_scores("A", {f"Q{i}": 1.0 for i in range(8)})
                            + _scores("B", {f"Q{i}": 0.0 for i in range(8)})))
    assert "covers 8 question(s)" in text
    assert "0.008" in text


def test_per_question_scores_survive_aggregation_so_pairs_can_be_compared():
    # The arms answer the same questions, so the comparison is paired. Dropping the pairs
    # during aggregation would force the report back onto a comparison of two means, which
    # is the test that could never fire.
    agg = aggregate(_scores("A", {"Q1": 1.0, "Q2": 0.0}))["A"]
    assert agg["per_question"] == {"Q1": 1.0, "Q2": 0.0}


def test_an_arm_with_no_measurable_question_reports_none_rather_than_zero():
    rows = [Score(arm="A", question_id="Q1", kind="lookup", holdout=False,
                  failed_to_run=True)]
    agg = aggregate(rows)
    assert agg["A"]["recall_mean"] is None
    assert "n/a" in report(agg), "an unmeasured arm must not be printed as 0.00"


def test_the_report_renders_with_one_arm_and_does_not_invent_a_comparison():
    text = report(aggregate(_scores("A", {"Q1": 1.0})))
    assert "every pair" not in text
