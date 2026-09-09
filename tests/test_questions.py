"""Tests for the frozen question set.

The article's whole claim to fairness is that these questions predate the graph schema
and the retrievers. That claim is only worth something if it is enforced, so the shape of
the set is checked here rather than described in prose.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import pathlib
import re
import sys
from collections import Counter

Q = pathlib.Path(__file__).resolve().parent.parent / "questions"
sys.path.insert(0, str(Q))

from questions import QUESTIONS, frozen_hash  # noqa: E402


def test_no_question_is_phrased_in_terms_of_the_schema():
    """If a question can only be asked using a label or a Cypher clause, it was written
    backwards from the model rather than forwards from a person."""
    BANNED = re.compile(
        r"\b(MATCH|MERGE|UNWIND|RETURN|cypher|node|edge|traverse|traversal|"
        r"ConfigurationItem|SUPPORTS|AFFECTS|sys_class_name|cmdb_rel_ci|embedding|"
        r"vector|cosine|index)\b", re.I)
    offenders = [q.id for q in QUESTIONS if BANNED.search(q.text)]
    assert offenders == [], f"these questions leak the schema: {offenders}"


def test_the_set_does_not_decide_its_own_result():
    """A set where the graph wins by construction is a demonstration, not a measurement.
    The first draft was 19 to 7 in the graph's favour and was rebalanced."""
    favours = Counter(q.expect_favours for q in QUESTIONS)
    graph = favours["graph"]
    other = favours["text"] + favours["hybrid"] + favours["either"]
    assert other >= graph * 0.8, (
        f"{graph} questions expect the graph to win against {other} that do not. "
        "The set is stacked."
    )


def test_every_kind_of_question_is_represented():
    kinds = Counter(q.kind for q in QUESTIONS)
    for kind in ("lookup", "semantic", "multi_hop", "aggregation", "temporal",
                 "negation"):
        assert kinds[kind] >= 3, f"only {kinds[kind]} questions of kind {kind}"


def test_enough_questions_are_held_out():
    """Reported results have to be checkable against questions nobody tuned against."""
    held = sum(1 for q in QUESTIONS if q.holdout)
    assert held >= len(QUESTIONS) * 0.20, f"only {held} of {len(QUESTIONS)} held out"


def test_every_question_says_what_a_right_answer_looks_like():
    """Without this, grading is whatever the grader felt like on the day."""
    for q in QUESTIONS:
        assert q.answer_shape.strip(), f"{q.id} does not say what a correct answer is"
        assert q.expect_favours in ("graph", "text", "hybrid", "either")


def test_question_ids_are_unique_and_the_hash_is_stable():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    assert frozen_hash() == frozen_hash(), "the hash is not stable"
    assert len(frozen_hash()) == 64
