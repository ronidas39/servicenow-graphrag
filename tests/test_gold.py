"""Tests for the gold supporting-record sets.

A gold set that scores every retriever identically measures nothing, and one built from
the retriever's own mechanism flatters it. Both failure modes are checked here.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

Q = pathlib.Path(__file__).resolve().parent.parent / "questions"
sys.path.insert(0, str(Q))

from gold import World, build  # noqa: E402
from questions import QUESTIONS  # noqa: E402

DATASET = pathlib.Path(__file__).resolve().parent.parent / "dataset"
pytestmark = pytest.mark.skipif(not (DATASET / "manifest.json").exists(),
                                reason="dataset not built")


@pytest.fixture(scope="module")
def gold():
    return build(World())


def test_every_question_has_a_verdict(gold):
    """A question with neither a gold set nor a judge is scored by nothing at all."""
    missing = [q.id for q in QUESTIONS if q.id not in gold]
    assert missing == [], f"no gold entry for {missing}"


def test_no_gold_set_is_accidentally_empty(gold):
    """⛔ AN EMPTY SET SCORES EVERY RETRIEVER THE SAME. It caught Q19 on the first run,
    where the honest answer really is none. That case is declared with expect_empty, so
    a real 'none' can be told from a rule that silently failed to run."""
    silent = [g.question_id for g in gold.values()
              if g.gradable and not g.supporting and not g.expect_empty]
    assert silent == [], f"these gold sets are empty and do not say why: {silent}"


def test_a_gold_set_is_not_the_whole_dataset(gold):
    """⛔ Q12 FIRST RETURNED 872 RECORDS for 'what does this application need', because
    the rule grabbed the first item with dependencies and that was a shared cluster. A
    gold set that large is not an answer, it is the estate."""
    world = World()
    total = len(world.incidents) + len(world.cis) + len(world.changes)
    for g in gold.values():
        if g.gradable and g.supporting:
            share = len(g.supporting) / total
            assert share < 0.30, (
                f"{g.question_id} expects {len(g.supporting):,} records, "
                f"{share:.0%} of everything. That is not a question, it is a dump.")


def test_the_multi_hop_gold_is_small_enough_to_be_wrong(gold):
    """A blast radius of twelve items can be got wrong. One of four thousand cannot be
    distinguished from a retriever that returns everything."""
    for qid in ("Q08", "Q12"):
        if qid in gold and gold[qid].supporting:
            assert len(gold[qid].supporting) < 100, (
                f"{qid} has {len(gold[qid].supporting)} supporting records; "
                "a hop cap or a better seed is needed")


def test_every_mechanical_rule_is_written_down(gold):
    """The article publishes these rules. A gold set whose rule is blank cannot be
    defended when somebody asks how the number was reached."""
    for g in gold.values():
        if g.gradable:
            assert g.rule.strip(), f"{g.question_id} has a gold set and no stated rule"


def test_the_split_between_mechanical_and_judged_is_declared(gold):
    """Mixing a hard grade and a soft one silently would let a judge's opinion stand in
    for a measurement."""
    hard = [g for g in gold.values() if g.gradable]
    soft = [g for g in gold.values() if not g.gradable]
    assert hard and soft, "both kinds of grading must exist"
    for g in soft:
        assert g.note.strip(), f"{g.question_id} is judged and does not say why"


# ── the binding bug, which made the whole measurement read as broken ───────────────

NAMED_IN_QUESTIONS = re.compile(
    r"\b(?:app|lnx|win|pg|my|ora|rdx|k8s|esx|nfs|lb|fw|sw|rack|cluster)[a-z0-9\-]*\d[a-z0-9\-]*\b",
    re.I)


def test_a_question_that_names_an_item_is_graded_against_that_item(gold):
    """⛔ THIS IS THE ONE THAT COST A DAY. Q12 asks what app1233 needs. The rule picked
    the first application with a workable dependency count instead, which was a different
    record, so the arms fetched app1233 and the gold described something else. The two id
    sets could never intersect and recall was structurally zero for every arm at every k,
    which looks exactly like a broken retriever.

    Nothing was wrong with retrieval. A question that names a record is a contract.
    """
    world = World()
    for q in QUESTIONS:
        named = set(NAMED_IN_QUESTIONS.findall(q.text))
        g = gold.get(q.id)
        if not named or g is None or not g.gradable or not g.bound_to:
            continue
        bound_name = world.cis[g.bound_to]["name"].lower()
        assert any(n.lower() == bound_name for n in named), (
            f"{q.id} names {sorted(named)} but its gold is bound to "
            f"{bound_name!r}. The question and its answer are about different records, "
            f"so every arm scores zero however good it is.")


def test_the_binding_resolves_by_name_and_refuses_to_guess():
    """A missing name must raise, not return None. A binding that silently goes absent is
    how the gold ends up describing a record the question never mentioned."""
    world = World()
    assert world.cis[world.resolve_named("app1233")]["name"] == "app1233"
    with pytest.raises(KeyError):
        world.resolve_named("app999999")


def test_a_production_question_is_not_bound_to_a_dev_record(gold):
    """"The payments service is down" is a production sentence. Nobody pages anyone about
    dev, and a dev service has a blast radius too small to be the question's answer."""
    world = World()
    g = gold["Q08"]
    assert world.cis[g.bound_to]["environment"] == "prd", (
        f"Q08 bound to a {world.cis[g.bound_to]['environment']} service")
    assert len(g.supporting) > 3, (
        "the bound service has almost no blast radius, so the article's opening "
        "question has no visible answer")


def test_the_measured_questions_are_not_all_ones_the_graph_should_win(gold):
    """⛔ THE BALANCE WAS DESIGNED INTO THE QUESTION SET AND LOST AT THE GRADING STEP.

    The frozen set is 19 graph, 14 text, 5 hybrid, deliberately, so the comparison could
    lose. But only the graph-favouring questions turned out to have mechanical answers.
    The recall column was computed on 10 questions of which SEVEN were predicted graph
    wins, and 16 of the 20 questions left ungraded were the text and hybrid ones.

    A comparison scored only on the cases one side is expected to win is a demonstration,
    not a measurement, and it is the exact failure the frozen question set was created to
    prevent. The balance has to survive into the gold sets or it was never real.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "retrieval"))
    from evaluate import ENUMERATION_CEILING
    measured = []
    for q in QUESTIONS:
        g = gold.get(q.id)
        if g is None or not g.gradable or g.expect_empty or not g.supporting:
            continue
        if len(g.supporting) > ENUMERATION_CEILING:
            continue
        measured.append(q)

    assert len(measured) >= 8, f"only {len(measured)} questions feed the recall column"

    # ⛔ A RAW COUNT IS THE WRONG TEST, and the first version of this used one. It asked
    # for at least three text-favouring questions, and the biased state it was written to
    # catch had exactly three. It passed on the defect.
    #
    # The real criterion is proportion. The frozen set chose its own balance on purpose,
    # and the measured subset has to resemble it. Anything else means the gold sets
    # silently re-weighted a question set that was designed to be able to lose.
    frozen_graph = sum(1 for q in QUESTIONS if q.expect_favours == "graph") / len(QUESTIONS)
    measured_graph = sum(1 for q in measured if q.expect_favours == "graph") / len(measured)
    drift = measured_graph - frozen_graph
    assert drift <= 0.15, (
        f"the measured questions are {measured_graph:.0%} graph-favouring against "
        f"{frozen_graph:.0%} in the frozen set, a drift of {drift:+.0%}. The recall column "
        f"is then mostly questions the graph was expected to win, and a table built from "
        f"it cannot be reported as a fair comparison. Measured: "
        f"{[(q.id, q.expect_favours) for q in measured]}")


def test_every_gold_id_exists_in_the_corpus(gold):
    """⛔ THREE TIMES NOW. Q08, then Q12, then Q20: a gold set naming things the corpus
    does not contain, so recall is structurally zero for every arm at every k and it looks
    exactly like a retrieval failure.

    Q20 was the worst of the three because it was the ONLY held out question reaching the
    recall column, so the whole "held out questions, which nobody tuned against" row of the
    report was one broken gold set printed once per arm.

    A gold id that is not in the corpus is not a hard question. It is an unanswerable one.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "retrieval"))
    from run import build_corpus
    world = World()
    corpus_ids = {c.source_id for c in build_corpus(world)}
    assert corpus_ids, "the corpus is empty, so this test checked nothing"

    for qid, g in sorted(gold.items()):
        if not g.gradable or not g.supporting:
            continue
        missing = sorted(g.supporting - corpus_ids)
        assert not missing, (
            f"{qid} has {len(missing)} gold id(s) that appear in no chunk, so no arm can "
            f"ever return them: {missing[:3]}")
