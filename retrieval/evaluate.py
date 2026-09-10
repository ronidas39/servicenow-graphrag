"""Run every arm against every question and produce the table Part 10 publishes.

The audit's objection to the original plan was that a single number per cell is not a
result: LLM answers are stochastic, the question set is small, and the difference between
two retrievers can easily be noise. So this reports a spread, states the run count, and
refuses to print a winner when the arms overlap.

Three separations that the article turns on, and that most comparisons collapse:

**Retrieval is scored apart from answering.** recall@k and MRR measure whether the right
records were found. Whether the answer was right is a different question with a different
grader. A retriever can find every supporting record and still produce a wrong answer, and
an article that reports one number cannot tell you which happened.

**Failing to run is not the same as being wrong.** A generated query that will not parse
is a reliability fact. Counting it as a miss buries it inside an accuracy number, so it is
counted and reported separately.

**Held out questions are reported on their own.** Anything tuned against the visible set
can be checked against questions nobody looked at.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import json
import pathlib
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


# ⛔ RECALL IS MEANINGLESS ONCE THE ANSWER IS BIGGER THAN THE BUDGET. Five questions
# here have gold sets of 158 to 16,788 records, while a 3,000 token budget holds about
# thirty. Recall on those is capped at 0.2 to 19 percent for EVERY arm, so a table of
# them measures the budget rather than the retriever, and the first run printed 0.00
# across the board while MRR showed hits were landing.
#
# Those questions are enumerations: "which incidents have no configuration item" is
# answered by counting, not by fetching. That is a real finding rather than an
# inconvenience, because it is a class of question no retriever answers at any k, and it
# is exactly what a query does and an index cannot. They are scored on precision, which
# asks whether what came back genuinely belongs, and reported in their own group.
ENUMERATION_CEILING = 60


@dataclass
class Score:
    arm: str
    question_id: str
    kind: str
    holdout: bool
    # Retrieval quality, against the frozen gold set. None when the question is judged
    # rather than measured, so a soft grade never silently fills a hard column.
    recall_at_k: float | None = None
    mrr: float | None = None
    precision_at_k: float | None = None
    # What it cost.
    tokens: int = 0
    latency_ms: float = 0.0
    # Reliability, kept out of the accuracy numbers.
    failed_to_run: bool = False
    error: str = ""
    # True when the gold set is larger than any budget could return, so recall is not
    # the metric and precision is reported instead.
    enumeration: bool = False
    # ⛔ WHAT AN ARM WOULD SCORE BY RETURNING DOCUMENTS AT RANDOM. Two enumeration gold
    # sets here are 12% and 20% of the whole corpus, so an arm that returns incidents at
    # all scores well on them without doing anything clever. Precision without this beside
    # it reads as skill when it is composition.
    chance_precision: float | None = None


def recall_and_mrr(retrieved: list[str], gold: set[str]) -> tuple[float, float, float]:
    """Recall, reciprocal rank and precision for one question.

    ⛔ AN EMPTY GOLD SET RAISES RATHER THAN SCORING ZERO. The one question whose correct
    answer is "none" is handled by its caller, before this, because it is graded on
    whether the arm invented rows. So an empty set arriving here means the gold set is
    broken, and the whole point of this article's first wasted evening was that a broken
    gold set scores 0.00 for every arm and reads exactly like a finding about retrieval.

    ⛔ AND THE DOCSTRING USED TO SAY IT RETURNED None WHILE THE CODE RETURNED ZEROS. The
    comment was the version I meant to write; the zeros were what ran.
    """
    if not gold:
        raise ValueError(
            "empty gold set. A question whose answer is 'none' is graded by its caller "
            "with expect_empty; anything else reaching here has a broken gold set, and "
            "scoring it 0.00 would hide that behind a number that looks like a result.")
    hits = [r for r in retrieved if r in gold]
    recall = len(set(hits)) / len(gold)
    precision = len(set(hits)) / len(retrieved) if retrieved else 0.0
    rr = 0.0
    for rank, rid in enumerate(retrieved, start=1):
        if rid in gold:
            rr = 1.0 / rank
            break
    return recall, rr, precision


def score_one(retrieved, question, gold, corpus_size: int | None = None) -> Score:
    """Turn one arm's output on one question into a row of the table."""
    s = Score(arm=retrieved.arm, question_id=question.id, kind=question.kind,
              holdout=question.holdout, tokens=retrieved.tokens,
              latency_ms=retrieved.latency_ms)
    if retrieved.error:
        s.failed_to_run = True
        s.error = retrieved.error
        return s
    if gold is None or not gold.gradable:
        # Judged elsewhere. Leaving the retrieval columns empty is the point: a judge's
        # opinion must not appear in a column labelled recall.
        return s
    if gold.expect_empty:
        # "None" is the answer. The arm is right if it returned nothing from the corpus
        # it should not have found, so precision is the only meaningful number.
        s.precision_at_k = 1.0 if not retrieved.record_ids else 0.0
        return s
    recall, mrr, precision = recall_and_mrr(retrieved.record_ids, gold.supporting)
    s.mrr, s.precision_at_k = mrr, precision
    if len(gold.supporting) > ENUMERATION_CEILING:
        # Recall left as None on purpose. Printing 0.003 next to 0.83 invites a
        # comparison between two numbers that do not mean the same thing.
        s.enumeration = True
        if corpus_size:
            s.chance_precision = len(gold.supporting) / corpus_size
    else:
        s.recall_at_k = recall
    return s


def aggregate(scores: list[Score]) -> dict:
    """Group by arm and by kind, with a spread rather than a single number."""
    by_arm: dict[str, list[Score]] = defaultdict(list)
    for s in scores:
        by_arm[s.arm].append(s)

    out: dict[str, dict] = {}
    for arm, rows in by_arm.items():
        measured = [r for r in rows if r.recall_at_k is not None]
        recalls = [r.recall_at_k for r in measured]
        out[arm] = {
            "questions": len(rows),
            "measured": len(measured),
            "failed_to_run": sum(1 for r in rows if r.failed_to_run),
            "recall_mean": statistics.fmean(recalls) if recalls else None,
            "recall_stdev": (statistics.stdev(recalls)
                             if len(recalls) > 1 else 0.0),
            "mrr_mean": (statistics.fmean([r.mrr for r in measured])
                         if measured else None),
            # ⛔ TWO TOKEN MEANS, BECAUSE ONE OF THEM IS A TRAP. An arm that declines a
            # question spends 0 tokens on it, and dividing by every question turns
            # "answered 3 of 39" into "costs 59 tokens". The bare traversal looked eight
            # times cheaper than every other arm on exactly that arithmetic, and the
            # article called it the cheap one. tokens_when_answered is what it costs to
            # answer; tokens_mean is what it costs to be asked.
            "tokens_mean": statistics.fmean([r.tokens for r in rows]) if rows else 0,
            "tokens_when_answered": (
                statistics.fmean([r.tokens for r in rows if not r.failed_to_run])
                if any(not r.failed_to_run for r in rows) else None),
            "answered": sum(1 for r in rows if not r.failed_to_run),
            "latency_p50": (statistics.median([r.latency_ms for r in rows])
                            if rows else 0),
            "latency_p95": (sorted(r.latency_ms for r in rows)[int(len(rows) * 0.95)]
                            if len(rows) > 1 else 0),
            # Kept so two arms can be compared question by question. They answer the same
            # set, so the comparison is paired, and a paired comparison needs the pairs.
            "per_question": {r.question_id: r.recall_at_k for r in measured},
            "by_kind": {},
            "holdout_recall": None,
            "enumeration_count": sum(1 for r in rows if r.enumeration),
            "enumeration_precision": (
                statistics.fmean([r.precision_at_k for r in rows
                                  if r.enumeration and r.precision_at_k is not None])
                if any(r.enumeration and r.precision_at_k is not None for r in rows)
                else None),
            "enumeration_chance": (
                statistics.fmean([r.chance_precision for r in rows
                                  if r.enumeration and r.chance_precision is not None])
                if any(r.enumeration and r.chance_precision is not None for r in rows)
                else None),
        }
        for kind in {r.kind for r in rows}:
            k = [r.recall_at_k for r in rows
                 if r.kind == kind and r.recall_at_k is not None]
            if k:
                out[arm]["by_kind"][kind] = statistics.fmean(k)
        held = [r.recall_at_k for r in rows if r.holdout and r.recall_at_k is not None]
        if held:
            out[arm]["holdout_recall"] = statistics.fmean(held)

        # ⛔ THE HELD OUT CHECK IS RUNNABLE ON PRECISION, AND THE ARTICLE SAID IT WAS NOT.
        # Section 117b reported "no held-out question has a gold set small enough to score
        # recall on", which is true and was treated as the end of it. Three of the ten
        # held-out questions are enumeration questions, and this file already scores those
        # on precision against a random baseline for the questions nobody held back. The
        # metric was there the whole time; nobody pointed it at the holdout.
        hp = [r.precision_at_k for r in rows
              if r.holdout and r.enumeration and r.precision_at_k is not None]
        hc = [r.chance_precision for r in rows
              if r.holdout and r.enumeration and r.chance_precision is not None]
        out[arm]["holdout_enumeration_count"] = len(hp)
        out[arm]["holdout_precision"] = statistics.fmean(hp) if hp else None
        out[arm]["holdout_chance"] = statistics.fmean(hc) if hc else None
    return out


def _sign_test(wins: int, losses: int) -> float:
    """Exact two sided binomial probability of a split this lopsided by chance alone.

    Written out rather than pulled from scipy so the article can show the arithmetic and
    so this file keeps no dependency a reader has to install to check the numbers.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    from math import comb
    extreme = min(wins, losses)
    tail = sum(comb(n, k) for k in range(extreme + 1))
    return min(1.0, 2 * tail / (2 ** n))


def is_a_real_difference(a: dict, b: dict) -> bool:
    """Whether two arms actually differ, compared question by question.

    ⛔ THE ARTICLE MAY NOT NAME A WINNER THIS RETURNS FALSE FOR. With a few dozen
    questions, a couple of points of recall is noise, and a table that prints two decimal
    places invites a conclusion the data cannot carry.

    ⛔ THE FIRST VERSION COMPARED THE MEAN GAP AGAINST THE POOLED SPREAD, AND THAT TEST
    COULD NEVER HAVE FIRED. Recall here is close to bimodal: an arm either finds a
    question's supporting records or finds none of them, so almost every question scores
    1.00 or 0.00 and the standard deviation sits near 0.5 whatever the arms do. An arm
    winning 0.90 against 0.10 would have been reported as "no difference", which would
    have made the article structurally unable to state its own finding. The spread was
    measuring how bimodal the questions are, not how uncertain the comparison is.

    The arms answer the SAME questions, so the comparison is paired. What matters is how
    many questions each arm wins, not how far apart two means are, and a sign test on
    those wins is the honest version of the check the first one was trying to be.
    """
    per_q = a.get("per_question") or {}
    other = b.get("per_question") or {}
    shared = [q for q in per_q if q in other]
    if not shared:
        return False
    wins = sum(1 for q in shared if per_q[q] > other[q])
    losses = sum(1 for q in shared if per_q[q] < other[q])
    if wins + losses < 3:
        # Two or fewer questions separate them. Nothing can be concluded from that,
        # however large the gap between the means looks.
        return False
    return _sign_test(wins, losses) < 0.05 and wins > losses


def report(agg: dict) -> str:
    # ⛔ EVERY MEAN CARRIES THE COUNT IT WAS TAKEN OVER. An arm that grades on 3 questions
    # and an arm that grades on 10 printed their recall in the same column with nothing
    # separating them, so 0.33 from three questions sat above 0.03 from ten and read as a
    # ranking. The "on" column is the denominator, and the two token columns are the cost
    # of answering against the cost of being asked.
    lines = [
        f"{'arm':16s} {'recall':>8s} {'on':>4s} {'±':>7s} {'MRR':>7s} "
        f"{'tok/ans':>8s} {'tok/ask':>8s} {'p50 ms':>8s} {'declined':>9s}",
        "-" * 82,
    ]
    for arm, a in sorted(agg.items(),
                         key=lambda kv: -(kv[1]["recall_mean"] or -1)):
        rc = f"{a['recall_mean']:.2f}" if a["recall_mean"] is not None else "n/a"
        sd = f"{a['recall_stdev']:.2f}" if a["recall_mean"] is not None else ""
        mr = f"{a['mrr_mean']:.2f}" if a["mrr_mean"] is not None else "n/a"
        ta = (f"{a['tokens_when_answered']:.0f}"
              if a.get("tokens_when_answered") is not None else "n/a")
        lines.append(
            f"{arm:16s} {rc:>8s} {a['measured']:>4d} {sd:>7s} {mr:>7s} "
            f"{ta:>8s} {a['tokens_mean']:>8.0f} "
            f"{a['latency_p50']:>8.0f} {a['failed_to_run']:>9d}")

    # ⛔ EVERY PAIR, NOT JUST THE TOP TWO. This used to test only the two arms with the
    # highest means. Here those two are identical, so it printed "no difference" about the
    # one comparison nobody cares about, while bm25 against vector went unreported. A
    # comparison table that hides most of its comparisons is not a comparison table.
    ranked = sorted(agg.items(), key=lambda kv: -(kv[1]["recall_mean"] or -1))
    if len(ranked) >= 2:
        lines.append("")
        lines.append("every pair, question by question:")
        lines.append(f"    {'comparison':32s} {'won':>4s} {'lost':>5s} {'tied':>5s} "
                     f"{'p':>7s}  verdict")
        separating: list[str] = []
        widest = 0
        for i, (a_name, a) in enumerate(ranked):
            for b_name, b in ranked[i + 1:]:
                pa, pb = a.get("per_question") or {}, b.get("per_question") or {}
                shared = [q for q in pa if q in pb]
                if not shared:
                    continue
                widest = max(widest, len(shared))
                wins = sum(1 for q in shared if pa[q] > pb[q])
                losses = sum(1 for q in shared if pa[q] < pb[q])
                p = _sign_test(wins, losses)
                real = is_a_real_difference(a, b)
                verdict = "SEPARATES" if real else "no"
                if real:
                    separating.append(f"{a_name} over {b_name}")
                lines.append(
                    f"    {a_name + ' vs ' + b_name:32s} {wins:>4d} {losses:>5d} "
                    f"{len(shared) - wins - losses:>5d} {p:>7.3f}  {verdict}")

        # ⛔ THIS PARAGRAPH USED TO SAY "no pair above separates" WHATEVER THE TABLE SAID.
        # It happened to be true every time it was printed, which is the only reason it
        # survived: an assertion that agrees with the data by luck reads exactly like a
        # measurement, right up to the run where it does not.
        lines.append("")
        if widest:
            # The smallest p any pair here could reach is a clean sweep of the widest
            # comparison, so it says what the question set is capable of showing.
            floor = _sign_test(widest, 0)
            lines.append(f"  A sign test needs six one-way wins for p < 0.05. The widest "
                         f"comparison above")
            lines.append(f"  covers {widest} question(s), so the smallest p reachable at "
                         f"all is {floor:.3f}.")
        if separating:
            lines.append(f"  {len(separating)} pair(s) separate: "
                         f"{', '.join(separating)}. Those, and only those, may be")
            lines.append(f"  named as a result.")
        else:
            lines.append(f"  No pair above separates, so the article must not name a "
                         f"winner from this table.")

    lines.append("")
    lines.append("held out questions only, which nobody tuned against:")
    # ⛔ SAY SO WHEN THE CHECK DID NOT RUN. This section used to print nothing at all when
    # no held out question reached the recall column, which reads as "there was nothing to
    # report" rather than "this was not measured". Worse, before Q20's gold was fixed it
    # printed 0.00 for every arm from ONE broken gold set, and that looked like a finding
    # about generalisation.
    # ── the held out check, on the metric that can carry it ──────────────────────
    if any(a.get("holdout_precision") is not None for a in agg.values()):
        lines.append("")
        lines.append("held out questions, scored on precision because their gold sets are "
                     "too large for recall:")
        n_held = max(a.get("holdout_enumeration_count", 0) for a in agg.values())
        lines.append(f"  {n_held} of them, none of which any retriever was tuned against")
        for arm, a in sorted(agg.items(),
                             key=lambda kv: -(kv[1].get("holdout_precision") or -1)):
            if a.get("holdout_precision") is None:
                continue
            seen = a.get("enumeration_precision")
            drift = (f"  against {seen:.2f} on the questions it was tuned with"
                     if seen is not None else "")
            lines.append(f"  {arm:16s} precision {a['holdout_precision']:.2f}"
                         f"  chance {a['holdout_chance']:.2f}{drift}")

    if all(a["holdout_recall"] is None for a in agg.values()):
        lines.append("  ⛔ NOT MEASURED ON RECALL. No held out question has a gold set")
        lines.append("     small enough to score recall on, so the recall column has no")
        lines.append("     held out row. The precision check above is the one that runs.")
    for arm, a in sorted(agg.items()):
        if a["holdout_recall"] is not None:
            drop = ""
            if a["recall_mean"]:
                delta = a["holdout_recall"] - a["recall_mean"]
                drop = f"  ({delta:+.2f} against the visible set)"
            lines.append(f"  {arm:16s} {a['holdout_recall']:.2f}{drop}")

    lines.append("")
    lines.append("enumeration questions, where the answer is larger than any budget:")
    lines.append("  scored on precision, because recall is capped by the budget for "
                 "every arm alike")
    chance = next((a["enumeration_chance"] for a in agg.values()
                   if a.get("enumeration_chance") is not None), None)
    if chance is not None:
        lines.append(f"  {'RANDOM BASELINE':16s} precision {chance:.2f}   "
                     f"an arm returning documents at random scores this")
    for arm, a in sorted(agg.items()):
        if a.get("enumeration_precision") is not None:
            got = a["enumeration_precision"]
            mark = ""
            if chance:
                mark = ("  at chance" if got <= chance * 1.5
                        else f"  {got / chance:.1f}x chance")
            lines.append(f"  {arm:16s} precision {got:.2f} "
                         f"over {a['enumeration_count']} questions{mark}")

    lines.append("")
    lines.append("by kind of question:")
    kinds = sorted({k for a in agg.values() for k in a["by_kind"]})
    header = "  " + " " * 16 + "".join(f"{k[:9]:>11s}" for k in kinds)
    lines.append(header)
    for arm, a in sorted(agg.items()):
        row = "".join(
            f"{a['by_kind'].get(k, float('nan')):>11.2f}" if k in a["by_kind"]
            else f"{'-':>11s}" for k in kinds)
        lines.append(f"  {arm:16s}{row}")
    return "\n".join(lines)


def save(scores: list[Score], agg: dict, meta: dict) -> pathlib.Path:
    """Write the raw scores as well as the summary.

    The summary is what the article prints. The raw rows are what somebody checks it
    against, so they ship together.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "scores.json"
    path.write_text(json.dumps({
        "meta": meta,
        "summary": agg,
        "scores": [asdict(s) for s in scores],
    }, indent=2, default=str) + "\n")
    return path
