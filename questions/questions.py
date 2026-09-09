"""The question set, frozen before any retriever exists.

⛔ THE ORDER MATTERS AND IT IS THE WHOLE POINT. An audit put it plainly: if the questions
are chosen after the graph schema is designed, the comparison is unfalsifiable, because
any question the schema handles well can be promoted to "the question vector search
cannot answer". So these are written now, before a single retriever is built, and the
file is hashed. The article publishes the hash so a reader can check the order rather
than take my word for it.

Two more things the same audit demanded, both honoured here:

**Buckets, not a pile.** A single question proves nothing. Each one declares the KIND of
question it is, and the article reports a win rate per kind rather than one number. Some
kinds should favour text search and are included for that reason. A comparison that only
contains questions the graph wins is a demonstration, not a measurement.

**A held out set.** A third of these are marked `holdout=True` and are not to be looked
at while the retrievers are being built or tuned. They exist so that the reported result
can be checked against questions nobody optimised against.

⛔ NOTHING HERE MENTIONS A NODE LABEL, A RELATIONSHIP TYPE OR A CYPHER CLAUSE. These are
questions an engineer asks at three in the morning, written in the words they would use.
If a question can only be phrased in terms of the schema, it was written backwards.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

# The kinds of question, and what each one is really testing.
LOOKUP = "lookup"              # one record, findable by naming it
SEMANTIC = "semantic"          # meaning rather than words; text search should do well
MULTIHOP = "multi_hop"         # needs a chain of relationships
AGGREGATION = "aggregation"    # counting or ranking, which an index cannot do
TEMPORAL = "temporal"          # ordering in time, before and after
NEGATION = "negation"          # what is absent, which similarity cannot express


@dataclass
class Question:
    id: str
    text: str
    kind: str
    # What a correct answer must contain, in words. Not a query, and not a schema
    # reference: this is what a person would accept as right.
    answer_shape: str
    # Written before any retriever existed. An honest prediction, recorded so the
    # article can say which expectations were wrong.
    expect_favours: str
    holdout: bool = False
    notes: str = ""


QUESTIONS: list[Question] = [
    # ── lookup: text search should be at least as good as anything ────────────────
    Question("Q01", "What is the current state of INC2000042 and who was it assigned to?",
             LOOKUP, "the state and the assignment group of that one ticket",
             "text", notes="Naming a record is exactly what an index is for."),
    Question("Q02", "Show me the resolution notes for the last ticket closed on pg0071.",
             LOOKUP, "the close notes of the most recent resolved incident on that item",
             "either"),
    Question("Q03", "What does the knowledge article about clearing a full log volume say "
             "to do first?", LOOKUP, "the first step of the workaround", "text"),

    # ── semantic: the same idea in different words ────────────────────────────────
    Question("Q04", "Find tickets where the checkout journey was slow for customers, "
             "however the engineer described it.", SEMANTIC,
             "incidents about latency on customer facing services, phrased various ways",
             "text", notes="No shared keyword. This is what embeddings are for."),
    Question("Q05", "Which incidents describe something filling up or running out of room?",
             SEMANTIC, "capacity incidents, including ones that never say the word disk",
             "text"),
    Question("Q06", "Has anyone reported a problem that sounds like a certificate issue "
             "without using the word certificate?", SEMANTIC,
             "incidents whose text implies expiry or trust failure", "text",
             holdout=True),
    Question("Q07", "Find the tickets where an engineer clearly had no idea what was "
             "wrong and escalated.", SEMANTIC,
             "incidents whose work notes express uncertainty or handover", "text"),

    # ── multi hop: the chain is the answer ────────────────────────────────────────
    Question("Q08", "The payments service is down. What else stops working?", MULTIHOP,
             "every item that depends on it, directly or through another item",
             "graph", notes="The article's opening question."),
    Question("Q09", "Which business services would be affected if cluster-us-east-01 "
             "failed?", MULTIHOP, "services reachable upward from that cluster",
             "graph"),
    Question("Q10", "We are failing over a database tonight. Which teams need telling?",
             MULTIHOP, "the assignment groups owning everything downstream of it",
             "graph"),
    Question("Q11", "Three incidents are open right now. Do they share a common cause "
             "further down the stack?", MULTIHOP,
             "a shared ancestor item, or a statement that there is none", "graph",
             holdout=True),
    Question("Q12", "What does app1233 actually need in order to work?", MULTIHOP,
             "the items it depends on, in dependency order", "graph"),
    Question("Q13", "Is anything in production still depending on an item that was "
             "decommissioned?", MULTIHOP,
             "production items whose dependencies are retired", "graph", holdout=True),

    # ── temporal: before and after ────────────────────────────────────────────────
    Question("Q14", "What changed near the payments service in the day before "
             "INC2019643 was raised?", TEMPORAL,
             "changes that completed on or near that item before the incident opened",
             "graph", notes="Must exclude changes raised in response to the incident."),
    Question("Q15", "Did any change run longer than it was supposed to and get followed "
             "by an incident?", TEMPORAL,
             "changes whose actual window exceeded the planned one, with a later ticket",
             "graph"),
    Question("Q16", "Which incidents were raised outside working hours last month?",
             TEMPORAL, "incidents opened at night or at the weekend in that window",
             "graph", holdout=True),
    Question("Q17", "How long did it take to resolve the last five capacity incidents "
             "on production databases?", TEMPORAL,
             "five durations, most recent first", "graph"),

    # ── aggregation: an index cannot count ────────────────────────────────────────
    Question("Q18", "Which item has caused the most incidents this year?", AGGREGATION,
             "one item and a count", "graph",
             notes="An ANN index returns neighbours. It cannot count."),
    Question("Q19", "How many production services have no recorded dependencies at all?",
             AGGREGATION, "a number, and ideally the list", "graph"),
    Question("Q20", "Which team receives the most tickets that were not theirs to fix?",
             AGGREGATION, "an assignment group and a count", "graph", holdout=True),
    Question("Q21", "What fraction of our dependency data has not been confirmed in over "
             "a year?", AGGREGATION, "a proportion, with the count it came from",
             "graph", notes="The staleness question. Nothing else can answer it."),
    Question("Q22", "Rank the five busiest items by how many other things depend on "
             "them.", AGGREGATION, "five items in order, with degree", "graph"),

    # ── negation: absence is not a similarity ─────────────────────────────────────
    Question("Q23", "Which production services have never had an incident?", NEGATION,
             "services with no ticket at all", "graph",
             notes="Similarity search has no way to express 'none'."),
    Question("Q24", "Are there any incidents with no configuration item recorded?",
             NEGATION, "the unlinked tickets, and how many", "graph"),
    Question("Q25", "Which changes were made to items that no service depends on?",
             NEGATION, "changes on orphaned items", "graph", holdout=True),

    # ── has this happened before: the honest hybrid case ──────────────────────────
    Question("Q26", "This looks like a replication lag problem on a production database. "
             "Has it happened before, and what fixed it?", SEMANTIC,
             "an earlier incident of the same kind, its problem record, and the workaround",
             "hybrid", notes="Text finds the symptom, the graph finds the fix."),
    Question("Q27", "Somebody reported the same thing last month. Which ticket was it "
             "and what did we do?", SEMANTIC,
             "the earlier ticket and its resolution", "hybrid"),
    Question("Q28", "Is there a known error for what I am looking at?", SEMANTIC,
             "the problem record and its knowledge article, if one exists", "hybrid",
             holdout=True),

    # ── the ones designed to be hard for everything ───────────────────────────────
    Question("Q29", "Which of our recurring problems still has no permanent fix?",
             MULTIHOP, "unresolved problem records and how many incidents each groups",
             "graph"),
    Question("Q30", "If I only had time to fix one thing this quarter, what should it "
             "be?", AGGREGATION,
             "a defensible answer combining incident volume, blast radius and staleness",
             "hybrid", notes="Deliberately open. Tests whether an answer is reasoned."),
    Question("Q31", "Show me everything we know about lnx0525.", LOOKUP,
             "its attributes, its dependencies, its tickets and its changes", "hybrid"),
    # ⛔ ADDED AFTER COUNTING THE SPLIT. The first draft was 19 questions favouring the
    # graph against 7 favouring text, which is a set that decides its own result. These
    # are the questions a text index should genuinely win, and they are here so the
    # comparison can lose.
    Question("Q33", "Find the tickets where somebody pasted a stack trace about a "
             "connection pool.", SEMANTIC,
             "incidents carrying that pasted block, whatever the item", "text",
             notes="Pasted output is pure text. No relationship helps."),
    Question("Q34", "Which tickets were written by someone in a hurry, with barely any "
             "detail?", SEMANTIC, "incidents whose notes are clipped and short", "text"),
    Question("Q35", "Show me anything describing a failover that did not go to plan.",
             SEMANTIC, "incidents about failover, however worded", "text"),
    Question("Q36", "Find tickets that reference another ticket number.", LOOKUP,
             "incidents whose text names another incident", "text",
             notes="A string pattern. A graph only sees it if somebody modelled it."),
    Question("Q37", "Which incidents blame a deploy or a config change in the words of "
             "the engineer, rather than through a linked change record?", SEMANTIC,
             "incidents whose notes attribute cause to a change, with no change link",
             "text", holdout=True,
             notes="The link is absent on purpose. Only the words carry it."),
    Question("Q38", "Are there tickets about the same symptom on completely unrelated "
             "systems?", SEMANTIC,
             "incidents sharing a symptom across items with no dependency between them",
             "text", holdout=True,
             notes="A graph actively misleads here: there is no path to follow."),
    Question("Q39", "What are people actually complaining about most often, in their "
             "own words?", SEMANTIC,
             "the common themes in the free text, not a category count", "text"),
    Question("Q40", "Which incidents mention a system other than the one they were "
             "raised against?", SEMANTIC,
             "tickets whose notes name a different item", "text", holdout=True,
             notes="Text can see this. A graph built only from links cannot."),
]


def frozen_hash() -> str:
    """A digest of the questions as written, so the article can prove the order.

    Only the question text and its kind go into the hash. Notes and predictions may be
    edited later without invalidating the claim that the QUESTIONS predate the schema.
    """
    payload = json.dumps(
        [{"id": q.id, "text": q.text, "kind": q.kind, "holdout": q.holdout}
         for q in QUESTIONS],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def report() -> str:
    from collections import Counter
    kinds = Counter(q.kind for q in QUESTIONS)
    favours = Counter(q.expect_favours for q in QUESTIONS)
    held = sum(1 for q in QUESTIONS if q.holdout)
    lines = [
        f"questions          : {len(QUESTIONS)}",
        f"held out           : {held}  ({held / len(QUESTIONS):.0%}, not looked at "
        f"while building or tuning)",
        "",
        "by kind:",
        *[f"  {k:14s} {v}" for k, v in kinds.most_common()],
        "",
        "written expectation, before any retriever exists:",
        *[f"  favours {k:8s} {v}" for k, v in favours.most_common()],
        "",
        f"frozen hash        : {frozen_hash()}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
