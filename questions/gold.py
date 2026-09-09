"""The supporting records each question should retrieve, derived from the data.

⛔ THIS IS THE PART THAT DECIDES WHETHER PART 10 IS A MEASUREMENT OR AN OPINION. An audit
put it plainly: labelling the right answers after seeing what a retriever returned
invalidates the whole comparison. So the gold set is computed from the graph and the
files, by rules written down here, before any retriever exists.

Two honest limits, stated rather than hidden:

**Not every question has a mechanical answer.** "Find the tickets where an engineer
clearly had no idea what was wrong" has no query that settles it. Those questions carry
`gradable=False` and are scored by a judge whose model and prompt the article publishes,
with its agreement against a human sample reported. Mixing the two kinds silently would
let a soft grade stand in for a hard one.

**A gold set derived from the graph flatters the graph.** If the supporting records for a
multi hop question are defined by traversal, then a traversal scores perfectly by
construction. That is why the rules below use the DATASET, not the retriever's own
mechanism, and why the article reports recall separately from whether the answer was
right. A retriever can find every supporting record and still produce a wrong answer, and
that difference is worth seeing.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

DATASET = pathlib.Path(__file__).resolve().parent.parent / "dataset"

# The relationship types that carry impact. Repeated here deliberately rather than
# imported from the loader: the gold set must not move when the loader's opinion moves.
IMPACT = {"Depends on::Used by", "Runs on::Runs", "Hosted on::Hosts"}

# ⛔ THE SAME ANCHOR THE GENERATOR USED. "Last month" has to mean one fixed window or the
# gold set changes every day the article is rebuilt, and a question about time stops being
# checkable. Written out rather than imported so this file does not depend on the
# generator, which is the same reason IMPACT is repeated above.
AS_OF = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _aware(stamp: str) -> datetime:
    """Timestamps come off the files both with and without an offset. Comparing the two
    kinds raises, so everything is read as UTC."""
    d = datetime.fromisoformat(stamp)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


@dataclass
class Gold:
    question_id: str
    # Record ids a correct answer must rest on. Empty when the question is not
    # mechanically decidable.
    supporting: set[str] = field(default_factory=set)
    gradable: bool = True
    # ⛔ "NONE" IS AN ANSWER, NOT A MISSING GOLD SET. Every production service in this
    # estate has a dependency, so the correct answer to Q19 is zero. That still
    # discriminates: a retriever that returns a list has hallucinated, which is exactly
    # what a negation question is for. Without this flag the empty set check could not
    # tell a real "none" from a rule that failed to run.
    expect_empty: bool = False
    rule: str = ""
    note: str = ""
    # When a question names a KIND of thing ("the payments service") rather than one
    # record, the record it was graded against. Published, so the article can say which
    # item the number refers to.
    bound_to: str = ""


def _load(name: str) -> list[dict]:
    return [json.loads(line)
            for line in (DATASET / f"{name}.jsonl").read_text().splitlines() if line]


class World:
    """The dataset, indexed the few ways the rules below need it."""

    def __init__(self) -> None:
        self.cis = {c["key"]: c for c in _load("configuration_items")}
        self.incidents = _load("incidents")
        self.changes = _load("changes")
        self.problems = _load("problems")
        self.knowledge = _load("knowledge")

        self.by_name = {c["name"]: c for c in self.cis.values()}
        self.incidents_on: dict[str, list[dict]] = defaultdict(list)
        for inc in self.incidents:
            if inc["ci_key"]:
                self.incidents_on[inc["ci_key"]].append(inc)

        # Impact runs from the thing depended upon toward the thing that depends on it.
        self.supports: dict[str, set[str]] = defaultdict(set)
        self.depends_on: dict[str, set[str]] = defaultdict(set)
        self.degree: dict[str, int] = defaultdict(int)
        for rel in _load("relationships"):
            p, c, kind = rel["parent_key"], rel["child_key"], rel["type_name"]
            self.degree[p] += 1
            self.degree[c] += 1
            if kind in IMPACT:
                self.supports[c].add(p)
                self.depends_on[p].add(c)

    def resolve_named(self, name: str) -> str:
        """The key of the item a question names, by the name it uses.

        Questions are written in the words an engineer would use, so they say app1233,
        not app-identity-prd-1232. Grading has to start from the same item the reader
        started from, and it raises rather than returning None: a binding that silently
        goes missing is exactly how a gold set ends up describing a different record than
        the question asks about.
        """
        ci = self.by_name.get(name)
        if ci is None:
            raise KeyError(f"{name} is named in a question but not in the estate")
        return ci["key"]

    def blast_radius(self, key: str, hops: int = 4) -> set[str]:
        """What stops working if this does. Capped, because a shared cluster otherwise
        drags in a third of the estate and the answer stops meaning anything."""
        seen, frontier = set(), {key}
        for _ in range(hops):
            nxt: set[str] = set()
            for k in frontier:
                nxt |= self.supports.get(k, set())
            nxt -= seen | {key}
            if not nxt:
                break
            seen |= nxt
            frontier = nxt
        return seen


# ── the rules, one per mechanically decidable question ────────────────────────────

def build(world: World) -> dict[str, Gold]:
    gold: dict[str, Gold] = {}

    def add(qid: str, supporting, rule: str, note: str = "") -> None:
        gold[qid] = Gold(qid, supporting=set(supporting), gradable=True,
                         rule=rule, note=note)

    def ungradable(qid: str, note: str) -> None:
        gold[qid] = Gold(qid, gradable=False, note=note)

    # ⛔ THE GOLD MUST ANSWER THE QUESTION THAT WAS ASKED. A first version picked the
    # busiest service in the estate and built a blast radius for it, while Q08 asks about
    # "the payments service". The question and its answer were about different things, so
    # every arm scored zero and it looked like a retrieval failure.
    #
    # A question that names a kind of thing rather than one record has to be BOUND to a
    # record before it can be graded, and the binding is recorded so the article can say
    # which item was asked about. The question text stays exactly as it was frozen.
    # ⛔ AN APPLICATION IS ALSO A cmdb_ci_service HERE, because the estate models the
    # application layer as an application service. Filtering on the class alone bound the
    # question to app0139, which is not what "the payments service" means. The key prefix
    # is what distinguishes the two layers.
    # ⛔ AND IT MUST BE THE PRODUCTION ONE. "The payments service is down" is a production
    # sentence: nobody pages anyone about dev. Ranking by the count of DIRECT edges bound
    # this to a dev service with three, when a production one reaches sixteen items. The
    # question is about blast radius, so blast radius is what the binding ranks on.
    payments = max(
        (k for k, c in world.cis.items()
         if k.startswith("svc-") and c["domain"] == "payments"
         and c["environment"] == "prd" and world.supports.get(k)),
        key=lambda k: len(world.blast_radius(k)), default=None)
    if payments:
        gold["Q08"] = Gold(
            "Q08", supporting=world.blast_radius(payments), gradable=True,
            rule="every item reachable upward from the bound service over impact "
                 "carrying edges",
            note=f"bound to {world.cis[payments]['name']}", bound_to=payments)

    # Q12 the reverse: what an item needs in order to work.
    # ⛔ NOT THE FIRST ITEM WITH DEPENDENCIES. That was a shared cluster with 872 of
    # them, so the gold set for "what does this application need" was the entire estate
    # underneath a rack. An application depends on a handful of things, which is what
    # makes the question answerable and what makes a wrong answer visible.
    # ⛔ AND IT MUST BE THE ITEM THE QUESTION NAMES. This is the defect that made the whole
    # measurement read as broken. Q12 asks what app1233 needs; this rule picked the first
    # application with a workable number of dependencies, which was a different item
    # entirely. So the retrievers correctly fetched records about app1233 while the gold
    # correctly held the dependencies of something else, the two sets could never
    # intersect, and recall was structurally zero for every arm at every k.
    #
    # Nothing about retrieval was wrong. A question that names a record is a contract, and
    # the grader has to honour the name rather than run its own selection next to it.
    target = world.resolve_named("app1233")
    if target:
        gold["Q12"] = Gold(
            "Q12", supporting=set(world.depends_on[target]), gradable=True,
            rule="the items the application depends on, one hop over impact carrying edges",
            note=f"bound to {world.cis[target]['name']}, "
                 f"{len(world.depends_on[target])} dependencies", bound_to=target)

    # Q18 which item caused the most incidents.
    if world.incidents_on:
        worst = max(world.incidents_on, key=lambda k: len(world.incidents_on[k]))
        add("Q18", {worst}, "the item with the most incidents in the dataset",
            f"{len(world.incidents_on[worst])} incidents")

    # Q19 production services with no dependencies recorded at all.
    q19 = {k for k, c in world.cis.items()
           if c["sys_class_name"] == "cmdb_ci_service" and c["environment"] == "prd"
           and not world.depends_on.get(k)}
    gold["Q19"] = Gold(
        "Q19", supporting=q19, gradable=True, expect_empty=True,
        rule="production services with no outgoing impact edge",
        note="the answer is none: a retriever returning rows has invented them")

    # Q21 the staleness question.
    stale = [r for r in _load("relationships") if r["last_discovered"]
             and datetime.fromisoformat(r["last_discovered"])
             < datetime.now().astimezone() - timedelta(days=365)]
    gold["Q21"] = Gold(
        "Q21", gradable=True, expect_empty=True,
        rule="a proportion, checked against the manifest rather than a record set",
        note=f"{len(stale):,} edges over a year old")

    # Q22 the five busiest items.
    add("Q22", {k for k, _ in sorted(world.degree.items(), key=lambda kv: -kv[1])[:5]},
        "the five items with the highest edge count")

    # Q23 production services that have never had an incident.
    add("Q23",
        {k for k, c in world.cis.items()
         if c["sys_class_name"] == "cmdb_ci_service" and c["environment"] == "prd"
         and not world.incidents_on.get(k)},
        "production services with no incident at all")

    # Q24 incidents with no configuration item.
    add("Q24", {i["number"] for i in world.incidents if not i["ci_key"]},
        "incidents whose cmdb_ci is empty")

    # Q25 changes on items nothing depends on.
    add("Q25",
        {c["number"] for c in world.changes
         if c["ci_key"] and not world.supports.get(c["ci_key"])},
        "changes against items with no incoming impact edge")

    # Q29 problems with no permanent fix.
    add("Q29", {p["number"] for p in world.problems if not p["resolved_at"]},
        "problem records that are still open")

    # Q36 tickets that reference another ticket by number.
    add("Q36",
        {i["number"] for i in world.incidents
         if "INC" in i["description"].replace(i["number"], "")},
        "incidents whose text names a different incident")

    # Q15 changes that overran and were followed by an incident on the same item.
    overran = []
    for ch in world.changes:
        if not (ch["actual_start"] and ch["actual_end"] and ch["ci_key"]):
            continue
        actual = (datetime.fromisoformat(ch["actual_end"])
                  - datetime.fromisoformat(ch["actual_start"]))
        planned = (datetime.fromisoformat(ch["planned_end"])
                   - datetime.fromisoformat(ch["planned_start"]))
        if actual <= planned * 1.3:
            continue
        end = datetime.fromisoformat(ch["actual_end"])
        if any(end < datetime.fromisoformat(i["opened_at"]) < end + timedelta(days=2)
               for i in world.incidents_on.get(ch["ci_key"], [])):
            overran.append(ch["number"])
    add("Q15", overran,
        "changes whose actual window exceeded the planned one by a third, followed "
        "within two days by an incident on the same item")

    # ⛔ FIVE QUESTIONS WERE CALLED JUDGEMENTS FOR REASONS THAT STOPPED BEING TRUE. They
    # were written off as "chosen at run time" or "may not exist in every rebuild" back
    # when the dataset was not reproducible and there was no way to resolve a name to a
    # record. Both of those got fixed, and nobody came back to these. The measured set
    # was five questions, which is why the spread swallowed every difference and the
    # report refused to name a winner.
    #
    # A question is a judgement when its answer is a matter of opinion, not when the rule
    # is inconvenient to write.

    # Q02 the last ticket closed on a named item.
    pg = world.resolve_named("pg0071")
    closed = sorted((i for i in world.incidents_on.get(pg, []) if i.get("resolved_at")),
                    key=lambda i: i["resolved_at"])
    if closed:
        gold["Q02"] = Gold(
            "Q02", supporting={closed[-1]["number"]}, gradable=True,
            rule="the most recently resolved incident on the named item",
            note=f"bound to pg0071, {len(closed)} resolved tickets, latest is the answer",
            bound_to=pg)

    # Q09 what a shared cluster holds up. Large on purpose: this is the question whose
    # answer the direction bug erased, and it is graded as an enumeration.
    cluster = world.resolve_named("cluster-us-east-01")
    affected = {k for k in world.blast_radius(cluster) if k.startswith("svc-")}
    if affected:
        gold["Q09"] = Gold(
            "Q09", supporting=affected, gradable=True,
            rule="business services reachable upward from the named cluster over impact "
                 "carrying edges",
            note=f"bound to cluster-us-east-01, {len(affected)} services affected",
            bound_to=cluster)

    # Q16 "last month" is only ambiguous if the dataset moves. It does not: the anchor is
    # fixed at AS_OF so that every rebuild produces the same window.
    start = (AS_OF.replace(day=1) - timedelta(days=1)).replace(day=1)
    end = AS_OF.replace(day=1)

    def _out_of_hours(stamp: str) -> bool:
        d = _aware(stamp)
        return d.weekday() >= 5 or d.hour < 8 or d.hour >= 18

    q16 = {i["number"] for i in world.incidents
           if start <= _aware(i["opened_at"]) < end and _out_of_hours(i["opened_at"])}
    add("Q16", q16,
        "incidents opened at a weekend or outside 08:00 to 18:00, in the calendar month "
        f"before the fixed anchor",
        f"the month is {start:%B %Y}, fixed by AS_OF so every rebuild agrees")

    # Q17 the last five, so the gold is small and the ORDER is part of being right.
    prd_dbs = {k for k, c in world.cis.items()
               if k.startswith("db-") and c["environment"] == "prd"}
    capacity = sorted((i for i in world.incidents
                       if i.get("ci_key") in prd_dbs and i.get("resolved_at")
                       and i.get("category") == "capacity"),
                      key=lambda i: i["resolved_at"])
    if capacity:
        add("Q17", {i["number"] for i in capacity[-5:]},
            "the five most recently resolved capacity incidents on production databases",
            f"{len(capacity)} candidates, the five newest are the answer")

    # Q31 everything about one named item: the item itself, its tickets, its changes.
    lnx = world.resolve_named("lnx0525")
    everything = ({lnx}
                  | {i["number"] for i in world.incidents_on.get(lnx, [])}
                  | {c["number"] for c in world.changes if c.get("ci_key") == lnx})
    gold["Q31"] = Gold(
        "Q31", supporting=everything, gradable=True,
        rule="the item, every incident raised against it, every change made to it",
        note=f"bound to lnx0525, {len(everything)} records in total", bound_to=lnx)

    # Q20 tickets that landed on a group the item does not belong to. "Not theirs to fix"
    # is decidable here because every item carries the team that owns it, so a ticket
    # whose assignment group does not match the item's domain went to the wrong queue.
    misrouted: dict[str, int] = defaultdict(int)
    for inc in world.incidents:
        ck, group = inc.get("ci_key"), inc.get("assignment_group")
        if ck and group and ck in world.cis:
            if world.cis[ck]["domain"] not in group.lower():
                misrouted[group] += 1
    if misrouted:
        worst_group = max(sorted(misrouted), key=lambda g: misrouted[g])
        # ⛔ THE GROUP NAME IS NOT A RETRIEVABLE RECORD. The first version made the gold
        # {"platform-support"}, which is an assignment group, and no chunk in the corpus
        # carries that id. Recall was therefore structurally 0.00 for every arm at every
        # k, forever, and because Q20 was the ONLY held out question reaching the recall
        # column, the entire "held out" row of the report was that one broken gold set
        # printed four times. Any claim about generalisation drawn from it was unsupported.
        #
        # This is the same id-space mismatch already fixed for Q08 and Q12. It survived
        # twice because nothing checked that a gold id exists in the corpus.
        #
        # The evidence a retriever can actually return is the misrouted TICKETS.
        add("Q20", {i["number"] for i in world.incidents
                    if i.get("assignment_group") == worst_group
                    and i.get("ci_key") in world.cis
                    and world.cis[i["ci_key"]]["domain"] not in worst_group.lower()},
            "the incidents routed to the group that receives the most tickets for items "
            "another team owns",
            f"the group is {worst_group}, with {misrouted[worst_group]:,} such tickets "
            f"across {len(misrouted)} groups")

    # Q01 names one record. The first version called this "not worth a rule", which was a
    # judgement about effort rather than about gradability. It is a lookup question, the
    # kind text search should win, and the measured set needs those as much as it needs
    # the graph ones.
    if any(i["number"] == "INC2000042" for i in world.incidents):
        add("Q01", {"INC2000042"},
            "the one incident the question names",
            "a named record: the simplest possible retrieval, included so the set has a "
            "floor case every arm should pass")

    # ⛔ EVERY GRADABLE QUESTION FAVOURED THE GRAPH, AND THAT MADE THE WHOLE COMPARISON
    # UNFAIR AT THE GRADING STEP. The frozen set is 19 graph, 14 text, 5 hybrid, which was
    # deliberate. But only the graph ones turned out to have mechanical answers, so the
    # recall column was computed on 10 questions of which 7 were predicted graph wins, and
    # 16 of the 20 questions left ungraded were the text and hybrid ones.
    #
    # The balance was designed into the question set and then lost when the gold sets were
    # written. These two rules exist to put text-favouring questions back INTO the measured
    # column, because a comparison that can only be scored on the cases one side wins is a
    # demonstration.

    # Q04 the checkout journey was slow, however the engineer described it. Bound to one
    # service the same way Q08 is, because "the checkout journey" names a kind of thing
    # rather than a record. This is a SEMANTIC question a text index should win, and the
    # measured column needs those or it only contains cases the graph is expected to take.
    # ⛔ PRODUCTION, for the same reason Q08 is. "The checkout journey was slow for
    # customers" is about customers, and customers are not on a disaster recovery copy.
    # The first binding here picked a dr service, which is the identical mistake Q08 made
    # before it was caught.
    checkout_svcs = {k for k, c in world.cis.items()
                     if c["domain"] == "checkout" and k.startswith("svc-")
                     and c["environment"] == "prd"}
    slow = defaultdict(list)
    for inc in world.incidents:
        if inc.get("category") == "latency" and inc.get("ci_key") in checkout_svcs:
            slow[inc["ci_key"]].append(inc["number"])
    if slow:
        busiest = max(sorted(slow), key=lambda k: len(slow[k]))
        gold["Q04"] = Gold(
            "Q04", supporting=set(slow[busiest]), gradable=True,
            rule="latency incidents raised against the bound checkout service, whatever "
                 "words the engineer used",
            note=f"bound to {world.cis[busiest]['name']}, {len(slow[busiest])} incidents",
            bound_to=busiest)

    # Q26 has this happened before. ⛔ THE ANSWER IS ANY EARLIER OCCURRENCE, NOT A
    # PARTICULAR ONE. A first version picked two specific incidents out of the several
    # hundred that qualify and demanded exactly those, which no retriever could satisfy
    # and which is not what the question asks. The gold is the whole set, so it is scored
    # on precision: did what came back genuinely belong.
    prd_dbs = {k for k, c in world.cis.items()
               if k.startswith("db-") and c["environment"] == "prd"}
    repl = {i["number"] for i in world.incidents
            if i.get("category") == "replication" and i.get("ci_key") in prd_dbs}
    repl_problem = next((p for p in sorted(world.problems, key=lambda x: x["number"])
                         if p.get("category") == "replication"), None)
    if repl and repl_problem:
        add("Q26", repl | {repl_problem["number"]},
            "every replication incident on a production database, and the problem record "
            "that groups that fault. Any of them is a correct 'it happened before'",
            f"{len(repl):,} prior occurrences, so this is scored on precision")

    # Everything else is a judgement, and says so.
    for qid, why in [
        # ⛔ A FOURTH FROZEN QUESTION THE DATA CANNOT ANSWER, and this one for a different
        # reason: it is underspecified rather than unsupported. "Somebody reported the same
        # thing last month. Which ticket was it?" names no item, no symptom and no ticket.
        # There is nothing for a retriever to start from, so no arm can answer it and
        # scoring it would measure nothing. The dataset does hold 4,509 linked repeats, so
        # the fault is the wording, not the data.
        ("Q27", "names no entry point at all, so nothing identifies which earlier ticket "
                "is meant. Underspecified as frozen, reported rather than rewritten"),
        ("Q03", "the answer is a sentence from an article, graded by a judge"),
        ("Q05", "implied meaning, not a keyword"),
        ("Q06", "asks for certificate faults described WITHOUT the word. Measured: 0 of "
                "4,030 certificate incidents avoid it, so the dataset cannot answer this. "
                "Reported, not rewritten"),
        ("Q07", "uncertainty is a tone, not a field"),
        ("Q10", "needs the owning teams of a blast radius, graded on the set"),
        ("Q11", "'do they share a cause' is a judgement about three open tickets"),
        ("Q13", "the estate models no retired item, so this question cannot "
                 "discriminate between arms. Part 10 says so rather than scoring it"),
        # ⛔ THIS QUESTION CANNOT BE ANSWERED AND THE ARTICLE SAYS SO OUT LOUD. Q14 asks
        # what changed near the PAYMENTS service before INC2019643. That incident is on
        # host-pricing-stg-1068-1, a staging pricing host, which has nothing to do with
        # payments, and no change touched it in the day before it opened.
        #
        # The questions were frozen and hashed before the dataset existed, which is the
        # entire basis for claiming the comparison was not designed around its answer.
        # The price of that is that a question can name a record which turns out not to
        # fit, and the honest move is to report it rather than to edit a hashed file and
        # quietly restore a clean sweep. Part 10 lists this as a question no arm was
        # scored on, and why.
        ("Q14", "the incident it names sits on a staging pricing host, not the payments "
                "service, and nothing changed on it in the window. Frozen before the "
                "data existed, reported rather than rewritten"),
        ("Q28", "whether a known error exists is graded against what is returned"),
        ("Q30", "deliberately open, tests whether the answer is reasoned"),
        # ⛔ A SECOND QUESTION THE DATASET CANNOT ANSWER, and this one costs more than Q14.
        # Q33 asks for tickets where somebody PASTED A STACK TRACE about a connection
        # pool. Measured: "connection pool" appears in 1,035 incidents, but only ever as
        # prose inside a work note, and the dataset contains ZERO stack traces. No
        # "Traceback", no "Exception", no "stack trace" anywhere in 60,000 tickets.
        #
        # It matters more than Q14 because Q33 is one of the questions added specifically
        # so that text search could WIN. A question set that cannot grade its own
        # text-favouring cases is unbalanced in the direction that flatters the graph,
        # which is the exact failure the balance was added to prevent. Part 10 reports
        # this rather than quietly scoring around it.
        ("Q33", "asks for a pasted stack trace. The dataset has connection pool text in "
                "work notes but no stack traces at all, so the question cannot be graded "
                "mechanically. Reported, not rewritten"),
        ("Q34", "'in a hurry' is a tone, and the dataset does not model it: zero incidents "
                "have a short description and no work notes"),
        ("Q35", "'did not go to plan' is a judgement"),
        ("Q37", "attribution in words with no link, graded by a judge"),
        ("Q38", "'completely unrelated' needs both text and the absence of a path"),
        ("Q39", "themes in free text, graded by a judge"),
        ("Q40", "names an item resolved at run time"),
    ]:
        ungradable(qid, why)

    return gold


def report(gold: dict[str, Gold]) -> str:
    hard = [g for g in gold.values() if g.gradable]
    soft = [g for g in gold.values() if not g.gradable]
    empty = [g.question_id for g in hard
             if not g.supporting and not g.expect_empty]
    lines = [
        f"questions with a mechanical gold set : {len(hard)}",
        f"questions graded by a judge          : {len(soft)}",
        "",
        "supporting records per mechanical question:",
    ]
    for g in sorted(hard, key=lambda g: g.question_id):
        lines.append(f"  {g.question_id}  {len(g.supporting):>6,}  {g.rule[:58]}")
        if g.note:
            lines.append(f"        {g.note}")
    if empty:
        lines += ["", f"⛔ EMPTY GOLD SETS, WHICH WOULD SCORE EVERY RETRIEVER PERFECTLY "
                      f"OR ZERO: {empty}"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(build(World())))
