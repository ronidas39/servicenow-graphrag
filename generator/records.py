"""Generate changes, problems and knowledge articles.

Three record types the first draft of this dataset did not have, and each one was
missing for a reason worth stating.

Changes are what makes "what changed near this" answerable, and that is the highest
value question in incident triage. It is also the easiest one to get wrong: a change
carries a planned window and an actual window, and they are not the same. Emergency
changes are frequently raised after the outage they belong to, so a naive time window
returns the change that exists BECAUSE of the incident and reports it as the cause.
The generator produces that trap on purpose, at a stated rate, so the article can teach
how to avoid it.

Problems and knowledge articles are how a real service desk finds a past fix. The
article promises to answer "has anyone solved this before" and resolved incidents alone
are not how that is done in practice.

Author: Roni Das
Created: 2026-09-08
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from estate import AS_OF, Estate
from incidents import CAUSES, Incident

CHANGE_TYPES = ["standard", "normal", "emergency"]
# Most changes are routine and pre approved. Emergencies are rare and are the ones that
# break naive correlation.
# Emergencies are rare. Most of the few that exist are reactive, so the reactive share
# below supplies almost all of them and the base weight here stays small. Setting both
# high once produced 20% emergency changes, which no change board would recognise.
CHANGE_TYPE_WEIGHTS = [0.50, 0.48, 0.02]

CHANGE_WORK = [
    "Upgrade {ci} to the current patch level",
    "Resize the volume on {ci}",
    "Roll out the new connection pool settings to {ci}",
    "Replace the expiring certificate on {ci}",
    "Add an index to the main table on {ci}",
    "Increase the memory limit for {ci}",
    "Fail {ci} over to the secondary and back",
    "Apply the vendor hotfix to {ci}",
    "Rotate the signing key used by {ci}",
    "Move {ci} behind the new load balancer",
]

CLOSE_CODES = ["successful", "successful with issues", "unsuccessful"]
CLOSE_WEIGHTS = [0.81, 0.14, 0.05]


@dataclass
class Change:
    number: str
    short_description: str
    description: str
    ci_key: str
    change_type: str
    risk: str
    opened_at: datetime
    planned_start: datetime
    planned_end: datetime
    actual_start: datetime | None
    actual_end: datetime | None
    close_code: str
    assigned_to: str
    # Set when this change was raised in response to an incident rather than before it.
    # These are the ones that look like a cause and are actually an effect.
    raised_after_incident: str | None = None


@dataclass
class Problem:
    number: str
    short_description: str
    description: str
    ci_key: str | None
    category: str
    opened_at: datetime
    resolved_at: datetime | None
    workaround: str
    cause_notes: str
    incident_numbers: list[str] = field(default_factory=list)


@dataclass
class KnowledgeArticle:
    number: str
    # Which problem this documents. Without it the only join available is the category,
    # which matches every article in that family and returns one incident many times.
    problem_number: str
    short_description: str
    text: str
    category: str
    published_at: datetime
    views: int


def generate_changes(
    estate: Estate,
    incidents: list[Incident],
    count: int = 8_000,
    seed: int = 20260908,
    reactive_rate: float = 0.06,
    overrun_rate: float = 0.22,
) -> list[Change]:
    """Generate change requests.

    Args:
        reactive_rate: share raised in response to an incident, after it opened. These
            are the reverse causation trap. A time window that ignores them will report
            the fix as the cause of the fault.
        overrun_rate: share whose actual window differs materially from the planned one.
            Matching on planned dates alone misses these entirely.
    """
    rng = random.Random(seed + 1)
    targets = [c for c in estate.cis
               if c.sys_class_name.endswith(("_appl", "_db_instance", "_linux_server",
                                             "_win_server", "_lb"))]
    people = [f"user{n:04d}" for n in range(1, 2001)]
    out: list[Change] = []

    for i in range(count):
        ctype = rng.choices(CHANGE_TYPES, weights=CHANGE_TYPE_WEIGHTS, k=1)[0]
        # reactive_rate is the share of ALL changes raised in response to an incident,
        # so it is applied directly. A first version derived it from the emergency
        # weight instead, which meant the argument did nothing and the real rate was
        # whatever the weights happened to produce.
        reactive = bool(incidents) and rng.random() < reactive_rate
        if reactive:
            ctype = "emergency"

        if reactive:
            # ⛔ THE CHANGE MUST BE ON THE SAME ITEM AS THE INCIDENT. A first version
            # picked the item at random, so a query that filtered changes by
            # configuration item never saw these at all and the trap never fired. The
            # whole point is that it sits right next to the fault in both time and place.
            candidates = [x for x in incidents if x.ci_key]
            parent = candidates[rng.randrange(len(candidates))]
            ci = next(c for c in estate.cis if c.key == parent.ci_key)
            opened = parent.opened_at + timedelta(minutes=rng.randint(20, 400))
            after = parent.number
        else:
            ci = rng.choice(targets)
            # ⛔ CLAMP THE OPENING TIME, NOT ONLY WHAT IS DERIVED FROM IT. A first fix
            # clamped the planned and actual windows and left this line alone, so 408
            # changes still opened in the future and dragged their windows with them.
            # Backwards only: a change is raised before the work, never after today.
            base = (incidents[rng.randrange(len(incidents))].opened_at
                    if incidents else AS_OF)
            opened = min(base + timedelta(days=rng.randint(-120, 30)),
                         AS_OF - timedelta(days=1))
            after = None

        # ⛔ NOTHING IS DATED AFTER NOW. The opening time can already be recent, and a
        # normal change adds a week of lead time on top, so 431 changes were scheduled
        # to start in the future. Incidents were clamped and changes were not, which is
        # the kind of half fix that survives because nobody measures the other file.
        now = AS_OF
        lead = {"standard": 1, "normal": 7, "emergency": 0}[ctype]
        planned_start = min(opened + timedelta(days=lead, hours=rng.randint(0, 20)), now)
        planned_end = min(planned_start + timedelta(hours=rng.choice([1, 2, 2, 4, 6])), now)

        if reactive or rng.random() < 0.94:
            drift = timedelta(minutes=rng.randint(-30, 45))
            actual_start = min(planned_start + drift, now)
            span = planned_end - planned_start
            if rng.random() < overrun_rate:
                span = span * rng.uniform(1.4, 3.2)
            actual_end = min(actual_start + span, now)
        else:
            actual_start = actual_end = None

        work = rng.choice(CHANGE_WORK).format(ci=ci.name)
        out.append(Change(
            number=f"CHG{100_000 + i}",
            short_description=work,
            description=(
                f"{work}.\n\n"
                f"Environment {ci.environment}, region {ci.region}. "
                f"{'Raised in response to ' + after + '.' if after else 'Planned work.'}\n\n"
                f"Backout: revert to the previous configuration and confirm the service "
                f"responds before handing back."
            ),
            ci_key=ci.key,
            change_type=ctype,
            risk=rng.choices(["low", "moderate", "high"], weights=[0.62, 0.30, 0.08], k=1)[0],
            opened_at=opened,
            planned_start=planned_start,
            planned_end=planned_end,
            actual_start=actual_start,
            actual_end=actual_end,
            close_code=rng.choices(CLOSE_CODES, weights=CLOSE_WEIGHTS, k=1)[0],
            assigned_to=rng.choice(people),
            raised_after_incident=after,
        ))

    out.sort(key=lambda c: c.opened_at)
    return out


def generate_problems(
    incidents: list[Incident],
    count: int = 900,
    seed: int = 20260908,
) -> list[Problem]:
    """Generate problem records, each grouping several incidents of the same kind.

    This is how a service desk records a known fault. Without it, "has this happened
    before" has nothing to search except the incidents themselves.
    """
    rng = random.Random(seed + 2)

    # ⛔ A PROBLEM IS ONE RECURRING FAULT, NOT A CATEGORY. Sampling a whole category
    # across 540 days and every region produced problems spanning a median of five
    # unrelated items, titled after only the first, with a root cause drawn independently
    # of anything its own members said. A traversal from an incident to its problem and
    # back returned unrelated tickets, which starves the graph side of the comparison.
    #
    # Grouping on the item AND the failure kind is what a service desk actually does, and
    # the generator now produces those clusters naturally because chronic faults recur.
    clusters: dict[tuple[str, str], list[Incident]] = {}
    for inc in incidents:
        if inc.ci_key:
            clusters.setdefault((inc.ci_key, inc.category), []).append(inc)
    usable = [members for members in clusters.values() if len(members) >= 2]
    usable.sort(key=len, reverse=True)
    if not usable:
        return []

    out: list[Problem] = []
    for i in range(min(count, len(usable))):
        members = sorted(usable[i], key=lambda x: x.opened_at)
        first = members[0]
        cat = first.category
        cause = rng.choice(CAUSES[cat])
        # Every member shares the item, so the title can name it truthfully.
        out.append(Problem(
            number=f"PRB{40_000 + i}",
            short_description=f"Recurring {cat} fault on {first.ci_key}",
            description=(
                f"Seen on {len(members)} incidents since {first.opened_at:%B %Y}, "
                f"all on the same item. Investigation found that {cause}."
            ),
            ci_key=first.ci_key,
            category=cat,
            opened_at=min(first.opened_at + timedelta(days=rng.randint(1, 30)),
                          AS_OF),
            resolved_at=(min(members[-1].opened_at + timedelta(days=rng.randint(2, 60)),
                             AS_OF)
                         if rng.random() < 0.64 else None),
            workaround=rng.choice([
                "Restart the affected process and monitor for an hour.",
                "Fail over to the secondary while the primary is rebuilt.",
                "Increase the limit temporarily and schedule the permanent fix.",
                "Disable the feature flag until the fix is released.",
            ]),
            cause_notes=f"Root cause: {cause}.",
            incident_numbers=[m.number for m in members],
        ))
    return out


def generate_knowledge(
    problems: list[Problem],
    seed: int = 20260908,
) -> list[KnowledgeArticle]:
    """Turn resolved problems into knowledge articles.

    A knowledge article is the thing an analyst actually finds when they search for a
    past fix, and it is written for a reader rather than for a ticket queue.
    """
    rng = random.Random(seed + 3)
    out: list[KnowledgeArticle] = []
    # A problem resolved right on the snapshot edge has no room for an article to be
    # published after it, and an article that predates its own resolution is nonsense.
    eligible = [p for p in problems
                if p.resolved_at and p.resolved_at < AS_OF - timedelta(days=22)]
    for n, prb in enumerate(eligible):
        out.append(KnowledgeArticle(
            number=f"KB{9_000 + n}",
            problem_number=prb.number,
            short_description=f"How to handle {prb.category} faults on this platform",
            text=(
                f"## Symptom\n\n{prb.short_description}.\n\n"
                f"## What causes it\n\n{prb.cause_notes}\n\n"
                f"## What to do now\n\n{prb.workaround}\n\n"
                f"## Permanent fix\n\nSee {prb.number}. Raise a change against the "
                f"affected item and apply the fix during the next window.\n\n"
                f"## Related incidents\n\n"
                + ", ".join(prb.incident_numbers[:6])
            ),
            category=prb.category,
            published_at=min(prb.resolved_at + timedelta(days=rng.randint(1, 21)),
                             AS_OF),
            views=int(rng.lognormvariate(4.2, 1.1)),
        ))
    return out


def report(changes: list[Change], problems: list[Problem],
           knowledge: list[KnowledgeArticle]) -> str:
    from collections import Counter

    n = len(changes)
    reactive = sum(1 for c in changes if c.raised_after_incident)
    noactual = sum(1 for c in changes if c.actual_start is None)
    overran = sum(
        1 for c in changes
        if c.actual_end and c.actual_end - c.actual_start > (c.planned_end - c.planned_start) * 1.3
    )
    solved = sum(1 for p in problems if p.resolved_at)
    return "\n".join([
        f"changes              : {n:,}",
        *[f"  {k:12s} {v:>6,} ({v / n:.0%})"
          for k, v in Counter(c.change_type for c in changes).most_common()],
        "",
        "the traps this data carries on purpose:",
        f"  raised AFTER the incident they look like they caused : {reactive:,} ({reactive / n:.0%})",
        f"  ran longer than planned, so planned dates mislead     : {overran:,} ({overran / n:.0%})",
        f"  never started, so actual dates are empty              : {noactual:,} ({noactual / n:.0%})",
        "",
        f"problems             : {len(problems):,}  ({solved:,} resolved)",
        f"knowledge articles   : {len(knowledge):,}",
        f"incidents grouped under a problem : "
        f"{sum(len(p.incident_numbers) for p in problems):,}",
    ])


if __name__ == "__main__":
    from estate import build_estate
    from incidents import generate as gen_incidents

    est = build_estate(services=2200)
    inc = gen_incidents(est, count=60_000)
    chg = generate_changes(est, inc)
    prb = generate_problems(inc)
    kb = generate_knowledge(prb)
    print(report(chg, prb, kb))
    print("\n" + "─" * 70 + "\n  A CHANGE THAT LOOKS LIKE A CAUSE AND IS AN EFFECT\n" + "─" * 70)
    c = next(x for x in chg if x.raised_after_incident)
    print(f"{c.number}  {c.change_type}  {c.short_description}")
    print(f"  opened {c.opened_at:%Y-%m-%d %H:%M}, after {c.raised_after_incident}")
    print(f"  planned {c.planned_start:%Y-%m-%d %H:%M} to {c.planned_end:%H:%M}")
    print(f"  actual  {c.actual_start:%Y-%m-%d %H:%M} to {c.actual_end:%H:%M}"
          if c.actual_start else "  actual  never started")
