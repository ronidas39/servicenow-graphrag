"""Generate incidents whose text a search engine can actually work with.

This is the file the whole comparison rests on. If the ticket text is eight rotating
phrases with no body, then similarity search has nothing to match and it loses the
comparison for a reason that is my fault rather than a real limitation. A previous
attempt did exactly that: eight hundred incidents carried eight distinct sentences and
not one description, which is why none of it survived.

Real analyst writing has properties a naive generator does not reproduce, and each one
is a lexical signal that similarity search legitimately exploits:

  · the same failure recurs, so near duplicate tickets exist
  · analysts reference other tickets by number
  · people paste the same stack trace and the same command output between tickets
  · notes are written in a hurry, so they are short, clipped and inconsistent
  · a ticket passes between teams and each one writes in a different register

Leave those out and similarity search is handicapped before the first question is asked.
So they are put in on purpose, at rates recorded here, and the article reports them.

⛔ The narrative text here is a template assembly, not a language model. That is
deliberate: it is reproducible, it costs nothing to regenerate, and the article can show
exactly how each sentence was built. The real wording variety comes from the number of
independent slots, not from a model's imagination.

Author: Roni Das
Created: 2026-09-08
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from estate import AS_OF, CI, Estate

# A symptom is what the person noticed. A cause is what it turned out to be. Keeping
# them separate is what stops every ticket reading the same, because the interesting
# text is the distance between the two.
SYMPTOMS: list[tuple[str, str]] = [
    ("checkout is timing out for some customers", "latency"),
    ("users report the page hangs then returns a blank error", "latency"),
    ("api returning 502 intermittently", "errors"),
    ("error rate above threshold on the payments endpoint", "errors"),
    ("batch job did not finish inside its window", "capacity"),
    ("queue is growing and not draining", "capacity"),
    ("disk on the primary is nearly full", "capacity"),
    ("replication lag climbing since this morning", "replication"),
    ("failover did not complete cleanly", "replication"),
    ("login fails for a subset of users", "auth"),
    ("certificate warning in the browser", "certificate"),
    ("scheduled report did not arrive", "batch"),
    ("connections refused during the evening peak", "capacity"),
    ("slow queries piling up on the read replica", "database"),
    ("memory climbing steadily since the deploy", "memory"),
]

CAUSES: dict[str, list[str]] = {
    "latency": [
        "a missing index meant a table scan on every request",
        "the connection pool was sized for the old traffic level",
        "an upstream call had no timeout, so threads were held open",
        "a cache was evicting far more than expected after the config change",
    ],
    "errors": [
        "a dependency was returning 500s and the retry made it worse",
        "a bad node stayed in the load balancer pool after a failed health check",
        "the deploy rolled out with the wrong environment variable",
        "a schema change went out ahead of the code that needed it",
    ],
    "capacity": [
        "log rotation had been failing quietly for weeks",
        "a consumer group had stopped without alerting",
        "the volume was never resized after the last growth in traffic",
        "a retry storm from one client saturated the workers",
    ],
    "replication": [
        "a long running transaction blocked the replica from catching up",
        "network between the two regions degraded for about forty minutes",
        "the replica ran out of disk and stopped applying changes",
    ],
    "auth": [
        "the token signing key had been rotated on one node only",
        "a group membership sync had not run since the weekend",
        "clock drift on one host put tokens outside their validity window",
    ],
    "certificate": [
        "the certificate expired and renewal had never been automated",
        "the intermediate chain was missing from one of the servers",
    ],
    "batch": [
        "the upstream file arrived late and the job had already given up",
        "a dependency job failed silently and this one waited forever",
    ],
    "database": [
        "statistics were stale so the planner chose a bad path",
        "an index was dropped during the migration and never recreated",
    ],
    "memory": [
        "a cache had no upper bound, so it grew until the process was killed",
        "a library upgrade changed the default pool size",
    ],
}

# Pasted material, keyed to the kind of failure. Real tickets are full of it, and the
# same block turns up in several tickets because people copy from the last one they
# solved. THE BLOCK MUST MATCH THE SYMPTOM. A first version chose these at random and
# produced a memory ticket carrying a replication lag warning, which tells any engineer
# reading it that the data is invented.
PASTED: dict[str, list[str]] = {
    "latency": [
        "```\nERROR  pool timeout: could not acquire connection within 30000ms\n"
        "  at Pool.acquire (pool.js:118)\n  at Handler.query (db.js:44)\n```",
        "```\np99 latency 8420ms (threshold 800ms) over the last 5m\n```",
    ],
    "errors": [
        "```\nupstream connect error or disconnect/reset before headers. "
        "reset reason: connection termination\n```",
        "```\nHTTP 502 rate 4.1% over 5m, was 0.02% before 14:10\n```",
    ],
    "capacity": [
        "```\n$ df -h /var\nFilesystem      Size  Used Avail Use% Mounted on\n"
        "/dev/nvme1n1    100G   97G  2.1G  98% /var\n```",
        "```\nconsumer lag 1284006 and rising, 0 active consumers\n```",
    ],
    "replication": [
        "```\nWARN  replication lag 428s (threshold 60s)\n```",
        "```\nSlave_IO_Running: Yes\nSlave_SQL_Running: No\n"
        "Last_Error: Disk full writing relay log\n```",
    ],
    "auth": [
        "```\ninvalid signature: kid 7f2a not found in JWKS\n```",
        "```\ntoken used before issued (iat 14:22:09, now 14:21:51)\n```",
    ],
    "certificate": [
        "```\nx509: certificate has expired or is not yet valid\n```",
        "```\nunable to get local issuer certificate\n```",
    ],
    "batch": [
        "```\njob nightly-settlement exited 1 after 04:12:33 (window 04:00)\n```",
    ],
    "database": [
        "```\nSeq Scan on transactions  (cost=0.00..1842301.00 rows=41 width=88)\n"
        "  Filter: (account_id = $1)\n```",
    ],
    "memory": [
        "```\n$ kubectl get pods | grep -v Running\nNAME                    READY   "
        "STATUS             RESTARTS\nworker-7d9f8c6b4-2xqk9  0/1     "
        "CrashLoopBackOff   14\n```",
        "```\nOOMKilled  memory.limit 2Gi  memory.peak 2Gi\n```",
    ],
}

# How an analyst actually writes at two in the morning.
CLIPPED: list[str] = [
    "looking now",
    "same as last week",
    "not seeing it from my side",
    "can repro",
    "escalating, out of my depth here",
    "handing over, see notes",
    "restarted, watching",
    "back to normal for now, keeping open",
    "raised with the vendor",
    "waiting on change approval",
]

CLOSE_NOTES: list[str] = [
    "Applied the fix and monitored for an hour. No recurrence.",
    "Resolved after the change went in. Closing.",
    "Root cause confirmed. Permanent fix tracked separately.",
    "Cleared after restart. Underlying cause still open as a problem record.",
    "Config corrected and rolled out to the rest of the fleet.",
]


@dataclass
class Incident:
    number: str
    short_description: str
    description: str
    ci_key: str | None
    category: str
    priority: int
    opened_at: datetime
    resolved_at: datetime | None
    assignment_group: str
    caller: str
    close_notes: str | None
    work_notes: list[tuple[datetime, str, str]] = field(default_factory=list)
    # Set when this ticket is a repeat of an earlier one. The article reports the rate.
    duplicate_of: str | None = None


def _priority(rng: random.Random) -> int:
    """Real queues are mostly low priority. A generator that spreads them evenly
    produces an estate where everything is an emergency, which flatters nothing."""
    return rng.choices([1, 2, 3, 4], weights=[0.02, 0.11, 0.48, 0.39], k=1)[0]


def _clamp(when: datetime) -> datetime:
    """Nothing in a ticket system is dated after now. 154 incidents used to resolve in
    the future, the worst by a month, because the resolution time was added to an
    opening time that could already be yesterday."""
    return min(when, AS_OF)


def _opened(rng: random.Random, start: datetime, days: int) -> datetime:
    """Tickets cluster in office hours, with a thin overnight tail from monitoring."""
    # ⛔ RESAMPLE THE WEEKDAY, DO NOT SHIFT IT. Moving a weekend date back one or two
    # days leaves Sunday on Saturday and piles both onto Thursday and Friday. Measured:
    # Mon 14.6, Tue 14.3, Wed 14.3, Thu 19.3, Fri 24.4, Sat 9.0, Sun 4.0. Real service
    # desk inflow peaks on MONDAY, because the weekend's problems are reported then.
    for _ in range(12):
        day = start + timedelta(days=rng.randint(0, days - 1))
        wd = day.weekday()
        keep = [0.215, 0.19, 0.18, 0.175, 0.16, 0.045, 0.035][wd]
        if rng.random() < keep / 0.215:
            break
    if rng.random() < 0.78:
        hour = int(rng.triangular(7, 19, 11))
    else:
        hour = rng.choice([0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23])
    return day.replace(hour=hour, minute=rng.randint(0, 59), second=rng.randint(0, 59))


def _mttr(rng: random.Random, priority: int) -> timedelta:
    """Resolution time is long tailed. A few tickets sit open for weeks."""
    base = {1: 1.2, 2: 4.0, 3: 20.0, 4: 60.0}[priority]
    hours = rng.lognormvariate(0, 0.9) * base
    return timedelta(hours=min(hours, 24 * 45))


def generate(
    estate: Estate,
    count: int = 60_000,
    days: int = 540,
    seed: int = 20260908,
    duplicate_rate: float = 0.14,
    cross_reference_rate: float = 0.22,
    paste_rate: float = 0.35,
    unlinked_rate: float = 0.17,
    neighbour_mention_rate: float = 0.26,
    chronic_rate: float = 0.45,
) -> list[Incident]:
    """Generate the incident set.

    Args:
        estate: the infrastructure these tickets are raised against.
        count: how many incidents.
        days: how far back the history runs.
        duplicate_rate: share that are a repeat of an earlier ticket on the same item.
            Repeat failures are normal and they are what makes "has this happened
            before" a real question.
        cross_reference_rate: share whose notes name another ticket by number. This is
            the strongest lexical signal in real ITSM text.
        paste_rate: share carrying a pasted block of output.
        unlinked_rate: share with no configuration item set. On a real instance this is
            common, and a graph that assumes the link is always there will miss them.
        chronic_rate: share of tickets raised against a fault that has already been
            seen on that item. Without it a repeat almost never has a predecessor to
            point at, because a random pairing rarely lands twice.
        neighbour_mention_rate: share whose work notes name an item ONE HOP away in the
            dependency graph. ⛔ This exists to keep the comparison honest. Without it
            no ticket ever mentions anything but its own item, so a multi hop question
            has zero textual evidence and the graph wins by construction rather than by
            being better. Real analysts write "checked lnx0341, same window" constantly.
            One hop is mentioned, two hops almost never, which is the decay a text
            baseline should genuinely suffer.

    Returns:
        The incidents, in the order they were opened.
    """
    rng = random.Random(seed)
    # One hop neighbours, so a note can mention what sits next to the failing item.
    neighbours: dict[str, list[str]] = {}
    names = {c.key: c.name for c in estate.cis}
    for rel in estate.rels:
        neighbours.setdefault(rel.parent_key, []).append(rel.child_key)
        neighbours.setdefault(rel.child_key, []).append(rel.parent_key)

    # ⛔ EVERY CLASS GETS TICKETS. An endswith filter admitted only services and the
    # two server classes, so databases, load balancers, clusters and storage had ZERO
    # incidents between them. The cause library is full of replication and disk
    # material, so "the replica ran out of disk" always landed on an application, and
    # the shared clusters the article warns about had no text history at all.
    targets: list[CI] = list(estate.cis)
    # ⛔ A SET OF STRINGS DOES NOT ITERATE IN A STABLE ORDER. Python randomises string
    # hashing per process, so this list came out shuffled differently on every run and
    # every assignment group in the file changed, which is why two builds from the same
    # seed produced different bytes. Sorting is the whole fix, and nothing about the
    # seed or the generator was at fault.
    groups = sorted(f"{d}-support" for d in {c.domain for c in estate.cis})
    people = [f"user{n:04d}" for n in range(1, 2001)]
    start = AS_OF - timedelta(days=days)

    # ⛔ TIMESTAMPS ARE ASSIGNED BEFORE ANYTHING REFERENCES ANYTHING ELSE, and the list
    # is put in time order before the second pass. A first version tracked "the last
    # ticket on this item" in loop order while the opening times were random, then
    # sorted at the end. 263 of 6000 repeats therefore pointed at a ticket that opened
    # LATER than them, which is nonsense and would have broken the article's "has this
    # happened before" question. A unit test caught it; reading samples by eye had not.
    # ⛔ REPEATS HAVE TO ACTUALLY RECUR. Once a repeat was required to be the same
    # FAULT on the same item, not merely the same item, the rate collapsed to 2%,
    # because a random pairing almost never lands on a combination already seen. Real
    # estates have chronic faults: the same box, the same failure, again next month.
    # So a share of tickets deliberately reuses a pairing that has already happened,
    # which is what makes "has this happened before" a real question.
    draft: list[tuple[datetime, str, str, CI, int]] = []
    seen_pairs: list[tuple[str, str, CI]] = []
    for i in range(count):
        if seen_pairs and rng.random() < chronic_rate:
            symptom, family, ci = rng.choice(seen_pairs)
        else:
            symptom, family = rng.choice(SYMPTOMS)
            ci = rng.choice(targets)
            seen_pairs.append((symptom, family, ci))
            if len(seen_pairs) > 4000:
                seen_pairs.pop(rng.randrange(len(seen_pairs)))
        draft.append((_opened(rng, start, days), symptom, family, ci, _priority(rng)))
    draft.sort(key=lambda d: d[0])

    out: list[Incident] = []
    # Keeps the last ticket seen per item, walked in time order so a repeat can only
    # ever point backwards.
    last_on_ci: dict[tuple[str, str], str] = {}
    # Recent tickets per failure kind, so a cross reference points at something that
    # actually looked similar rather than at any ticket that happened to exist.
    related: dict[str, list[str]] = {}

    for i, (opened, symptom, family, ci, priority) in enumerate(draft):
        number = f"INC{2_000_000 + i}"

        # ⛔ A REPEAT IS THE SAME FAULT, NOT JUST THE SAME BOX. Keying only on the
        # item meant 86% of repeats pointed at a ticket in a different category: a
        # memory fault claiming to be the same thing as a replication fault. The
        # article's flagship question had a 14% precision ceiling on the very edge that
        # is supposed to answer it.
        is_dup = rng.random() < duplicate_rate and (ci.key, family) in last_on_ci
        cause = rng.choice(CAUSES[family])

        # ⛔ DECIDE THE LINK BEFORE WRITING THE WORDS. A first version nulled ci_key
        # AFTER the name had already been printed into both the title and the body, so
        # every one of the 17% "unlinked" tickets still named its item in plain text and
        # the link was recoverable by exact match in one query. The stated lesson, that
        # a graph missing the link will miss those tickets, was false in the data.
        linked = rng.random() >= unlinked_rate
        if linked:
            subject = ci.name
        else:
            # What somebody writes when they did not fill the field in: the service in
            # words, or the kind of box, never the record's own name.
            subject = rng.choice([
                f"the {ci.domain} service",
                f"one of the {ci.domain} boxes",
                f"the {ci.environment} {ci.domain} stack",
                f"a host behind {ci.domain}",
            ])
        head = f"{subject}: {symptom}"
        # Who noticed depends on where it broke. Nobody outside the company reports a
        # fault in a development environment, and a first version had customers doing
        # exactly that, which is the same kind of tell as a mismatched stack trace.
        if ci.environment in ("prd", "dr"):
            reporter = rng.choice([
                "Reported by the service desk", "Raised by monitoring",
                "Noticed by the on call engineer", "Reported by a customer",
                "Escalated from the contact centre",
            ])
        else:
            reporter = rng.choice([
                "Raised by monitoring", "Noticed by the team during testing",
                "Picked up during the morning check", "Found by the developer on duty",
            ])
        scope = "Several users affected" if priority <= 2 else "Impact looks limited so far"
        body = [
            f"{symptom.capitalize()}. First noticed on {subject} in {ci.environment}.",
            f"{reporter} at {opened:%H:%M}. {scope}.",
        ]
        if rng.random() < paste_rate:
            body.append(rng.choice(PASTED[family]))
        if is_dup:
            body.append(f"Looks like the same thing as {last_on_ci[(ci.key, family)]}.")
        elif rng.random() < cross_reference_rate and out:
            # ⛔ A REFERENCE TO A RANDOM TICKET CARRIES NO INFORMATION. The docstring
            # calls this the strongest lexical signal in real ITSM text and it was
            # pointing at any earlier ticket in the estate: different item, different
            # fault, different symptom. A reader measuring the lift from following
            # cross references would have got exactly nothing, and a graph built on them
            # gets eleven thousand noise edges. An analyst references a ticket because
            # it looked like this one.
            pool = related.get(family) or []
            other = rng.choice(pool) if pool else None
            if other and other != number:
                body.append(f"Possibly related to {other}, similar symptoms.")

        resolved = _clamp(opened + _mttr(rng, priority)) if rng.random() < 0.93 else None

        notes: list[tuple[datetime, str, str]] = []
        when = opened
        for _ in range(rng.choices([1, 2, 3, 4, 5], weights=[.24, .3, .24, .14, .08], k=1)[0]):
            when = _clamp(when + timedelta(minutes=rng.randint(8, 900)))
            if resolved and when > resolved:
                break
            author = rng.choice(people)
            near = neighbours.get(ci.key) or []
            if near and rng.random() < neighbour_mention_rate:
                # An analyst checking the thing next door, which is the only textual
                # trace a one hop question ever has.
                other = names.get(rng.choice(near), "")
                text = rng.choice([
                    f"Checked {other} as well, same window.",
                    f"{other} looks clean, ruling it out.",
                    f"Asked the team who own {other} to look.",
                    f"Nothing on {other}, so it is not upstream.",
                ])
            elif rng.random() < 0.45:
                text = rng.choice(CLIPPED)
            else:
                text = f"Checked {subject}. {cause.capitalize()}."
            notes.append((when, author, text))

        out.append(Incident(
            number=number,
            short_description=head,
            description="\n\n".join(body),
            ci_key=ci.key if linked else None,
            category=family,
            priority=priority,
            opened_at=opened,
            resolved_at=resolved,
            # ⛔ ROUTING IS NOT NOISE. Picking any queue at random meant 6.3% of
            # tickets reached the right team against a 6.7% chance baseline, so "which
            # team owns this" was unanswerable by any method. Real desks route mostly
            # correctly and misroute a known minority.
            assignment_group=(f"{ci.domain}-support" if rng.random() < 0.82
                              else rng.choice(groups)),
            caller=rng.choice(people),
            close_notes=rng.choice(CLOSE_NOTES) if resolved else None,
            work_notes=notes,
            duplicate_of=last_on_ci.get((ci.key, family)) if is_dup else None,
        ))
        last_on_ci[(ci.key, family)] = number
        pool = related.setdefault(family, [])
        pool.append(number)
        if len(pool) > 400:
            pool.pop(0)

    # Already in time order from the draft sort. Kept as a guard, not a fix.
    assert all(a.opened_at <= b.opened_at for a, b in zip(out, out[1:]))
    return out


def report(incidents: list[Incident]) -> str:
    """The properties the article has to declare, measured rather than claimed."""
    from collections import Counter

    n = len(incidents)
    heads = Counter(i.short_description.split(": ", 1)[-1] for i in incidents)
    bodies = [i.description for i in incidents]
    uniq_bodies = len({hashlib.md5(b.encode()).hexdigest() for b in bodies})
    dup = sum(1 for i in incidents if i.duplicate_of)
    xref = sum(1 for i in incidents if "related to INC" in i.description)
    pasted = sum(1 for i in incidents if "```" in i.description)
    nolink = sum(1 for i in incidents if not i.ci_key)
    notes = sum(len(i.work_notes) for i in incidents)
    words = sum(len(i.description.split()) for i in incidents) / max(1, n)

    return "\n".join([
        f"incidents            : {n:,}",
        f"distinct symptoms    : {len(heads)}",
        f"distinct description bodies : {uniq_bodies:,}  ({uniq_bodies / n:.0%} unique)",
        f"average body length  : {words:.0f} words",
        f"work notes           : {notes:,}  ({notes / n:.1f} per incident)",
        "",
        "the properties similarity search needs, and their rates:",
        f"  repeat of an earlier ticket : {dup:,} ({dup / n:.0%})",
        f"  names another ticket        : {xref:,} ({xref / n:.0%})",
        f"  carries pasted output       : {pasted:,} ({pasted / n:.0%})",
        f"  no configuration item set   : {nolink:,} ({nolink / n:.0%})",
        "",
        "priority spread:",
        *[f"  P{k} {v:>7,} ({v / n:.0%})"
          for k, v in sorted(Counter(i.priority for i in incidents).items())],
    ])


if __name__ == "__main__":
    from estate import build_estate

    est = build_estate(services=2200)
    inc = generate(est, count=60_000)
    print(report(inc))
    print("\n" + "─" * 70 + "\n  ONE SAMPLE\n" + "─" * 70)
    s = next(i for i in inc if "```" in i.description and i.duplicate_of)
    print(f"{s.number}  P{s.priority}  {s.short_description}\n")
    print(s.description)
    print("\nwork notes:")
    for when, who, text in s.work_notes:
        print(f"  {when:%Y-%m-%d %H:%M}  {who}  {text}")
