"""Tests for the dataset generator.

Every property tested here is one the article states as fact. If a test fails, an
article claim is wrong, not merely a script. That is the reason these exist: three real
bugs were found by reading sample output by eye, and reading by eye does not scale to
sixty thousand records.

The bugs that got through before these tests existed, all now covered below:
  · a pasted stack trace that did not match the symptom it sat under
  · a customer reporting a fault in a development environment
  · a reactive change landing on a different item from the incident it responded to,
    which silently disabled the whole reverse causation trap

Author: Roni Das
Created: 2026-09-08
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import timedelta

import pytest

GEN = pathlib.Path(__file__).resolve().parent.parent / "generator"
sys.path.insert(0, str(GEN))

from estate import CLASSES, build_estate  # noqa: E402
from incidents import PASTED, generate as gen_incidents  # noqa: E402


def _load_rows(name: str) -> list[dict]:
    """Read a built dataset file. These tests check the artefact that ships, not a
    freshly generated one, because the file is what gets loaded and measured."""
    path = DATASET / f"{name}.jsonl"
    if not path.exists():
        pytest.skip(f"{name}.jsonl not built yet")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

from records import (  # noqa: E402
    generate_changes,
    generate_knowledge,
    generate_problems,
)

FENCE = "```"


# ── fixtures ───────────────────────────────────────────────────────────────────────
# Module scope: these are expensive to build and no test mutates them.

@pytest.fixture(scope="module")
def estate():
    return build_estate(services=500, seed=1234)


@pytest.fixture(scope="module")
def incidents(estate):
    return gen_incidents(estate, count=6000, seed=1234)


@pytest.fixture(scope="module")
def changes(estate, incidents):
    return generate_changes(estate, incidents, count=1500, seed=1234)


@pytest.fixture(scope="module")
def problems(incidents):
    return generate_problems(incidents, count=200, seed=1234)


# ── the estate ─────────────────────────────────────────────────────────────────────

def test_estate_is_reproducible_from_its_seed():
    # Arrange
    a = build_estate(services=60, seed=99)
    b = build_estate(services=60, seed=99)

    # Act
    keys_a = [c.key for c in a.cis]
    keys_b = [c.key for c in b.cis]

    # Assert
    assert keys_a == keys_b
    assert len(a.rels) == len(b.rels)


def test_a_different_seed_gives_a_different_estate():
    a = build_estate(services=60, seed=1)
    b = build_estate(services=60, seed=2)
    assert [c.key for c in a.cis] != [c.key for c in b.cis]


def test_every_relationship_points_at_items_that_exist(estate):
    keys = {c.key for c in estate.cis}
    dangling = [r for r in estate.rels
                if r.parent_key not in keys or r.child_key not in keys]
    assert dangling == [], f"{len(dangling)} edges point at an item that is not in the estate"


def test_no_item_depends_on_itself(estate):
    assert [r for r in estate.rels if r.parent_key == r.child_key] == []


def test_the_estate_contains_a_dependency_loop(estate):
    """A real estate has loops. Without one the article cannot show what they do to a
    traversal, and a reader would meet their first loop in production."""
    adjacency: dict[str, set[str]] = {}
    for r in estate.rels:
        adjacency.setdefault(r.parent_key, set()).add(r.child_key)

    # Any back edge in a depth first walk is a loop.
    colour: dict[str, int] = {}
    found = False

    def walk(node: str) -> bool:
        colour[node] = 1
        for nxt in adjacency.get(node, ()):
            if colour.get(nxt) == 1:
                return True
            if colour.get(nxt) is None and walk(nxt):
                return True
        colour[node] = 2
        return False

    sys.setrecursionlimit(20000)
    for start in list(adjacency):
        if colour.get(start) is None and walk(start):
            found = True
            break
    assert found, "no dependency loop in the estate"


def test_some_items_carry_far_more_edges_than_the_rest(estate):
    """The supernode. An uncapped traversal through one of these returns the estate,
    and the article teaches a hop cap because of it."""
    degree: dict[str, int] = {}
    for r in estate.rels:
        degree[r.parent_key] = degree.get(r.parent_key, 0) + 1
        degree[r.child_key] = degree.get(r.child_key, 0) + 1

    counts = sorted(degree.values(), reverse=True)
    median = counts[len(counts) // 2]
    assert counts[0] > median * 20, (
        f"busiest item has {counts[0]} edges against a median of {median}; "
        "nothing here behaves like a shared cluster"
    )


def test_a_meaningful_share_of_edges_are_stale(estate):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    stale = [r for r in estate.rels
             if r.last_discovered and (now - r.last_discovered).days > 365]
    share = len(stale) / len(estate.rels)
    assert 0.10 < share < 0.30, f"stale share is {share:.0%}, which is not lifelike"


def test_every_class_used_is_a_real_servicenow_table(estate):
    """These were read off the live instance. An invented table name would fail at load
    time, after an hour of ingestion."""
    allowed = set(CLASSES.values())
    assert {c.sys_class_name for c in estate.cis} <= allowed


def test_every_configuration_item_has_a_unique_name(estate):
    """The identification engine identifies on name, not on our key. A first version
    reused 120 names, one of them 61 times, and every batch containing two of them was
    abandoned an hour into a load. Unique keys did not help, because nothing identifies
    on the key."""
    from collections import Counter
    names = Counter(c.name for c in estate.cis)
    reused = {n: k for n, k in names.items() if k > 1}
    assert reused == {}, f"{len(reused)} names are used more than once, worst {max(reused.values())}x"


def test_every_class_is_independently_identifiable(estate):
    """ServiceNow refuses a dependent class unless the payload also carries its
    container and the containment relationship. Measured on the instance: cmdb_ci_appl,
    cmdb_ci_db_instance and cmdb_ci_storage_device are all dependent, and using them
    made every load batch fail."""
    DEPENDENT = {
        "cmdb_ci_appl", "cmdb_ci_db_instance", "cmdb_ci_database",
        "cmdb_ci_app_server", "cmdb_ci_web_server", "cmdb_ci_storage_device",
    }
    used = {c.sys_class_name for c in estate.cis}
    assert not (used & DEPENDENT), f"dependent classes in the estate: {used & DEPENDENT}"


def test_discovery_source_uses_real_choice_values(estate):
    """These are choice values on the instance, not free text. An invented one makes the
    identification engine reject the whole payload with INVALID_INPUT_DATA."""
    VALID = {"ServiceNow", "ServiceWatch", "Manual Entry", "ImportSet",
             "Other Automated", "PatternDesigner", "AgentClientCollector"}
    used = {c.discovery_source for c in estate.cis}
    assert used <= VALID, f"invented discovery sources: {used - VALID}"


# ── incidents ──────────────────────────────────────────────────────────────────────

def test_pasted_output_always_matches_the_symptom(incidents):
    """The bug that started this file. A memory fault carrying a replication lag
    warning tells any engineer the data is invented."""
    wrong = []
    for inc in incidents:
        if FENCE not in inc.description:
            continue
        if not any(block in inc.description for block in PASTED[inc.category]):
            wrong.append(inc.number)
    assert wrong == [], f"{len(wrong)} incidents carry output from the wrong failure kind"


def test_nobody_outside_the_company_reports_a_development_fault(estate, incidents):
    env = {c.key: c.environment for c in estate.cis}
    external = ("Reported by a customer", "Escalated from the contact centre")
    wrong = [
        i.number for i in incidents
        if any(p in i.description for p in external)
        and i.ci_key and env.get(i.ci_key) in ("dev", "stg")
    ]
    assert wrong == [], f"{len(wrong)} development faults reported by an outsider"


def test_descriptions_are_overwhelmingly_unique(incidents):
    """If the text repeats, similarity search has nothing to work with and loses the
    comparison for a reason that is the generator's fault."""
    bodies = {i.description for i in incidents}
    share = len(bodies) / len(incidents)
    assert share > 0.95, f"only {share:.0%} of descriptions are unique"


def test_descriptions_are_long_enough_to_embed(incidents):
    lengths = [len(i.description.split()) for i in incidents]
    average = sum(lengths) / len(lengths)
    assert average > 25, f"average description is {average:.0f} words, too thin to retrieve on"


def test_priorities_look_like_a_real_queue(incidents):
    from collections import Counter
    share = {k: v / len(incidents) for k, v in Counter(i.priority for i in incidents).items()}
    assert share[1] < 0.06, "too many P1s; everything cannot be an emergency"
    assert share[3] + share[4] > 0.75, "a real queue is mostly routine work"


def test_the_lexical_signals_similarity_search_needs_are_present(incidents):
    """Each of these is a real property of analyst writing and each one is a signal a
    keyword or vector search legitimately uses. Leaving them out handicaps one side of
    the comparison before it starts."""
    n = len(incidents)
    repeats = sum(1 for i in incidents if i.duplicate_of) / n
    named = sum(1 for i in incidents if "related to INC" in i.description) / n
    pasted = sum(1 for i in incidents if FENCE in i.description) / n

    assert 0.05 < repeats < 0.25, f"repeat rate {repeats:.0%}"
    assert 0.10 < named < 0.35, f"cross reference rate {named:.0%}"
    assert 0.20 < pasted < 0.50, f"pasted output rate {pasted:.0%}"


def test_a_real_share_of_incidents_have_no_configuration_item(incidents):
    """On a real instance this field is often empty. A graph that assumes it is always
    set will silently miss those tickets."""
    share = sum(1 for i in incidents if not i.ci_key) / len(incidents)
    assert 0.08 < share < 0.30, f"{share:.0%} unlinked, which is not lifelike"


def test_every_linked_incident_points_at_an_item_that_exists(estate, incidents):
    keys = {c.key for c in estate.cis}
    assert [i.number for i in incidents if i.ci_key and i.ci_key not in keys] == []


def test_no_incident_is_resolved_before_it_was_opened(incidents):
    assert [i.number for i in incidents if i.resolved_at and i.resolved_at < i.opened_at] == []


def test_work_notes_are_written_after_the_incident_opened(incidents):
    early = [
        i.number for i in incidents
        for when, _, _ in i.work_notes
        if when < i.opened_at
    ]
    assert early == []


def test_a_duplicate_points_at_an_earlier_ticket_on_the_same_item(incidents):
    """A repeat must reference a ticket that actually came before it, or the article's
    "has this happened before" question has nothing true underneath it."""
    when = {i.number: i.opened_at for i in incidents}
    wrong = [
        i.number for i in incidents
        if i.duplicate_of and i.duplicate_of in when and when[i.duplicate_of] > i.opened_at
    ]
    assert wrong == [], f"{len(wrong)} tickets are a repeat of something that came later"


def test_most_incidents_are_raised_during_working_hours(incidents):
    day = sum(1 for i in incidents if 7 <= i.opened_at.hour < 20)
    share = day / len(incidents)
    assert 0.6 < share < 0.9, f"{share:.0%} raised in the day, which is not a real pattern"


# ── changes, problems, knowledge ───────────────────────────────────────────────────

def test_a_reactive_change_sits_on_the_incident_it_responded_to(changes, incidents):
    """The bug that disabled the reverse causation trap. If the change is on a different
    item, a query filtered by configuration item never sees it and the trap never fires."""
    ci_of = {i.number: i.ci_key for i in incidents}
    wrong = [
        c.number for c in changes
        if c.raised_after_incident and ci_of.get(c.raised_after_incident) != c.ci_key
    ]
    assert wrong == [], f"{len(wrong)} reactive changes are on the wrong item"


def test_a_reactive_change_opens_after_its_incident(changes, incidents):
    opened = {i.number: i.opened_at for i in incidents}
    wrong = [
        c.number for c in changes
        if c.raised_after_incident and c.opened_at <= opened[c.raised_after_incident]
    ]
    assert wrong == [], "a reactive change cannot predate the incident it reacted to"


def test_emergency_changes_are_rare(changes):
    share = sum(1 for c in changes if c.change_type == "emergency") / len(changes)
    assert share < 0.15, f"{share:.0%} emergency changes; no change board would recognise that"


def test_some_changes_ran_outside_their_planned_window(changes):
    """If planned and actual always agree, the article cannot teach why the difference
    matters, and a reader will match on the wrong field in production."""
    drifted = [
        c for c in changes
        if c.actual_end and abs((c.actual_end - c.planned_end).total_seconds()) > 1800
    ]
    assert len(drifted) / len(changes) > 0.10


def test_no_change_ends_before_it_starts(changes):
    assert [c.number for c in changes
            if c.actual_start and c.actual_end and c.actual_end < c.actual_start] == []
    assert [c.number for c in changes if c.planned_end < c.planned_start] == []


def test_a_problem_groups_one_fault_on_one_item(problems, incidents):
    """A problem record is a recurring fault, not a category. Sampling a whole category
    produced problems spanning a median of five unrelated items, titled after only the
    first, so a traversal from an incident to its problem and back returned tickets that
    had nothing to do with it. That starves the graph side of the comparison."""
    by_number = {i.number: i for i in incidents}
    for prb in problems:
        members = [by_number[n] for n in prb.incident_numbers if n in by_number]
        assert members, f"{prb.number} groups nothing that exists"
        items = {m.ci_key for m in members}
        assert len(items) == 1, f"{prb.number} spans {len(items)} items"
        kinds = {m.category for m in members}
        assert len(kinds) == 1, f"{prb.number} mixes {kinds}"
        assert prb.ci_key in items, f"{prb.number} is titled after an item it does not group"


def test_a_problem_groups_incidents_that_actually_exist(problems, incidents):
    numbers = {i.number for i in incidents}
    for prb in problems:
        assert set(prb.incident_numbers) <= numbers
        assert len(prb.incident_numbers) >= 2, "a problem grouping one incident is just an incident"


def test_a_problem_opens_after_the_first_incident_it_groups(problems, incidents):
    opened = {i.number: i.opened_at for i in incidents}
    for prb in problems:
        first = min(opened[n] for n in prb.incident_numbers)
        assert prb.opened_at >= first


def test_knowledge_articles_only_come_from_solved_problems(problems):
    """Every article documents a problem that was actually solved, and it pairs by the
    link rather than by position. A first version zipped two lists and broke the moment
    the eligibility rule changed, which is a test measuring the wrong thing."""
    kb = generate_knowledge(problems, seed=1234)
    solved = {p.number for p in problems if p.resolved_at}
    assert kb, "no articles were produced at all"
    assert len(kb) <= len(solved)
    for a in kb:
        assert a.problem_number in solved, f"{a.number} documents an unsolved problem"
        assert a.text.strip()
        assert "## What to do now" in a.text


def test_knowledge_is_published_after_its_problem_was_solved(problems):
    """An article that predates its own resolution is nonsense. Paired by problem
    number, because the two lists are no longer the same length: a problem resolved on
    the snapshot edge has no room for an article to be published after it."""
    kb = generate_knowledge(problems, seed=1234)
    by_number = {p.number: p for p in problems}
    assert kb
    for article in kb:
        prb = by_number[article.problem_number]
        assert article.published_at > prb.resolved_at, (
            f"{article.number} is published before {prb.number} was resolved")


# ── the built dataset on disk ──────────────────────────────────────────────────────

DATASET = pathlib.Path(__file__).resolve().parent.parent / "dataset"


@pytest.mark.skipif(not (DATASET / "manifest.json").exists(),
                    reason="dataset not built yet; run generator/build.py")
def test_the_manifest_rates_match_the_files_they_describe():
    """The article quotes these rates. If the manifest and the file disagree, the
    article is quoting a number that is not in the data readers downloaded."""
    manifest = json.loads((DATASET / "manifest.json").read_text())
    rows = [json.loads(line) for line in (DATASET / "incidents.jsonl").read_text().splitlines()]

    measured = {
        "incidents_repeating_an_earlier_ticket":
            sum(1 for r in rows if r["duplicate_of"]) / len(rows),
        "incidents_with_pasted_output":
            sum(1 for r in rows if FENCE in r["description"]) / len(rows),
        "incidents_with_no_configuration_item":
            sum(1 for r in rows if not r["ci_key"]) / len(rows),
    }
    for key, actual in measured.items():
        claimed = manifest["measured_rates"][key]
        assert abs(claimed - actual) < 0.005, (
            f"{key}: manifest says {claimed:.1%}, the file says {actual:.1%}"
        )


@pytest.mark.skipif(not (DATASET / "manifest.json").exists(), reason="dataset not built")
def test_every_file_the_manifest_names_exists_with_the_row_count_it_claims():
    manifest = json.loads((DATASET / "manifest.json").read_text())
    for name, meta in manifest["files"].items():
        path = DATASET / f"{name}.jsonl"
        assert path.exists(), f"{name} is in the manifest and not on disk"
        assert sum(1 for _ in path.open()) == meta["rows"]


@pytest.mark.skipif(not (DATASET / "incidents.jsonl").exists(), reason="dataset not built")
def test_every_timestamp_written_carries_a_timezone():
    """A timestamp without a zone is the most expensive kind of missing information in
    this whole article, and it would be indefensible to ship one in the dataset."""
    with (DATASET / "incidents.jsonl").open() as f:
        for _ in range(500):
            row = json.loads(next(f))
            assert row["opened_at"].endswith("+00:00")
            if row["resolved_at"]:
                assert row["resolved_at"].endswith("+00:00")

# ── the three properties that keep the comparison honest ───────────────────────────
# An audit measured that the article's whole premise was decided by this generator
# rather than by retrieval. These three tests are the guard against that returning.

def test_a_service_stack_is_not_recoverable_by_matching_a_name(estate, incidents):
    """THE ONE THAT MATTERED MOST. Every item in a stack used to be named from the same
    "{domain}-{env}-{index}" token, so a plain substring search recovered the whole stack
    at 78% recall with no graph at all. The question the article says a graph is needed
    for was answerable by string matching.

    ⛔ The stack is taken from the KEYS, which encode real ownership, not from a graph
    traversal. A first version of this test walked two hops and passed even with the leak
    put back, because two hops through a shared cluster reaches a third of the estate and
    buries the eight items that actually belong to the service.
    """
    on_item: dict[str, list] = {}
    for i in incidents:
        if i.ci_key:
            on_item.setdefault(i.ci_key, []).append(i)

    services = [c for c in estate.cis if c.key.startswith("svc-") and c.key in on_item]
    assert services, "no service carries a ticket, so this measures nothing"

    tp = fn = 0
    checked = 0
    for svc in sorted(services, key=lambda c: -len(on_item[c.key]))[:15]:
        tag = svc.key[len("svc-"):]
        # Everything the service owns: its application, its hosts, its database, its
        # load balancer. The key carries the ownership, the name no longer does.
        owned = {c.key for c in estate.cis
                 if c.key.endswith(tag) or f"-{tag}-" in c.key} - {svc.key}
        truth = {i.number for i in incidents if i.ci_key in owned}
        if not truth:
            continue
        checked += 1
        needle = svc.name.lower()
        found = {i.number for i in incidents
                 if needle in (i.short_description + " " + i.description).lower()}
        tp += len(truth & found)
        fn += len(truth - found)

    assert checked >= 5 and tp + fn >= 20, (
        f"only {checked} services with {tp + fn} owned tickets; this test would pass "
        "on nothing, which is the failure mode it exists to prevent"
    )
    recall = tp / (tp + fn)
    assert recall < 0.20, (
        f"a substring search on the service name recovers {recall:.0%} of the tickets "
        "raised against the things it owns. The names are spelling out the graph."
    )


def test_some_tickets_name_a_neighbour_so_text_has_a_real_chance(estate, incidents):
    """The opposite rigging, and just as bad. If no ticket ever mentions anything but
    its own item, a multi hop question has zero textual evidence and the graph wins by
    construction rather than by being better. Real analysts write "checked lnx0341,
    same window" all the time."""
    names = {c.key: c.name for c in estate.cis}
    children: dict[str, set[str]] = {}
    for r in estate.rels:
        children.setdefault(r.parent_key, set()).add(r.child_key)
        children.setdefault(r.child_key, set()).add(r.parent_key)

    linked = [i for i in incidents if i.ci_key]
    crossing = 0
    for i in linked:
        text = " ".join(n[2] for n in i.work_notes)
        if any(names.get(nb, "\0") in text for nb in children.get(i.ci_key, ())):
            crossing += 1

    share = crossing / len(linked)
    assert 0.10 < share < 0.55, (
        f"{share:.0%} of tickets name a one hop neighbour. Below 10% the text baseline "
        "cannot answer a multi hop question at all; above 55% the graph is no longer "
        "buying anything the words do not already give away."
    )


def test_an_unlinked_ticket_never_names_its_own_item(estate, incidents):
    """The 17% with no configuration item are supposed to be the tickets a graph misses.
    A first version nulled the link AFTER printing the name into the title and the body,
    so the link was recoverable by exact match and the lesson was false."""
    allnames = {c.name for c in estate.cis}
    leaked = [
        i.number for i in incidents if not i.ci_key
        and any(n in (i.short_description + " " + i.description) for n in allnames)
    ]
    assert leaked == [], f"{len(leaked)} unlinked tickets still name a real item"


@pytest.mark.skipif(not (DATASET / "manifest.json").exists(), reason="dataset not built")
def test_nothing_in_the_dataset_is_dated_in_the_future():
    """A ticket system does not record things that have not happened. 154 incidents used
    to resolve in the future and 408 changes to start in one, because a resolution time
    was added to an opening time that could already be recent. Fixed once for incidents
    and not for changes, which is the kind of half fix that survives because nobody
    measures the other file. Every file is measured here."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    late = []
    for name, fields in [
        ("incidents", ["opened_at", "resolved_at"]),
        ("changes", ["opened_at", "planned_start", "planned_end",
                     "actual_start", "actual_end"]),
        ("problems", ["opened_at", "resolved_at"]),
        ("knowledge", ["published_at"]),
    ]:
        for line in (DATASET / f"{name}.jsonl").read_text().splitlines():
            row = json.loads(line)
            for f in fields:
                if row.get(f) and datetime.fromisoformat(row[f]) > now:
                    late.append(f"{name}.{f}")
    assert late == [], f"{len(late)} timestamps are in the future, e.g. {late[:4]}"


@pytest.mark.skipif(not (DATASET / "manifest.json").exists(), reason="dataset not built")
def test_work_note_timestamps_carry_a_timezone_too():
    """The only timezone guard read incidents' own two fields and never the work notes,
    which are the text the article embeds."""
    with (DATASET / "incidents.jsonl").open() as f:
        for _ in range(400):
            row = json.loads(next(f))
            for when, _who, _text in row["work_notes"]:
                assert when.endswith("+00:00"), f"{row['number']} has a naive note time"


def test_two_builds_from_one_seed_are_byte_identical():
    """The manifest publishes a checksum per file and the article calls the dataset
    reproducible. It was not: timestamps came from datetime.now(), so every rebuild
    produced different bytes and a reader checking the checksum would have found it
    wrong. Everything is now anchored to a stated as-of date, which is also what a real
    export is."""
    import dataclasses
    import hashlib

    def digest(rows):
        h = hashlib.sha256()
        for row in rows:
            h.update(json.dumps(dataclasses.asdict(row), default=str,
                                sort_keys=True).encode())
        return h.hexdigest()

    a = build_estate(services=120, seed=777)
    b = build_estate(services=120, seed=777)
    assert digest(a.cis) == digest(b.cis), "two estate builds differ"
    assert digest(a.rels) == digest(b.rels), "two edge builds differ"

    ia = gen_incidents(a, count=800, seed=777)
    ib = gen_incidents(b, count=800, seed=777)
    assert digest(ia) == digest(ib), "two incident builds differ"


def test_the_as_of_date_is_in_the_past_and_stated():
    """Anchoring is only honest if the anchor is visible. Everything in the dataset is
    dated relative to it, so a reader has to be able to see what 'now' means here."""
    from estate import AS_OF
    from datetime import datetime, timezone
    assert AS_OF.tzinfo is not None, "the anchor must carry a timezone"
    assert AS_OF < datetime.now(timezone.utc), "the anchor cannot be in the future"


def test_every_relationship_type_is_one_the_instance_actually_has(estate):
    """⛔ I WROTE ONE BACKWARDS, IN AN ARTICLE ABOUT RELATIONSHIP DIRECTION.
    "Owned by::Owns" does not exist; the instance has "Owns::Owned by", because the
    parent owns and the child is owned. The loader refused the whole phase, which is why
    it resolves every type by name before writing a row, but the estate should not have
    produced it in the first place.

    This list was read off the instance with a query, not recalled. Anything not on it
    is either invented or inverted, and both fail an hour into a load.
    """
    ON_THE_INSTANCE = {
        "Depends on::Used by", "Runs on::Runs", "Hosted on::Hosts",
        "IP Connection::IP Connection", "In Rack::Rack contains",
        "Managed by::Manages", "Owns::Owned by", "Located in Zone::Zone contains",
        "Contains::Contained by", "Registered on::Has registered",
        "Located In::Contains Room", "Cools::Cooled By", "Feeds::Fed By",
        "Distributed by::Distributes", "Instantiates::Instantiated by",
        "Connected by::Connects", "Consumes::Consumed by",
    }
    used = {r.type_name for r in estate.rels}
    invented = used - ON_THE_INSTANCE
    assert invented == set(), (
        f"these relationship types are not on the instance: {sorted(invented)}. "
        "Check for an inverted name before assuming it is missing."
    )


# ── relationship direction, in an article about relationship direction ─────────────

# What the parent must be for each type name. A ServiceNow relationship type is written
# "what the parent is to the child::what the child is to the parent", so the parent is
# always the subject of the FIRST phrase.
PARENT_IS = {
    "Depends on::Used by":            ("svc", "app", "lb"),
    "Runs on::Runs":                  ("app",),
    "Hosted on::Hosts":               ("host", "db"),
    "In Rack::Rack contains":         ("host",),
    "Located in Zone::Zone contains": ("host",),
    "Managed by::Manages":            ("app", "svc"),
    "Owns::Owned by":                 ("rack",),
}


def _layer(key: str) -> str:
    for prefix in ("svc-", "app-", "host-", "db-", "cluster-", "rack-", "san-", "lb-"):
        if key.startswith(prefix):
            return prefix.rstrip("-")
    return key.split("-")[0]


def test_every_relationship_points_the_way_its_name_says():
    """⛔ 24 PERCENT OF THE GRAPH POINTED BACKWARDS, in the article whose whole subject is
    which way an edge goes. The generator wrote the container as parent for every
    containment type, so "Hosted on::Hosts" said the cluster was hosted on the server.

    The cost was not cosmetic. It removed the graph's upward direction entirely: a
    cluster had 950 dependencies and nothing depending on it, so "what breaks if this
    cluster fails" correctly returned nothing, and the multi hop questions the article
    exists to answer had no answer to find.

    ⛔ Note "Owns::Owned by" is NOT inverted and must stay as it is. The owner is the
    subject of "Owns", so the owner is already the parent. The fix is per type name; a
    blanket swap would have broken the one edge type that was right.
    """
    from collections import Counter
    rels = _load_rows("relationships")
    wrong = []
    for kind, allowed in PARENT_IS.items():
        seen = Counter(_layer(r["parent_key"]) for r in rels if r["type_name"] == kind)
        for layer, n in seen.items():
            if layer not in allowed:
                wrong.append(f"{kind}: {n:,} edges with a {layer!r} parent, "
                             f"expected one of {allowed}")
    assert not wrong, "relationships point the wrong way:\n  " + "\n  ".join(wrong)


def test_a_shared_cluster_has_things_depending_on_it():
    """The symptom the direction bug produced, asserted directly so the cause cannot come
    back wearing a different shape."""
    rels = _load_rows("relationships")
    supports = {}
    impact = {"Runs on::Runs", "Depends on::Used by", "Hosted on::Hosts"}
    for r in rels:
        if r["type_name"] in impact:
            supports.setdefault(r["child_key"], set()).add(r["parent_key"])
    clusters = [k for k in {r["child_key"] for r in rels} if k.startswith("cluster-")]
    assert clusters, "no clusters in the estate"
    for c in clusters:
        assert len(supports.get(c, ())) > 100, (
            f"{c} has {len(supports.get(c, ()))} items depending on it. A shared cluster "
            f"with nothing above it means the edges point away from the estate.")
