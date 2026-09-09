"""Ask the graph the questions the article promises it can answer.

This exists to prove the model works before a word of the article is written. If the
opening question cannot be answered here, the whole premise is wrong and it is better to
find that out now.

Each query is written the way the article will teach it: filtered to the relationship
types that carry impact, capped in depth so a shared cluster cannot drag in the estate,
and reporting how old its own evidence is.

Author: Roni Das
Created: 2026-09-08
"""

from __future__ import annotations

import pathlib
import re
import time

from neo4j import GraphDatabase

IMPACT = ["Depends on::Used by", "Runs on::Runs", "Hosted on::Hosts"]


def read_env() -> tuple[str, str, str]:
    env: dict[str, str] = {}
    from env import env_path
    for line in env_path().read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return env["NEO4J_URI"], env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"]


def show(title: str, rows: list, took: float, note: str = "") -> None:
    print(f"\n{'─' * 74}\n  {title}\n{'─' * 74}")
    if note:
        print(f"  {note}")
    if not rows:
        print("  no rows")
        return
    for r in rows[:8]:
        print("   ", {k: (str(v)[:44] if v is not None else None) for k, v in r.items()})
    if len(rows) > 8:
        print(f"    ... and {len(rows) - 8} more")
    print(f"  {len(rows)} rows in {took * 1000:.0f}ms")


def main() -> None:
    uri, user, pw = read_env()
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    driver.verify_connectivity()

    with driver.session() as s:
        # ⛔ ASK ABOUT SOMETHING THINGS DEPEND ON. Edges run from the thing depended
        # upon toward the thing that depends on it, so a service at the top of its own
        # stack has NO outgoing impact edges and its blast radius is correctly empty.
        # A first version seeded on the busiest service and reported zero rows, which
        # read like a broken query and was in fact a well chosen question about a leaf.
        seed = s.run("""
            MATCH (c:ConfigurationItem)-[r:SUPPORTS]->(:ConfigurationItem)
            WHERE r.carries_impact AND c.environment = 'prd'
            WITH c, count(r) AS dependents
            MATCH (c)<-[:AFFECTS]-(i:Incident)
            RETURN c.key AS key, c.name AS name, dependents, count(i) AS incidents
            ORDER BY incidents DESC, dependents DESC LIMIT 1
        """).single()
        print(f"  asking about: {seed['name']}  ({seed['incidents']} incidents, "
      f"{seed['dependents']} things depend on it)")

        # 1. BLAST RADIUS. What breaks if this breaks.
        t0 = time.time()
        rows = s.run("""
            MATCH (start:ConfigurationItem {key:$key})
            MATCH path = (start)-[rels:SUPPORTS*1..4]->(affected:ConfigurationItem)
            WHERE all(r IN rels WHERE r.carries_impact)
            WITH DISTINCT affected, min(length(path)) AS hops,
                 reduce(oldest = datetime(), r IN rels |
                        CASE WHEN r.last_discovered < oldest
                             THEN r.last_discovered ELSE oldest END) AS oldest_evidence
            RETURN affected.name AS name, affected.sys_class_name AS class, hops,
                   duration.inDays(oldest_evidence, datetime()).days AS evidence_age_days
            ORDER BY hops, name LIMIT 40
        """, key=seed["key"]).data()
        stale = [r for r in rows if r["evidence_age_days"] and r["evidence_age_days"] > 365]
        show("1. BLAST RADIUS: what stops working if this does", rows, time.time() - t0,
             note=f"{len(stale)} of {len(rows)} rest on evidence over a year old")

        # 2. The same query WITHOUT the impact filter, to show what the filter buys.
        t0 = time.time()
        unfiltered = s.run("""
            MATCH (start:ConfigurationItem {key:$key})
            MATCH (start)-[:SUPPORTS*1..4]->(affected:ConfigurationItem)
            RETURN count(DISTINCT affected) AS n
        """, key=seed["key"]).single()["n"]
        filtered = s.run("""
            MATCH (start:ConfigurationItem {key:$key})
            MATCH (start)-[rels:SUPPORTS*1..4]->(affected:ConfigurationItem)
            WHERE all(r IN rels WHERE r.carries_impact)
            RETURN count(DISTINCT affected) AS n
        """, key=seed["key"]).single()["n"]
        print(f"\n{'─' * 74}\n  2. WHAT THE IMPACT FILTER BUYS\n{'─' * 74}")
        print(f"  without filtering by relationship type : {unfiltered:,} items")
        print(f"  filtered to types that carry impact    : {filtered:,} items")
        print(f"  the filter removes {unfiltered - filtered:,} items that cannot be affected")

        # 3. CHANGE CORRELATION, done honestly. Actual dates, and changes raised after
        #    the incident are excluded because they are the effect, not the cause.
        t0 = time.time()
        rows = s.run("""
            MATCH (i:Incident)-[:AFFECTS]->(ci:ConfigurationItem)
            WHERE i.opened_at > datetime() - duration({days: 400})
            MATCH (ci)<-[:SUPPORTS*0..2]-(near:ConfigurationItem)
            MATCH (ch:Change)-[:CHANGES]->(near)
            WHERE ch.actual_end IS NOT NULL
              AND ch.actual_end < i.opened_at
              AND ch.actual_end > i.opened_at - duration({hours: 24})
              AND (ch.raised_after_incident IS NULL)
            RETURN i.number AS incident, i.short_description AS symptom,
                   ch.number AS change, ch.change_type AS type,
                   duration.inSeconds(ch.actual_end, i.opened_at).minutes AS minutes_before
            ORDER BY minutes_before LIMIT 10
        """).data()
        show("3. CHANGE CORRELATION: what changed just before the fault", rows,
             time.time() - t0,
             note="excludes changes raised in response to the incident, which are the effect")

        # 4. The trap, shown. Include the reactive changes and the answer gets worse.
        t0 = time.time()
        trap = s.run("""
            MATCH (i:Incident)-[:AFFECTS]->(ci:ConfigurationItem)
            MATCH (ch:Change)-[:CHANGES]->(ci)
            WHERE ch.raised_after_incident = i.number
            RETURN count(*) AS n
        """).single()["n"]
        print(f"\n{'─' * 74}\n  4. THE TRAP\n{'─' * 74}")
        print(f"  changes raised AFTER the incident, on the same item: {trap:,}")
        print("  a time window that does not exclude these reports the fix as the cause")

        # 5. HAS THIS HAPPENED BEFORE. The question vector search should be good at.
        t0 = time.time()
        rows = s.run("""
            MATCH (i:Incident)-[:REPEATS]->(earlier:Incident)
            MATCH (p:Problem)-[:GROUPS]->(earlier)
            OPTIONAL MATCH (k:KnowledgeArticle)-[:DOCUMENTS]->(p)
            RETURN i.number AS incident, earlier.number AS seen_before,
                   p.number AS problem, p.workaround AS workaround,
                   k.number AS article
            LIMIT 6
        """).data()
        show("5. HAS THIS HAPPENED BEFORE, and what was done", rows, time.time() - t0)

        # 6. The supernode, to prove the cap is needed rather than assert it.
        busiest = s.run("""
            MATCH (c:ConfigurationItem)-[r:SUPPORTS]-()
            RETURN c.name AS name, count(r) AS edges ORDER BY edges DESC LIMIT 1
        """).single()
        t0 = time.time()
        capped = s.run("""
            MATCH (c:ConfigurationItem {name:$name})-[:SUPPORTS*1..3]-(o)
            RETURN count(DISTINCT o) AS n
        """, name=busiest["name"]).single()["n"]
        took = time.time() - t0
        print(f"\n{'─' * 74}\n  6. WHY THE HOP CAP EXISTS\n{'─' * 74}")
        print(f"  busiest item: {busiest['name']} with {busiest['edges']:,} edges")
        print(f"  three hops from it reaches {capped:,} items in {took * 1000:.0f}ms")
        print(f"  the whole estate is "
              f"{s.run('MATCH (c:ConfigurationItem) RETURN count(c) AS n').single()['n']:,}")

    driver.close()


if __name__ == "__main__":
    main()
