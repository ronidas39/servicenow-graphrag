"""Load the dataset into Neo4j, with the modelling decisions made explicit.

Every decision here is one the article argues for, so the code and the prose have to
agree. The four that matter most:

**Direction.** A ServiceNow relationship type is named "parent descriptor::child
descriptor", for example "Depends on::Used by". Read it backwards and every blast radius
answer inverts while still looking plausible, because the query returns rows either way.
One canonical direction is chosen here and written down: the arrow points from the thing
that is depended upon toward the thing that depends on it, so impact flows along the
arrow.

**Not every relationship carries impact.** A rack contains a server. The server does not
fail because the rack was moved. Measured on the instance, `cmdb_rel_type` holds only a
name, two descriptors and an endpoint, so there is NO column saying which types propagate.
The list is a decision, it is written down here, and every traversal filters on it.

**Classes are a hierarchy, not a label.** A configuration item is stored in its own class
table and belongs to every parent table above it. Flattening that to one `:CI` label
throws away the cheapest filter available at three in the morning.

**Freshness belongs on the edge.** Real dependency data rots. An answer that cannot say
how old its evidence is will be believed when it should not be.

Author: Roni Das
Created: 2026-09-08
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import time
from typing import Iterator

from neo4j import GraphDatabase

DATASET = pathlib.Path(__file__).resolve().parent.parent / "dataset"

# The class hierarchy, so one item can carry several labels. Read off the instance
# rather than invented: a linux server is a Server, and a Server is a ConfigurationItem.
CLASS_LABELS: dict[str, list[str]] = {
    "cmdb_ci_linux_server": ["LinuxServer", "Server", "ConfigurationItem"],
    "cmdb_ci_win_server": ["WindowsServer", "Server", "ConfigurationItem"],
    "cmdb_ci_server": ["Server", "ConfigurationItem"],
    "cmdb_ci_service": ["Service", "ConfigurationItem"],
    "cmdb_ci_cluster": ["Cluster", "ConfigurationItem"],
    "cmdb_ci_lb": ["LoadBalancer", "ConfigurationItem"],
    "cmdb_ci_storage_server": ["StorageServer", "Server", "ConfigurationItem"],
}

# ⛔ THE DECISION, NOT A LOOKUP. Nothing on the instance says which relationship types
# carry impact, so this list is ours and the article defends it. A traversal that does
# not filter on it returns the estate and calls it a blast radius.
IMPACT_TYPES = {"Depends on::Used by", "Runs on::Runs", "Hosted on::Hosts"}

CONSTRAINTS = [
    "CREATE CONSTRAINT ci_key IF NOT EXISTS "
    "FOR (c:ConfigurationItem) REQUIRE c.key IS UNIQUE",
    "CREATE CONSTRAINT incident_number IF NOT EXISTS "
    "FOR (i:Incident) REQUIRE i.number IS UNIQUE",
    "CREATE CONSTRAINT change_number IF NOT EXISTS "
    "FOR (c:Change) REQUIRE c.number IS UNIQUE",
    "CREATE CONSTRAINT problem_number IF NOT EXISTS "
    "FOR (p:Problem) REQUIRE p.number IS UNIQUE",
    "CREATE CONSTRAINT kb_number IF NOT EXISTS "
    "FOR (k:KnowledgeArticle) REQUIRE k.number IS UNIQUE",
    "CREATE CONSTRAINT person_id IF NOT EXISTS "
    "FOR (p:Person) REQUIRE p.user_id IS UNIQUE",
    "CREATE CONSTRAINT group_name IF NOT EXISTS "
    "FOR (g:Group) REQUIRE g.name IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX incident_opened IF NOT EXISTS FOR (i:Incident) ON (i.opened_at)",
    "CREATE INDEX incident_category IF NOT EXISTS FOR (i:Incident) ON (i.category)",
    "CREATE INDEX change_start IF NOT EXISTS FOR (c:Change) ON (c.actual_start)",
    "CREATE INDEX ci_environment IF NOT EXISTS FOR (c:ConfigurationItem) ON (c.environment)",
    "CREATE INDEX ci_name IF NOT EXISTS FOR (c:ConfigurationItem) ON (c.name)",
]


def read_env() -> tuple[str, str, str]:
    env: dict[str, str] = {}
    from env import env_path
    path = env_path()
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return env["NEO4J_URI"], env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"]


def rows(name: str) -> Iterator[dict]:
    with (DATASET / f"{name}.jsonl").open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def batched(it: Iterator[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for row in it:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_batches(session, cypher: str, source: Iterator[dict], size: int,
                label: str, **params) -> int:
    """Send rows in batches. UNWIND is the whole point: one round trip per batch
    instead of one per row, which is the difference between minutes and hours."""
    total = 0
    t0 = time.time()
    for batch in batched(source, size):
        session.run(cypher, rows=batch, **params)
        total += len(batch)
        if total % (size * 20) == 0:
            print(f"    {label:22s} {total:>8,}  {total / (time.time() - t0):>7.0f}/s",
                  flush=True)
    print(f"    {label:22s} {total:>8,}  done in {time.time() - t0:.0f}s", flush=True)
    return total


def load(uri: str, user: str, password: str, batch: int, wipe: bool) -> None:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()

    with driver.session() as s:
        if wipe:
            print("  clearing the database")
            while True:
                got = s.run(
                    "MATCH (n) WITH n LIMIT 20000 DETACH DELETE n RETURN count(n) AS n"
                ).single()["n"]
                if not got:
                    break

        # ⛔ CONSTRAINTS AND INDEXES BEFORE THE DATA, ALWAYS. A MERGE without a
        # supporting constraint scans every node of that label, so the load starts fast
        # and gets slower with every row. Creating them afterwards on a loaded database
        # also means the uniqueness violation is discovered at the end rather than at
        # the row that caused it.
        print("  constraints and indexes first")
        for stmt in CONSTRAINTS + INDEXES:
            s.run(stmt)
        s.run("CALL db.awaitIndexes(300)")

        print("  loading nodes")
        run_batches(s, """
            UNWIND $rows AS row
            MERGE (c:ConfigurationItem {key: row.key})
            SET c.name = row.name,
                c.sys_class_name = row.sys_class_name,
                c.environment = row.environment,
                c.region = row.region,
                c.domain = row.domain,
                c.operational_status = row.operational_status,
                c.install_status = row.install_status,
                c.discovery_source = row.discovery_source,
                c.last_discovered = datetime(row.last_discovered)
        """, rows("configuration_items"), batch, "configuration items")

        # The extra labels go on in a second pass, one statement per class, because a
        # label cannot be parameterised in Cypher.
        for cls, labels in CLASS_LABELS.items():
            extra = [x for x in labels if x != "ConfigurationItem"]
            if not extra:
                continue
            s.run(
                f"MATCH (c:ConfigurationItem {{sys_class_name: $cls}}) "
                f"SET c:{':'.join(extra)}", cls=cls,
            )
        print("    class labels applied")

        run_batches(s, """
            UNWIND $rows AS row
            MERGE (i:Incident {number: row.number})
            SET i.short_description = row.short_description,
                i.description       = row.description,
                i.category          = row.category,
                i.priority          = row.priority,
                i.opened_at         = datetime(row.opened_at),
                i.resolved_at       = CASE WHEN row.resolved_at IS NULL
                                      THEN NULL ELSE datetime(row.resolved_at) END,
                i.close_notes       = row.close_notes,
                i.duplicate_of      = row.duplicate_of,
                i.work_note_text    = reduce(t = '', n IN row.work_notes | t + ' ' + n[2])
        """, rows("incidents"), batch, "incidents")

        run_batches(s, """
            UNWIND $rows AS row
            MERGE (c:Change {number: row.number})
            SET c.short_description = row.short_description,
                c.description   = row.description,
                c.change_type   = row.change_type,
                c.risk          = row.risk,
                c.opened_at     = datetime(row.opened_at),
                c.planned_start = datetime(row.planned_start),
                c.planned_end   = datetime(row.planned_end),
                c.actual_start  = CASE WHEN row.actual_start IS NULL
                                  THEN NULL ELSE datetime(row.actual_start) END,
                c.actual_end    = CASE WHEN row.actual_end IS NULL
                                  THEN NULL ELSE datetime(row.actual_end) END,
                c.close_code    = row.close_code,
                c.raised_after_incident = row.raised_after_incident
        """, rows("changes"), batch, "changes")

        run_batches(s, """
            UNWIND $rows AS row
            MERGE (p:Problem {number: row.number})
            SET p.short_description = row.short_description,
                p.description = row.description,
                p.category    = row.category,
                p.workaround  = row.workaround,
                p.cause_notes = row.cause_notes,
                p.opened_at   = datetime(row.opened_at)
        """, rows("problems"), batch, "problems")

        run_batches(s, """
            UNWIND $rows AS row
            MERGE (k:KnowledgeArticle {number: row.number})
            SET k.short_description = row.short_description,
                k.text = row.text, k.category = row.category,
                k.published_at = datetime(row.published_at), k.views = row.views
        """, rows("knowledge"), batch, "knowledge")

        print("  loading relationships")
        # ⛔ ONE CANONICAL DIRECTION. ServiceNow stores parent and child, and the type
        # name says which is which: "Depends on::Used by" means the parent depends on
        # the child. So the child is what the parent needs, and impact travels from
        # child to parent. Every edge is written that way, so a traversal never has to
        # ask which end it is standing on.
        run_batches(s, """
            UNWIND $rows AS row
            MATCH (parent:ConfigurationItem {key: row.parent_key})
            MATCH (child:ConfigurationItem  {key: row.child_key})
            MERGE (child)-[r:SUPPORTS {type_name: row.type_name}]->(parent)
            SET r.last_discovered = CASE WHEN row.last_discovered IS NULL
                                    THEN NULL ELSE datetime(row.last_discovered) END,
                r.carries_impact  = row.type_name IN $impact
        """, rows("relationships"), batch, "dependency edges",
            impact=sorted(IMPACT_TYPES))

        run_batches(s, """
            UNWIND $rows AS row
            WITH row WHERE row.ci_key IS NOT NULL
            MATCH (i:Incident {number: row.number})
            MATCH (c:ConfigurationItem {key: row.ci_key})
            MERGE (i)-[:AFFECTS]->(c)
        """, rows("incidents"), batch, "incident to item")

        run_batches(s, """
            UNWIND $rows AS row
            MATCH (ch:Change {number: row.number})
            MATCH (c:ConfigurationItem {key: row.ci_key})
            MERGE (ch)-[:CHANGES]->(c)
        """, rows("changes"), batch, "change to item")

        run_batches(s, """
            UNWIND $rows AS row
            MATCH (p:Problem {number: row.number})
            UNWIND row.incident_numbers AS num
            MATCH (i:Incident {number: num})
            MERGE (p)-[:GROUPS]->(i)
        """, rows("problems"), batch, "problem to incidents")

        run_batches(s, """
            UNWIND $rows AS row
            MATCH (k:KnowledgeArticle {number: row.number})
            MATCH (p:Problem {number: row.problem_number})
            MERGE (k)-[:DOCUMENTS]->(p)
        """, rows("knowledge"), batch, "article to problem")

        run_batches(s, """
            UNWIND $rows AS row
            WITH row WHERE row.duplicate_of IS NOT NULL
            MATCH (a:Incident {number: row.number})
            MATCH (b:Incident {number: row.duplicate_of})
            MERGE (a)-[:REPEATS]->(b)
        """, rows("incidents"), batch, "repeat incidents")

        # ⛔ THE PEOPLE AND THE GROUPS WERE MISSING, AND THE ARTICLE DESCRIBED THEM.
        # Part 6 section 61 argues that an assignment group and a caller have to be nodes,
        # because "which teams need telling" gathers teams from many items at once and a
        # property cannot be gathered. The constraints for Person and Group were created
        # here and nothing ever wrote one, so Part 7's node table described a graph this
        # loader did not build, and the article's own claim checker validated the prose
        # against that imaginary graph by counting them from the dataset.
        #
        # They are derived rather than loaded, because the dataset has no people file:
        # every distinct caller and assignment_group on an incident.
        print("  people and groups, derived from the incidents")
        run_batches(s, """
            UNWIND $rows AS row
            WITH row WHERE row.assignment_group IS NOT NULL
            MERGE (g:Group {name: row.assignment_group})
            WITH g, row
            MATCH (i:Incident {number: row.number})
            MERGE (i)-[:ASSIGNED_TO]->(g)
        """, rows("incidents"), batch, "assignment groups")

        run_batches(s, """
            UNWIND $rows AS row
            WITH row WHERE row.caller IS NOT NULL
            MERGE (p:Person {user_id: row.caller})
            WITH p, row
            MATCH (i:Incident {number: row.number})
            MERGE (i)-[:RAISED_BY]->(p)
        """, rows("incidents"), batch, "callers")

        print("\n  what landed:")
        for label in ("ConfigurationItem", "Server", "Service", "Incident", "Change",
                      "Problem", "KnowledgeArticle", "Person", "Group"):
            n = s.run(f"MATCH (n:{label}) RETURN count(n) AS n").single()["n"]
            print(f"    {label:20s} {n:>8,}")
        for rel in ("SUPPORTS", "AFFECTS", "CHANGES", "GROUPS", "REPEATS",
                    "DOCUMENTS", "ASSIGNED_TO", "RAISED_BY"):
            n = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n").single()["n"]
            print(f"    {rel:20s} {n:>8,}")
        impact = s.run(
            "MATCH ()-[r:SUPPORTS]->() WHERE r.carries_impact RETURN count(r) AS n"
        ).single()["n"]
        print(f"    of those SUPPORTS, carrying impact: {impact:,}")

    driver.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Load the dataset into Neo4j.")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--wipe", action="store_true", help="clear the database first")
    args = ap.parse_args()
    uri, user, password = read_env()
    print(f"  target: {uri}")
    load(uri, user, password, args.batch, args.wipe)


if __name__ == "__main__":
    main()
