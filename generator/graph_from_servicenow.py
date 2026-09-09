"""Build the Neo4j graph by reading ServiceNow, not by reading the local files.

This is the article's actual premise and the previous loader quietly broke it. That one
read `dataset/*.jsonl` and wrote a perfect graph, which meant the comparison was run
against files rather than against the platform. Anything ServiceNow does to the data on
the way in and out was invisible: the two halves of every field, the timezone the display
half is rendered in, references that arrive as a sys_id rather than a name, and paging.

Reading it back through the platform is not ceremony. It is the only way the graph
reflects what a reader would actually get, and it is where three of the article's
teaching moments live.

⛔ EVERY READ GOES THROUGH snowloader. It is the package the article teaches, so the
article and the code have to agree, and its CMDBLoader returns the dependency edges with
the relationship type and its direction already resolved, which is the hardest part.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import argparse
import pathlib
import re
import time
from typing import Any, Iterator

from neo4j import GraphDatabase
from snowloader import (
    ChangeLoader,
    CMDBLoader,
    IncidentLoader,
    KnowledgeBaseLoader,
    ProblemLoader,
    RelationshipLoader,
    SnowConnection,
)

from load_neo4j import CLASS_LABELS, CONSTRAINTS, IMPACT_TYPES, INDEXES

from env import env_path  # noqa: E402
ENV_PATH = env_path()


def read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in ENV_PATH.read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return env


def connect(env: dict[str, str]) -> SnowConnection:
    """Open the ServiceNow connection the article uses.

    ⛔ display_value="all" is the whole lesson in one argument. The default returns only
    the DISPLAY half of every field, and for a timestamp that is the signed in user's
    local clock rather than the stored instant. Take the convenient half, call it UTC,
    and every join in this file is silently wrong by that account's offset.
    """
    host = env["SERVICENOW_INSTANCE"]
    return SnowConnection(
        instance_url=host if host.startswith("http") else f"https://{host}",
        username=env["SERVICENOW_USER"],
        password=env["SERVICENOW_PASSWORD"],
        display_value="all",
        # ⛔ THE DEFAULTS ASSUME A HEALTHY INSTANCE. A developer instance answered a
        # full incident page in more than 60 seconds and snowloader gave up after three
        # retries. Smaller pages and a longer patience, with a pause between requests so
        # a shared instance is not hammered while a person is using it.
        page_size=200,
        timeout=180,
        max_retries=5,
        retry_backoff=3.0,
        request_delay=0.05,
    )


def half(value: Any, want: str = "value") -> Any:
    """Pull one half of a field that ServiceNow answered twice.

    want="value"          the stored half. Correct for timestamps and for references.
    want="display_value"  the shown half. Correct for a name a person reads.
    """
    if isinstance(value, dict):
        return value.get(want, value.get("value"))
    return value


def stamp(value: Any) -> str | None:
    """A ServiceNow timestamp as something Neo4j will accept.

    Always the STORED half, which is UTC, and always given an explicit zone. A naive
    timestamp is the most expensive missing information in this whole article.
    """
    raw = half(value, "value")
    if not raw:
        return None
    return str(raw).replace(" ", "T") + "Z"


def batched(it: Iterator[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for row in it:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def run(session, cypher: str, source: Iterator[dict], size: int, label: str,
        **params) -> int:
    total, t0 = 0, time.time()
    for chunk in batched(source, size):
        session.run(cypher, rows=chunk, **params)
        total += len(chunk)
        print(f"    {label:24s} {total:>8,}", end="\r", flush=True)
    print(f"    {label:24s} {total:>8,}  in {time.time() - t0:.0f}s")
    return total


# ── reading each table through snowloader ──────────────────────────────────────────

def read_cis(conn: SnowConnection, query: str, limit: int | None) -> tuple[list, list]:
    """Configuration items, then every dependency edge, in two bulk reads.

    ⛔ THIS USED `CMDBLoader(include_relationships=True)` AND IT HUNG FOR 112 MINUTES.
    That option fetches cmdb_rel_ci separately for EVERY configuration item, so on this
    instance it is 19,195 requests rather than one read of a 40,709 row table. Running
    them 16 at a time did not fix the shape, it just made 16 requests hang at once: the
    process sat at 0% CPU with sixteen sockets in CLOSE_WAIT, having printed nothing,
    while the instance answered a count in 1.8 seconds the whole time.

    It is the same mistake Part 7 section 73 warns about, one round trip per row, made
    against ServiceNow instead of against Neo4j. `RelationshipLoader` reads the whole
    table page by page, which is roughly 204 requests at 200 rows a page.

    ⛔ AND IT PRINTS PROGRESS. The old version could not tell you whether it was working
    or stuck, and that is why nobody noticed for nearly two hours. A read with no output
    is indistinguishable from a hang.
    """
    t0 = time.time()
    loader = CMDBLoader(conn, query=query, include_relationships=False)
    docs = loader.concurrent_load(max_workers=8)
    if limit:
        docs = docs[:limit]
    print(f"    configuration items      {len(docs):>8,}  in {time.time() - t0:.0f}s")

    items, by_id = [], set()
    for d in docs:
        m = d.metadata
        sid = half(m.get("sys_id"))
        if not sid:
            continue
        by_id.add(sid)
        items.append({
            "sys_id": sid,
            "name": half(m.get("name"), "display_value") or "",
            "sys_class_name": half(m.get("sys_class_name")) or "cmdb_ci",
            "serial_number": half(m.get("serial_number")) or "",
            "environment": half(m.get("environment"), "display_value") or "",
            "operational_status": half(m.get("operational_status")) or "",
            "install_status": half(m.get("install_status")) or "",
            "discovery_source": half(m.get("discovery_source"), "display_value") or "",
            "last_discovered": stamp(m.get("last_discovered")),
        })

    # ⛔ PARENT AND CHILD COME OFF THE ROW, so there is no direction to infer. The old
    # code read each edge from one end and worked out which way it pointed from whether
    # snowloader called it inbound or outbound, which is the single modelling mistake
    # that inverts every answer while still returning rows. cmdb_rel_ci already has both
    # columns, and the type is written "parent descriptor::child descriptor" to match.
    t0, edges, seen = time.time(), [], 0
    for d in RelationshipLoader(conn).lazy_load():
        seen += 1
        if seen % 2000 == 0:
            print(f"    dependency edges         {seen:>8,}", end="\r", flush=True)
        m = d.metadata
        parent, child = half(m.get("parent_sys_id")), half(m.get("child_sys_id"))
        # An edge to something outside the item set cannot be drawn, and keeping it would
        # make the relationship count disagree with the graph that gets built.
        if not parent or not child or parent not in by_id or child not in by_id:
            continue
        edges.append({"parent": parent, "child": child,
                      "type_name": half(m.get("type")) or ""})
    print(f"    dependency edges         {len(edges):>8,}  of {seen:,} rows, "
          f"in {time.time() - t0:.0f}s")
    return items, edges


def read_incidents(conn: SnowConnection, query: str, limit: int | None) -> list[dict]:
    out = []
    for d in IncidentLoader(conn, query=query).load(limit=limit):
        m = d.metadata
        out.append({
            "number": half(m.get("number")) or "",
            "short_description": half(m.get("short_description"), "display_value") or "",
            "description": half(m.get("description"), "display_value") or "",
            "category": half(m.get("category"), "display_value") or "",
            "priority": half(m.get("priority")) or "",
            # ⛔ The stored half. The shown half is the signed in user's local clock.
            "opened_at": stamp(m.get("opened_at")),
            "resolved_at": stamp(m.get("resolved_at")),
            "close_notes": half(m.get("close_notes"), "display_value") or "",
            # A reference field: the sys_id is what joins, the display value is a name.
            "ci_sys_id": half(m.get("cmdb_ci"), "value") or "",
            "text": d.page_content or "",
        })
    return out


def read_changes(conn: SnowConnection, query: str, limit: int | None) -> list[dict]:
    out = []
    for d in ChangeLoader(conn, query=query).load(limit=limit):
        m = d.metadata
        out.append({
            "number": half(m.get("number")) or "",
            "short_description": half(m.get("short_description"), "display_value") or "",
            "change_type": half(m.get("type"), "display_value") or "",
            "opened_at": stamp(m.get("opened_at")),
            "planned_start": stamp(m.get("start_date")),
            "planned_end": stamp(m.get("end_date")),
            "actual_start": stamp(m.get("work_start")),
            "actual_end": stamp(m.get("work_end")),
            "ci_sys_id": half(m.get("cmdb_ci"), "value") or "",
        })
    return out


def read_problems(conn: SnowConnection, query: str, limit: int | None) -> list[dict]:
    out = []
    for d in ProblemLoader(conn, query=query).load(limit=limit):
        m = d.metadata
        out.append({
            "number": half(m.get("number")) or "",
            "short_description": half(m.get("short_description"), "display_value") or "",
            "cause_notes": half(m.get("cause_notes"), "display_value") or "",
            "workaround": half(m.get("work_around"), "display_value") or "",
            "opened_at": stamp(m.get("opened_at")),
            "ci_sys_id": half(m.get("cmdb_ci"), "value") or "",
        })
    return out


def read_knowledge(conn: SnowConnection, limit: int | None) -> list[dict]:
    out = []
    for d in KnowledgeBaseLoader(conn).load(limit=limit):
        m = d.metadata
        out.append({
            "number": half(m.get("number")) or half(m.get("sys_id")) or "",
            "short_description": half(m.get("short_description"), "display_value") or "",
            "text": d.page_content or "",
            "category": half(m.get("topic"), "display_value") or "",
        })
    return out


# ── writing the graph ──────────────────────────────────────────────────────────────

def build(env: dict[str, str], query: str, limit: int | None, batch: int,
          wipe: bool) -> None:
    conn = connect(env)
    driver = GraphDatabase.driver(
        env["NEO4J_URI"], auth=(env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"]))
    driver.verify_connectivity()

    print("  reading ServiceNow through snowloader")
    t0 = time.time()
    items, edges = read_cis(conn, query, limit)
    print(f"    configuration items      {len(items):>8,}")
    print(f"    dependency edges         {len(edges):>8,}")
    incidents = read_incidents(conn, "", limit)
    print(f"    incidents                {len(incidents):>8,}")
    changes = read_changes(conn, "", limit)
    print(f"    changes                  {len(changes):>8,}")
    problems = read_problems(conn, "", limit)
    print(f"    problems                 {len(problems):>8,}")
    knowledge = read_knowledge(conn, limit)
    print(f"    knowledge articles       {len(knowledge):>8,}")
    print(f"  read in {time.time() - t0:.0f}s\n")

    linked = sum(1 for i in incidents if i["ci_sys_id"])
    print(f"  incidents carrying a configuration item: {linked:,} of {len(incidents):,}"
          f"  ({linked / max(1, len(incidents)):.0%})")
    if edges and not linked:
        print("  ⚠ every incident came back unlinked. Either the loader never set "
              "cmdb_ci, or this account cannot read the field.")

    with driver.session() as s:
        if wipe:
            print("\n  clearing the database")
            while s.run("MATCH (n) WITH n LIMIT 20000 DETACH DELETE n "
                        "RETURN count(n) AS n").single()["n"]:
                pass
        print("  constraints and indexes first")
        for stmt in CONSTRAINTS + INDEXES:
            s.run(stmt)
        s.run("CALL db.awaitIndexes(300)")

        print("  writing the graph")
        run(s, """
            UNWIND $rows AS row
            MERGE (c:ConfigurationItem {sys_id: row.sys_id})
            SET c.name = row.name, c.key = row.serial_number,
                c.sys_class_name = row.sys_class_name,
                c.environment = row.environment,
                c.operational_status = row.operational_status,
                c.install_status = row.install_status,
                c.discovery_source = row.discovery_source,
                c.last_discovered = CASE WHEN row.last_discovered IS NULL
                                    THEN NULL ELSE datetime(row.last_discovered) END
        """, iter(items), batch, "configuration items")

        for cls, labels in CLASS_LABELS.items():
            extra = [x for x in labels if x != "ConfigurationItem"]
            if extra:
                s.run(f"MATCH (c:ConfigurationItem {{sys_class_name: $cls}}) "
                      f"SET c:{':'.join(extra)}", cls=cls)

        run(s, """
            UNWIND $rows AS row
            MATCH (p:ConfigurationItem {sys_id: row.parent})
            MATCH (c:ConfigurationItem {sys_id: row.child})
            MERGE (c)-[r:SUPPORTS {type_name: row.type_name}]->(p)
            SET r.carries_impact = row.type_name IN $impact
        """, iter(edges), batch, "dependency edges", impact=sorted(IMPACT_TYPES))

        run(s, """
            UNWIND $rows AS row
            MERGE (i:Incident {number: row.number})
            SET i.short_description = row.short_description,
                i.description = row.description, i.category = row.category,
                i.priority = row.priority, i.close_notes = row.close_notes,
                i.text = row.text,
                i.opened_at = CASE WHEN row.opened_at IS NULL
                              THEN NULL ELSE datetime(row.opened_at) END,
                i.resolved_at = CASE WHEN row.resolved_at IS NULL
                                THEN NULL ELSE datetime(row.resolved_at) END
        """, iter(incidents), batch, "incidents")

        run(s, """
            UNWIND $rows AS row
            WITH row WHERE row.ci_sys_id <> ''
            MATCH (i:Incident {number: row.number})
            MATCH (c:ConfigurationItem {sys_id: row.ci_sys_id})
            MERGE (i)-[:AFFECTS]->(c)
        """, iter(incidents), batch, "incident to item")

        run(s, """
            UNWIND $rows AS row
            MERGE (c:Change {number: row.number})
            SET c.short_description = row.short_description,
                c.change_type = row.change_type,
                c.opened_at = CASE WHEN row.opened_at IS NULL
                              THEN NULL ELSE datetime(row.opened_at) END,
                c.planned_start = CASE WHEN row.planned_start IS NULL
                                  THEN NULL ELSE datetime(row.planned_start) END,
                c.actual_end = CASE WHEN row.actual_end IS NULL
                               THEN NULL ELSE datetime(row.actual_end) END
        """, iter(changes), batch, "changes")

        run(s, """
            UNWIND $rows AS row
            WITH row WHERE row.ci_sys_id <> ''
            MATCH (ch:Change {number: row.number})
            MATCH (c:ConfigurationItem {sys_id: row.ci_sys_id})
            MERGE (ch)-[:CHANGES]->(c)
        """, iter(changes), batch, "change to item")

        run(s, """
            UNWIND $rows AS row
            MERGE (p:Problem {number: row.number})
            SET p.short_description = row.short_description,
                p.cause_notes = row.cause_notes, p.workaround = row.workaround
        """, iter(problems), batch, "problems")

        run(s, """
            UNWIND $rows AS row
            MERGE (k:KnowledgeArticle {number: row.number})
            SET k.short_description = row.short_description,
                k.text = row.text, k.category = row.category
        """, iter(knowledge), batch, "knowledge articles")

        print("\n  what landed:")
        for label in ("ConfigurationItem", "Server", "Service", "Incident", "Change",
                      "Problem", "KnowledgeArticle"):
            n = s.run(f"MATCH (n:{label}) RETURN count(n) AS n").single()["n"]
            print(f"    {label:20s} {n:>8,}")
        for rel in ("SUPPORTS", "AFFECTS", "CHANGES"):
            n = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n").single()["n"]
            print(f"    {rel:20s} {n:>8,}")

    driver.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the graph by reading ServiceNow through snowloader.")
    ap.add_argument("--query", default="", help="encoded query to narrow the CI read")
    ap.add_argument("--limit", type=int, help="stop after this many rows per table")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--wipe", action="store_true")
    args = ap.parse_args()
    build(read_env(), args.query, args.limit, args.batch, args.wipe)


if __name__ == "__main__":
    main()
