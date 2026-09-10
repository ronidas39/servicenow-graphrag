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
import collections
import json
import pathlib

# ⛔ CHECKPOINTS LIVE BESIDE THE DATASET, so a read that dies costs the last page rather
# than the last hour. The first version had none, and an hour of incident reading was
# thrown away when one truncated page hit the default on_error="raise".
CACHE = pathlib.Path(__file__).resolve().parent.parent / "dataset" / ".checkpoints"
CACHE.mkdir(parents=True, exist_ok=True)
import re
import time
from typing import Any, Iterator

from neo4j import GraphDatabase
from snowloader import (
    FileCheckpoint,
    ChangeLoader,
    CMDBLoader,
    IncidentLoader,
    KnowledgeBaseLoader,
    ProblemLoader,
    RelationshipLoader,
    SnowConnection,
)
from snowloader.fields import is_sys_id

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
        # ⛔ 50, NOT 200. With display_value="all" every field arrives twice, and
        # 200 rows put a page past 650KB, which is where this instance starts
        # truncating its own JSON. Measured: failures at 669,895 and 858,873 bytes.
        page_size=50,
        timeout=180,
        max_retries=5,
        retry_backoff=3.0,
        request_delay=0.05,
    )


def half(value: Any, want: str = "value") -> Any:
    """Pull one half of a field that ServiceNow answered twice.

    want="value"          the stored half. Correct for timestamps and for references.
    want="display_value"  the shown half. Correct for a name a person reads.

    ⛔ THIS IS ONLY SAFE WHEN THE FIELD STILL HAS TWO HALVES. Once a loader has curated a
    field down to one string, `half` has nothing to choose between and hands back
    whichever half that loader picked. See `joins_on` below, which is what a reference
    field must go through instead.
    """
    if isinstance(value, dict):
        return value.get(want, value.get("value"))
    return value


def joins_on(m: dict, field: str) -> str:
    """The sys_id of a reference field, whatever shape the loader left it in.

    ⛔ THIS FUNCTION EXISTS BECAUSE THE OBVIOUS CODE LOADED ZERO EDGES AND LOOKED FINE.
    Both incidents and changes were read with the same line, `half(m.get("cmdb_ci"))`,
    and changes produced 10,877 relationships while incidents produced none. Every count
    in between was plausible: 66,127 incidents in, 55,803 with a linked item, 55,803 rows
    processed by the write. Only the relationship count at the end was zero, and only
    because section 75 asks for it.

    The cause is in the loaders, not in the data. `ChangeLoader` curates `cmdb_ci` as the
    stored half, and `IncidentLoader` curates the same key as the SHOWN half, so on
    incidents that field holds `mer-dev-db-192`, a name, and matching a name against
    sys_id finds nothing. One key, two meanings, two loaders, one package.

    Resolving by name is not the fix, and measuring says so: all 12,844 distinct
    references do resolve to a name in this estate, but 634 names sit on more than one
    item and 426 of the references land on one of them. A display value is a label. It
    was never a key, and `MacBook Pro 17"` is on 173 different items.

    The sys_id was there the whole time. snowloader's `expand_reference_keys` puts the
    second half of every field beside the first under a predictable name, and the
    `_sys_id` suffix means precisely "you can join on this". So: take the companion key,
    and accept the curated key only when it actually looks like a sys_id.
    """
    companion = half(m.get(f"{field}_sys_id")) or ""
    if companion:
        return str(companion)
    direct = half(m.get(field)) or ""
    return str(direct) if is_sys_id(direct) else ""


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



# ⛔ EVERY READ GOES THROUGH HERE, AND THE FIRST VERSION USED NONE OF IT. `load()` is
# sequential, intolerant and unresumable, and I called it for incidents, changes, problems
# and knowledge while calling `concurrent_load` for the configuration items. Reading
# 66,127 incidents that way took over an hour and then died on a truncated page.
#
# snowloader already has the four things that fix it:
#
#   concurrent_lazy_load   pages fetched in parallel, results streamed, so memory stays
#                          flat and wall clock drops roughly with max_workers
#   on_error="skip"        one malformed page does not kill the run. This instance
#                          truncates its JSON somewhere past 700KB, and the default
#                          "raise" turned that into a dead read after 18 retries
#   checkpoint             a resumable cursor, so a failure costs the last page and not
#                          the last hour
#   keyset                 a sys_id cursor instead of a deep offset, which is exactly what
#                          Part 5 section 48 tells the reader to do
#
# ⛔ AND IT PRINTS. A read with no output cannot be told apart from a hang, which is how
# the first version cost 112 minutes before anybody looked.
def stream(loader, label: str, limit: int | None, workers: int = 16, attempts: int = 6):
    """Read one table in parallel, survive a bad page, and say how it is going.

    ⛔ `on_error="skip"` IS NOT ENOUGH ON THIS INSTANCE, and finding that out cost a
    second full run. The instance truncates its JSON somewhere past 650KB. snowloader
    retries, and on the last attempt a partially parsed page yields a STRING where the
    document builder expects a mapping:

        File "snowloader/loaders/cmdb.py", line 102, in _record_to_document
            sys_id = _raw_value(record.get("sys_id"))
        AttributeError: 'str' object has no attribute 'get'

    That happens after the page has been accepted, inside the conversion, so `on_error`
    never sees it and the whole read dies. The checkpoint is what makes this survivable:
    each attempt resumes where the last one stopped rather than starting again.
    """
    t0, out = time.time(), []
    ckpt = FileCheckpoint(CACHE / f"{label}.checkpoint.json")
    for attempt in range(1, attempts + 1):
        try:
            for doc in loader.concurrent_lazy_load(max_workers=workers, on_error="skip",
                                                   checkpoint=ckpt):
                out.append(doc)
                if len(out) % 2000 == 0:
                    print(f"    {label:24s} {len(out):>8,}", end="\r", flush=True)
                if limit and len(out) >= limit:
                    break
            break
        except (AttributeError, TypeError, ValueError) as exc:
            # ⛔ NAMED EXCEPTIONS ONLY. A bare except here would swallow a genuine bug in
            # the mapping code below and report a short read as a complete one.
            if attempt == attempts:
                print(f"\n    {label:24s} gave up after {attempts} attempts: {exc}")
                break
            print(f"\n    {label:24s} page failed ({type(exc).__name__}), resuming from "
                  f"the checkpoint, attempt {attempt + 1} of {attempts}")
    print(f"    {label:24s} {len(out):>8,}  in {time.time() - t0:.0f}s")
    return out


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
    docs = stream(CMDBLoader(conn, query=query, include_relationships=False),
                  "configuration items", limit)

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
            # ⛔ THE COMPANION KEY AGAIN. CMDBLoader curates sys_class_name as the shown
            # half, so this field held "Linux Server" while CLASS_LABELS is keyed on
            # "cmdb_ci_linux_server". Nothing errored. Every item simply stayed a bare
            # ConfigurationItem and the Server and Service counts came back zero.
            "sys_class_name": (half(m.get("sys_class_name_value"))
                               or half(m.get("sys_class_name")) or "cmdb_ci"),
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
    for d in stream(IncidentLoader(conn, query=query), "incidents", limit):
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
            # A reference field. `joins_on` and not `half`, and the docstring on
            # `joins_on` explains what reading this one with `half` actually cost.
            "ci_sys_id": joins_on(m, "cmdb_ci"),
            "text": d.page_content or "",
        })
    return out


def read_changes(conn: SnowConnection, query: str, limit: int | None) -> list[dict]:
    out = []
    for d in stream(ChangeLoader(conn, query=query), "changes", limit):
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
            "ci_sys_id": joins_on(m, "cmdb_ci"),
        })
    return out


def read_problems(conn: SnowConnection, query: str, limit: int | None) -> list[dict]:
    out = []
    for d in stream(ProblemLoader(conn, query=query), "problems", limit):
        m = d.metadata
        out.append({
            "number": half(m.get("number")) or "",
            "short_description": half(m.get("short_description"), "display_value") or "",
            "cause_notes": half(m.get("cause_notes"), "display_value") or "",
            "workaround": half(m.get("work_around"), "display_value") or "",
            "opened_at": stamp(m.get("opened_at")),
            "ci_sys_id": joins_on(m, "cmdb_ci"),
        })
    return out


def read_knowledge(conn: SnowConnection, limit: int | None) -> list[dict]:
    out = []
    for d in stream(KnowledgeBaseLoader(conn), "knowledge", limit):
        m = d.metadata
        out.append({
            "number": half(m.get("number")) or half(m.get("sys_id")) or "",
            "short_description": half(m.get("short_description"), "display_value") or "",
            "text": d.page_content or "",
            "category": half(m.get("topic"), "display_value") or "",
        })
    return out


# ── writing the graph ──────────────────────────────────────────────────────────────

def cached(label: str, refresh: bool, produce):
    """Read a table once and keep the flattened rows on disk.

    Section 50 of the article says to do exactly this and it is not only advice for the
    reader. A full read of this instance takes about eighty minutes, and the first time the
    write step failed on a constraint, all of it had to happen again to retry a step that
    takes four minutes. The cache holds the plain dictionaries the readers already produce,
    not loader objects, so nothing has to be reconstructed to use it.
    """
    path = CACHE / f"{label}.rows.json"
    if path.exists() and not refresh:
        rows = json.loads(path.read_text())
        print(f"    {label:24s} {len(rows):>8,}  from cache")
        return rows
    rows = produce()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows


def build(env: dict[str, str], query: str, limit: int | None, batch: int,
          wipe: bool, refresh: bool = False) -> None:
    conn = connect(env)
    driver = GraphDatabase.driver(
        env["NEO4J_URI"], auth=(env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"]))
    driver.verify_connectivity()

    print("  reading ServiceNow through snowloader")
    t0 = time.time()
    pair = cached("cis", refresh, lambda: list(read_cis(conn, query, limit)))
    items, edges = pair[0], pair[1]
    print(f"    configuration items      {len(items):>8,}")
    print(f"    dependency edges         {len(edges):>8,}")
    incidents = cached("incidents", refresh, lambda: read_incidents(conn, "", limit))
    print(f"    incidents                {len(incidents):>8,}")
    changes = cached("changes", refresh, lambda: read_changes(conn, "", limit))
    print(f"    changes                  {len(changes):>8,}")
    problems = cached("problems", refresh, lambda: read_problems(conn, "", limit))
    print(f"    problems                 {len(problems):>8,}")
    knowledge = cached("knowledge", refresh, lambda: read_knowledge(conn, limit))
    print(f"    knowledge articles       {len(knowledge):>8,}")
    print(f"  read in {time.time() - t0:.0f}s\n")

    linked = sum(1 for i in incidents if i["ci_sys_id"])
    print(f"  incidents carrying a configuration item: {linked:,} of {len(incidents):,}"
          f"  ({linked / max(1, len(incidents)):.0%})")
    if edges and not linked:
        print("  ⚠ every incident came back unlinked. Either the loader never set "
              "cmdb_ci, or this account cannot read the field.")

    # A real CMDB is not the dataset this article generates. Most of the items already on a
    # developer instance carry no serial number at all, and `key` is uniquely constrained,
    # so the second blank one fails the whole write. Counting them here makes the shape of
    # the instance visible before the load rather than as a constraint error 4,800 seconds
    # into it.
    serials = collections.Counter((i.get("serial_number") or "").strip() for i in items)
    blank_key = serials.get("", 0)
    if blank_key:
        print(f"  configuration items with no serial number: {blank_key:,} of "
              f"{len(items):,}  ({blank_key / max(1, len(items)):.0%})")
    shared = {s: n for s, n in serials.items() if s and n > 1}
    if shared:
        worst, n_worst = max(shared.items(), key=lambda kv: kv[1])
        print(f"  serial numbers shared by more than one item: {len(shared):,}, covering "
              f"{sum(shared.values()):,} items. The worst is on {n_worst} of them.")

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
        # ⛔ A SERIAL NUMBER IS NOT A KEY, AND THIS INSTANCE PROVES IT TWICE OVER. `key` is
        # uniquely constrained. Writing the serial number into it failed first on the empty
        # string, because most items on a developer instance have none, and then on a
        # DUPLICATE: fifteen separate configuration items here share the serial L3BB911.
        # The identifier that is actually unique is the one this MERGE already uses, so the
        # key is the sys_id and the serial number keeps its own name.
        run(s, """
            UNWIND $rows AS row
            MERGE (c:ConfigurationItem {sys_id: row.sys_id})
            SET c.name = row.name,
                c.key = row.sys_id,
                c.serial_number = CASE WHEN trim(coalesce(row.serial_number, '')) = ''
                                  THEN NULL ELSE row.serial_number END,
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
    ap.add_argument("--refresh", action="store_true",
                    help="re-read every table instead of using the cached rows")
    args = ap.parse_args()
    build(read_env(), args.query, args.limit, args.batch, args.wipe,
          args.refresh)


if __name__ == "__main__":
    main()

