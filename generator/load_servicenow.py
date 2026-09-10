"""Load the dataset into a real ServiceNow instance.

Two paths, and the difference matters enough that mixing them corrupts the CMDB.

Incidents, changes, problems and knowledge go through a Scripted REST API that inserts
server side with business rules turned off. Measured on a developer instance: 27 records
a second, against 0.16 for the obvious one at a time approach and 2.79 for twenty
parallel workers. The gain is not the network, it is the forty five active insert business
rules on incident and task together, plus the SLA, metric, flow and audit engines that fire
on every insert.

Configuration items do NOT take that path. Writing a CI directly skips the identification
and reconciliation engine, which is the thing that stops two discovery sources creating
two records for one server. Skip it and you manufacture duplicate CIs, which is the one
mistake that would make a CMDB owner discard the whole article.

The loader is restartable. A full run is roughly an hour, and an hour is long enough for
a laptop to sleep, a network to drop or a developer instance to be reclaimed. Progress is
written after every batch, so a restart continues rather than starting again or, worse,
inserting everything twice.

Author: Roni Das
Created: 2026-09-08
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

DATASET = pathlib.Path(__file__).resolve().parent.parent / "dataset"
STATE = DATASET / ".load-progress.json"

# The Scripted REST endpoint created for this article. It inserts server side with
# setWorkflow(false), which is what makes the fast path fast.
#
# ⛔ THE NUMBER IN THAT PATH IS NOT YOURS. ServiceNow builds the path from the API's
# namespace, which is your instance's application scope id, and every instance has a
# different one. Hardcoding mine meant every reader got a 404 from an endpoint that
# exists only on my instance. Part 4 section 37b shows where to read yours off the
# Scripted REST API record, and it goes in .env.local like everything else.
BULK_PATH = os.environ.get("SERVICENOW_BULK_PATH", "")


@dataclass
class Target:
    base: str
    auth: str


def read_env() -> Target:
    """Read credentials from .env.local, which is never committed."""
    env: dict[str, str] = {}
    from env import read_env as _read
    env = _read()
    host = env["SERVICENOW_INSTANCE"]
    base = host if host.startswith("http") else f"https://{host}"
    token = base64.b64encode(
        f"{env['SERVICENOW_USER']}:{env['SERVICENOW_PASSWORD']}".encode()
    ).decode()
    return Target(base=base, auth=f"Basic {token}")


def call(t: Target, path: str, body: dict | None = None,
         method: str = "GET", timeout: int = 300) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(t.base + path, data=data, method=method)
    req.add_header("Authorization", t.auth)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


# ── progress, so a restart continues rather than repeats ───────────────────────────

def load_state() -> dict[str, int]:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def save_state(state: dict[str, int]) -> None:
    """Write progress atomically, because a torn progress file is worse than none.

    ⛔ write_text TRUNCATES AND THEN WRITES. Anything reading the file in that window sees
    an empty or partial one, and this file is written after every single batch for an hour.
    A reader saw {"incidents": 50} where the real state had six keys, which is exactly the
    shape of a truncated write that happened to remain parseable.

    That matters beyond a confusing read. If the process is killed mid-write, the loader's
    own memory of where it got to is destroyed, and the next run either repeats an hour of
    work or, worse, believes it is further along than it is. Writing to a temporary file in
    the same directory and renaming is atomic on POSIX, so a reader sees either the old
    state or the new one and never a half of either.
    """
    tmp = STATE.with_suffix(STATE.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(STATE)


def read_rows(name: str) -> list[dict]:
    path = DATASET / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── the fast path, for records that are not configuration items ────────────────────

# ⛔ A CHOICE FIELD IS NOT FREE TEXT, AND SERVICENOW WILL NOT TELL YOU. The generator's
# fault families are the taxonomy the article reasons about, but the instance only
# accepts its own six categories, read from sys_choice. The close code being used,
# "Solved (Permanently)", is not on the list at all and was accepted anyway, which is
# the worst kind of silent success: the record exists and the field is meaningless.
#
# The fault family is kept in subcategory where a real choice exists, so nothing is
# thrown away, and the article says plainly that a real instance constrains you to its
# own list. That constraint is worth a paragraph rather than an apology.
CATEGORY_FOR_FAMILY = {
    "latency": "software", "errors": "software", "batch": "software",
    "certificate": "software", "auth": "password_reset",
    "capacity": "hardware", "memory": "hardware",
    "replication": "database", "database": "database",
}
SUBCATEGORY_FOR_FAMILY = {"memory": "memory", "database": "sql server",
                          "replication": "oracle", "batch": "email"}


def to_incident(row: dict, ci_sys_id: dict[str, str] | None = None) -> dict:
    """Map a generated incident onto real ServiceNow field names.

    ⛔ cmdb_ci IS THE POINT. Without it the instance holds sixty thousand tickets
    attached to nothing, and the graph cannot be rebuilt from ServiceNow at all.
    """
    out = {
        "short_description": row["short_description"],
        "description": row["description"],
        "priority": str(row["priority"]),
        "urgency": str(min(3, row["priority"])),
        "category": CATEGORY_FOR_FAMILY.get(row["category"], "software"),
        "opened_at": row["opened_at"].replace("T", " ")[:19],
        "correlation_id": row["number"],
    }
    sub = SUBCATEGORY_FOR_FAMILY.get(row["category"])
    if sub:
        out["subcategory"] = sub
    sid = (ci_sys_id or {}).get(row.get("ci_key") or "")
    if sid:
        out["cmdb_ci"] = sid
    if row.get("resolved_at"):
        # ⛔ ALL FOUR OR NONE. Setting state to Resolved without close_code, close_notes
        # AND resolved_at makes gr.insert() return null with no error message at all.
        # Measured field by field: state alone inserts 0 of 6, state with close_code
        # inserts 0 of 6, all four together inserts 6 of 6. Nothing tells you which
        # field is missing, which is why the loader below counts what landed rather
        # than what it sent.
        out["resolved_at"] = row["resolved_at"].replace("T", " ")[:19]
        out["closed_at"] = row["resolved_at"].replace("T", " ")[:19]
        out["close_notes"] = row.get("close_notes") or "Resolved."
        out["close_code"] = "Solution provided"
        out["state"] = "6"
    return out


def to_change(row: dict, ci_sys_id: dict[str, str] | None = None) -> dict:
    out = {
        "short_description": row["short_description"],
        "description": row["description"],
        "type": row["change_type"],
        "risk": {"low": "4", "moderate": "3", "high": "2"}[row["risk"]],
        "start_date": row["planned_start"].replace("T", " ")[:19],
        "end_date": row["planned_end"].replace("T", " ")[:19],
        "correlation_id": row["number"],
    }
    sid = (ci_sys_id or {}).get(row.get("ci_key") or "")
    if sid:
        out["cmdb_ci"] = sid
    if row.get("actual_start"):
        out["work_start"] = row["actual_start"].replace("T", " ")[:19]
    if row.get("actual_end"):
        out["work_end"] = row["actual_end"].replace("T", " ")[:19]
    return out


def to_problem(row: dict, ci_sys_id: dict[str, str] | None = None) -> dict:
    out = {
        "short_description": row["short_description"],
        "description": row["description"],
        "cause_notes": row["cause_notes"],
        "work_around": row["workaround"],
        "correlation_id": row["number"],
    }
    sid = (ci_sys_id or {}).get(row.get("ci_key") or "")
    if sid:
        out["cmdb_ci"] = sid
    return out


# ⛔ kb_knowledge HAS A MANDATORY REFERENCE AND NOTHING SAYS SO. Without
# kb_knowledge_base every insert returns null with no error, exactly like the resolved
# state trap on incidents. Measured: 0 of 4 without it, 4 of 4 with it. Resolved by
# TITLE at run time, because the sys_id differs on every instance.
KB_BASE: str = ""


def resolve_knowledge_base(t: Target) -> str:
    """Find a knowledge base to file the articles under."""
    for title in ("IT", "Knowledge"):
        q = urllib.parse.urlencode({"sysparm_query": f"title={title}",
                                    "sysparm_fields": "sys_id", "sysparm_limit": 1})
        found = call(t, f"/api/now/table/kb_knowledge_base?{q}").get("result", [])
        if found:
            return found[0]["sys_id"]
    any_base = call(t, "/api/now/table/kb_knowledge_base?sysparm_fields=sys_id"
                       "&sysparm_limit=1").get("result", [])
    if not any_base:
        raise SystemExit("\n  STOPPED: this instance has no knowledge base to file "
                         "articles under, and kb_knowledge_base is mandatory.")
    return any_base[0]["sys_id"]


def to_knowledge(row: dict, ci_sys_id: dict[str, str] | None = None) -> dict:
    """Knowledge articles were missing from the loader entirely, while the docstring
    claimed they went through the fast path."""
    return {
        "kb_knowledge_base": KB_BASE,
        "short_description": row["short_description"],
        "text": row["text"],
        "article_type": "text",
        "workflow_state": "published",
    }


MAPPERS = {
    "incidents": ("incident", to_incident),
    "changes": ("change_request", to_change),
    "problems": ("problem", to_problem),
    "knowledge": ("kb_knowledge", to_knowledge),
}



def already_there(t: Target, table: str, numbers: list[str]) -> set[str]:
    """Which of these correlation ids the instance already holds.

    ⛔ THIS IS WHY correlation_id IS WRITTEN ON EVERY ROW. A gateway timeout after the
    server committed looks exactly like one before it, so a blind retry inserts the batch
    twice. Reproduced against a stub: three attempts, 150 rows inserted, 50 unique. The
    id was being written and never read, which made it decoration rather than a guard.
    """
    # ⛔ PROVE THE FIELD IS REAL BEFORE TRUSTING ANY ANSWER FROM IT. ServiceNow ignores a
    # query on a column that does not exist rather than rejecting it, so filtering
    # kb_knowledge on correlation_id returned every row in the table, and this function
    # then read a field that was not in the response, found nothing, and reported "none
    # of these are loaded yet". A guard against duplicates that always says "go ahead" is
    # worse than no guard, because the loader prints that it checked.
    #
    # Two tables in this project have no correlation_id: kb_knowledge and cmdb_rel_ci.
    # Neither says so. One impossible value settles it: on a real field nothing matches,
    # on a phantom one everything does.
    probe = urllib.parse.urlencode({
        "sysparm_query": "correlation_id=zzz-this-value-cannot-exist-zzz",
        "sysparm_count": "true"})
    if int(call(t, f"/api/now/stats/{table}?{probe}")["result"]["stats"]["count"]):
        raise SystemExit(
            f"\n  STOPPED: {table} has no usable correlation_id field. A filter on it "
            f"matched an impossible value, which means the column does not exist and "
            f"every query against it silently returns the whole table. This phase cannot "
            f"be made idempotent that way.")

    found: set[str] = set()
    for i in range(0, len(numbers), 60):
        window = numbers[i:i + 60]
        q = urllib.parse.urlencode({
            "sysparm_query": "correlation_idIN" + ",".join(window),
            "sysparm_fields": "correlation_id",
            "sysparm_limit": len(window),
        })
        for row in call(t, f"/api/now/table/{table}?{q}", timeout=180).get("result", []):
            cid = row.get("correlation_id")
            if cid:
                found.add(cid)
    return found


def load_fast(t: Target, name: str, rows: list[dict], start_at: int,
              batch: int, workers: int, state: dict,
              ci_sys_id: dict[str, str] | None = None) -> int:
    """Load through the server side script. Returns the index reached."""
    table, mapper = MAPPERS[name]
    todo = rows[start_at:]
    if not todo:
        print(f"  {name:12s} already complete ({len(rows):,})")
        return len(rows)

    print(f"  {name:12s} {len(todo):,} to load into {table}, "
          f"resuming at {start_at:,}, {workers} workers x {batch}")

    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    done = start_at
    t0 = time.time()

    def send(chunk: list[dict]) -> int:
        pending = list(chunk)

        # ⛔ ASK BEFORE SENDING, NOT ONLY AFTER A TIMEOUT. This check used to live only in
        # the retry path, so it protected against a gateway dying after the commit and
        # protected against nothing else. Re-running the loader over rows that had already
        # landed raised no exception, never reached the retry, and inserted them a second
        # time. Measured: three problems loaded twice by a single re-run, on a table that
        # already held all 900.
        #
        # "Safe to restart" has to mean safe to run again, which is what a person actually
        # does. The cost is one query per batch.
        already = already_there(t, table, [r["number"] for r in pending])
        if already:
            pending = [r for r in pending if r["number"] not in already]
            if not pending:
                return len(already)
        # Rows confirmed present count as landed, or the shortfall check below fires on
        # a batch that is genuinely complete.
        landed = len(already)
        for attempt in range(4):
            if not pending:
                break
            payload = {"table": table,
                       "rows": [mapper(r, ci_sys_id) for r in pending],
                       "skip_business_rules": True}
            try:
                res = call(t, BULK_PATH, payload, "POST", timeout=600)
                return landed + int(res.get("result", res).get("inserted", 0))
            except (urllib.error.HTTPError, urllib.error.URLError,
                    TimeoutError, json.JSONDecodeError) as e:
                if isinstance(e, urllib.error.HTTPError) and e.code not in (
                        429, 500, 502, 503, 504):
                    raise
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)
                # ⛔ ASK WHAT LANDED BEFORE SENDING IT AGAIN. Without this the retry
                # replays rows the server already committed and the instance ends up
                # with duplicates that no count in this script would ever notice.
                done = already_there(t, table, [r["number"] for r in pending])
                landed += len(done)
                pending = [r for r in pending if r["number"] not in done]
                if done:
                    print(f"    retry: {len(done)} of that batch had already landed",
                          flush=True)
        return landed

    # Batches are submitted in order and progress is only advanced for a contiguous run
    # of completed batches, so a crash never records more than actually landed.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for group_start in range(0, len(chunks), workers):
            group = chunks[group_start:group_start + workers]
            counts = list(pool.map(send, group))
            sent = sum(len(c) for c in group)
            landed = sum(counts)
            # ⛔ COUNT WHAT LANDED, NOT WHAT WAS SENT. A first version advanced progress
            # by the size of the batch, so a run that silently inserted 19 of 200
            # recorded 200 and a restart skipped the 181 that never arrived. Failing
            # loudly here is the whole reason the earlier defect was found at all.
            if landed < sent:
                raise SystemExit(
                    f"\n  STOPPED: sent {sent} rows to {table}, the server inserted "
                    f"{landed}.\n  Progress is at {done:,} and is safe to resume from.\n"
                    f"  A silent shortfall means a field was rejected. Test the mapping "
                    f"field by field before rerunning."
                )
            done += landed
            state[name] = done
            save_state(state)
            rate = (done - start_at) / max(0.1, time.time() - t0)
            left = (len(rows) - done) / max(0.1, rate)
            print(f"    {done:>7,}/{len(rows):,}  {sum(counts):>4} in this group  "
                  f"{rate:>5.1f} rec/s  about {left / 60:>4.0f} min left", flush=True)
    return done


# ── the correct path for configuration items ───────────────────────────────────────

SYSIDS = DATASET / ".ci-sysids.json"


def load_sysids() -> dict[str, str]:
    return json.loads(SYSIDS.read_text()) if SYSIDS.exists() else {}


def ci_progress(rows: list[dict], sys_ids: dict[str, str], recorded: int) -> int:
    """How far the configuration item phase really got.

    ⛔ A COUNTER AND REALITY DIVERGED, AND THE COUNTER WON. After a full run the sys_id
    map held all 11,891 entries and every item was on the instance, while the progress
    file said 325. Resuming from the counter re-ran the entire phase: twenty five wasted
    minutes, and only harmless because the identification engine updates rather than
    duplicating, which is exactly why it is used.

    The sys_id map IS the record of what landed, so it is the authority. The counter is
    kept only as a floor, in case the map is missing.
    """
    landed = sum(1 for r in rows if r["key"] in sys_ids)
    return max(recorded, landed)


def load_cis(t: Target, rows: list[dict], start_at: int, batch: int, state: dict,
             sys_ids: dict[str, str]) -> int:
    """Load configuration items through the identification engine.

    This is slower than the fast path on purpose. The engine decides whether an incoming
    payload is a new item, an update to one that exists, or ambiguous. Bypassing it is
    how a CMDB ends up with two records for one server.
    """
    todo = rows[start_at:]
    if not todo:
        print(f"  {'cis':12s} already complete ({len(rows):,})")
        return len(rows)

    print(f"  {'cis':12s} {len(todo):,} through the identification engine, "
          f"resuming at {start_at:,}")
    done = start_at
    t0 = time.time()

    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        # ⛔ THE DATA SOURCE IS A QUERY PARAMETER, NOT A FIELD ON THE ITEM. Sending it
        # inside values gives "invalid data source [null]" and rejects the whole batch.
        payload = {"items": [{
            "className": r["sys_class_name"],
            # ⛔ THE SYNTHETIC KEY DOES NOT GO IN serial_number, AND IT USED TO.
            # serial_number is an identification attribute on the out of box CMDB
            # identification rules for hardware classes, so the engine may identify,
            # merge or split real configuration items on whatever is in it. Writing
            # `host-payments-prd-042` there invites the IRE to reconcile a generated
            # row against a real one that happens to share it, and a service does not
            # have a serial number at all. Nothing here read the field back either:
            # every later lookup in this file matches on `name`. So it is gone.
            #
            # If you need to carry an external key into a real CMDB, `sys_object_source`
            # is the field pair ServiceNow provides for it, keyed by the data source
            # named in the query parameter below. `correlation_id` is a task field and
            # does not exist on cmdb_ci.
            "values": {
                "name": r["name"],
                "operational_status": str(r["operational_status"]),
                "install_status": str(r["install_status"]),
            },
        } for r in chunk]}
        try:
            res = call(t, "/api/now/identifyreconcile?sysparm_data_source=ServiceNow",
                       payload, "POST", timeout=600)
            items = res.get("result", res).get("items", [])
            # ⛔ KEEP THE SYS IDS. Without them nothing can point at a configuration
            # item afterwards: no incident link, no change link, no dependency row, and
            # the graph cannot be rebuilt from ServiceNow at all. The engine returns
            # them in payload order, so they pair with the chunk.
            for r, item in zip(chunk, items):
                sid = item.get("sysId")
                if sid and sid != "Unknown":
                    sys_ids[r["key"]] = sid
            bad = [i for i in items if i.get("errors")]
            if bad:
                msg = bad[0]["errors"][0].get("message", "")[:200]
                raise SystemExit(
                    f"\n  STOPPED: the identification engine rejected "
                    f"{len(bad)} of {len(items)} items.\n  First error: {msg}\n"
                    f"  Progress is at {done:,} and is safe to resume from."
                )
        except urllib.error.HTTPError as e:
            print(f"    identification engine returned {e.code}: {e.read()[:200]!r}")
            raise
        done += len(chunk)
        state["configuration_items"] = done
        save_state(state)
        # Atomic for the same reason as save_state: this map is the loader's only
        # record of which configuration items exist, and a torn copy is unrecoverable.
        _tmp = SYSIDS.with_suffix(SYSIDS.suffix + ".tmp")
        _tmp.write_text(json.dumps(sys_ids))
        _tmp.replace(SYSIDS)
        rate = (done - start_at) / max(0.1, time.time() - t0)
        print(f"    {done:>7,}/{len(rows):,}  {rate:>5.1f} rec/s  "
              f"about {(len(rows) - done) / max(0.1, rate) / 60:>4.0f} min left", flush=True)
    return done


def load_relationships(t: Target, rows: list[dict], start_at: int, batch: int,
                       workers: int, state: dict, sys_ids: dict[str, str]) -> int:
    """Write the dependency rows into cmdb_rel_ci.

    This phase did not exist. Without it the instance has configuration items and no
    edges between them, so a graph built from ServiceNow is a pile of unconnected nodes
    and every question in the article is unanswerable.

    ⛔ The relationship TYPE is a reference to a record, not a string. It is resolved by
    name once, up front, because resolving it per row would be seventeen thousand extra
    round trips.
    """
    todo = rows[start_at:]
    if not todo:
        print(f"  {'relationships':12s} already complete ({len(rows):,})")
        return len(rows)

    wanted = sorted({r["type_name"] for r in rows})
    type_id: dict[str, str] = {}
    for name in wanted:
        q = urllib.parse.urlencode({"sysparm_query": f"name={name}",
                                    "sysparm_fields": "sys_id", "sysparm_limit": 1})
        found = call(t, f"/api/now/table/cmdb_rel_type?{q}").get("result", [])
        if found:
            type_id[name] = found[0]["sys_id"]
    missing = [n for n in wanted if n not in type_id]
    if missing:
        raise SystemExit(f"\n  STOPPED: these relationship types do not exist on the "
                         f"instance: {missing}\n  Nothing was written.")
    print(f"  {'relationships':12s} {len(todo):,} rows, {len(type_id)} types resolved, "
          f"resuming at {start_at:,}")

    # Only rows whose BOTH ends were loaded can be written.
    usable = [r for r in todo
              if r["parent_key"] in sys_ids and r["child_key"] in sys_ids]
    skipped = len(todo) - len(usable)
    if skipped:
        print(f"    {skipped:,} skipped: an endpoint was never loaded")

    chunks = [usable[i:i + batch] for i in range(0, len(usable), batch)]
    done = start_at
    t0 = time.time()

    def send(chunk: list[dict]) -> int:
        # ⛔ cmdb_rel_ci HAS NO correlation_id FIELD, AND SERVICENOW NEVER SAYS SO.
        # A correlation_id was added here to make this phase idempotent, the way it is on
        # incidents and changes. The insert accepted it and returned success, and the
        # field was silently discarded, because that column does not exist on this table.
        #
        # Worse, a QUERY on a field that does not exist is also silently ignored rather
        # than rejected. Filtering cmdb_rel_ci on correlation_idISNOTEMPTY returned all
        # 40,709 rows. So did correlation_idISEMPTY. So did an exact match on a value
        # nothing could hold. A check written against it reported a confident number that
        # meant nothing at all, and it took reading one row field by field to see it.
        #
        # The idempotency key for a relationship is the relationship: parent, type and
        # child are real columns and they discriminate. Verified: the exact triple matches
        # one row, the same pair reversed matches none.
        payload = {"table": "cmdb_rel_ci", "skip_business_rules": True, "rows": [{
            "parent": sys_ids[r["parent_key"]],
            "child": sys_ids[r["child_key"]],
            "type": type_id[r["type_name"]],
        } for r in chunk]}
        res = call(t, BULK_PATH, payload, "POST", timeout=600)
        return int(res.get("result", res).get("inserted", 0))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(chunks), workers):
            group = chunks[i:i + workers]
            landed = sum(pool.map(send, group))
            sent = sum(len(c) for c in group)
            if landed < sent:
                raise SystemExit(
                    f"\n  STOPPED: sent {sent} dependency rows, the server wrote "
                    f"{landed}.\n  Progress is at {done:,}.")
            done += landed
            state["relationships"] = done
            save_state(state)
            rate = (done - start_at) / max(0.1, time.time() - t0)
            print(f"    {done:>7,}/{len(rows):,}  {rate:>5.1f} rec/s", flush=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="Load the dataset into ServiceNow.")
    ap.add_argument("--only", choices=["cis", "relationships", "incidents", "changes",
                                       "problems", "knowledge"],
                    help="load one table and stop")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="stop after this many rows per table")
    ap.add_argument("--restart", action="store_true", help="ignore saved progress")
    args = ap.parse_args()

    if not DATASET.exists():
        sys.exit("dataset not built. Run generator/build.py first.")

    target = read_env()

    # Read the endpoint path here rather than at import time, so the error names the
    # thing the reader has to do rather than failing on a module constant.
    global BULK_PATH
    from env import read_env as _read_env_dict
    BULK_PATH = (os.environ.get("SERVICENOW_BULK_PATH")
                 or _read_env_dict().get("SERVICENOW_BULK_PATH", ""))
    if not BULK_PATH:
        sys.exit(
            "\n  SERVICENOW_BULK_PATH is not set.\n"
            "  It is the path of the Scripted REST API you created in Part 4 section 37,\n"
            "  and it contains YOUR instance's namespace, not mine. Open the Scripted\n"
            "  REST API record and copy the Base API path, then add it to .env.local:\n"
            "    SERVICENOW_BULK_PATH=/api/<your-namespace>/bulkload/insert")

    state = {} if args.restart else load_state()
    if args.restart and STATE.exists():
        STATE.unlink()

    print(f"  instance reachable: ", end="", flush=True)
    call(target, "/api/now/table/incident?sysparm_limit=1")
    print("yes\n")

    # ⛔ ORDER MATTERS. Configuration items first, because everything else points at
    # them. Then the dependency rows, then the records that reference a CI.
    order = ["configuration_items", "relationships", "incidents", "changes",
             "problems", "knowledge"]
    if args.only:
        order = ["configuration_items" if args.only == "cis" else args.only]

    sys_ids = {} if args.restart else load_sysids()
    for name in order:
        rows = read_rows(name)
        if args.limit:
            rows = rows[:args.limit]
        start_at = state.get(name, 0)
        if name == "configuration_items":
            start_at = ci_progress(rows, sys_ids, start_at)
            if start_at and start_at > state.get(name, 0):
                print(f"  {'cis':12s} progress file said {state.get(name, 0):,}, the "
                      f"sys_id map says {start_at:,}. Trusting the map.")
                state[name] = start_at
                save_state(state)
            load_cis(target, rows, start_at, args.batch, state, sys_ids)
        elif name == "knowledge":
            globals()["KB_BASE"] = resolve_knowledge_base(target)
            load_fast(target, name, rows, start_at, args.batch, args.workers,
                      state, sys_ids)
        elif name == "relationships":
            load_relationships(target, rows, start_at, args.batch, args.workers,
                               state, sys_ids)
        else:
            load_fast(target, name, rows, start_at, args.batch, args.workers,
                      state, sys_ids)

    print("\n  done. Progress file kept so a rerun is a no op.")


if __name__ == "__main__":
    main()
