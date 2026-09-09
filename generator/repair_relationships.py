"""Remove the dependency rows that were written pointing the wrong way, so they can be
written again pointing the right way.

The estate generator wrote the container as the parent for every containment type, but a
ServiceNow relationship type is named "what the parent is to the child::what the child is
to the parent". So "Hosted on::Hosts" with the cluster as parent says the cluster is
hosted on the server, which is the sentence backwards. Four of the eight types in this
estate were written that way, which is 16,032 of 28,694 edges.

The instance cannot simply be loaded again. The insert path writes cmdb_rel_ci with no
correlation_id, so nothing on the row identifies which dataset row produced it, and a
second run would add a corrected copy beside the backwards one rather than replacing it.

⛔ AND IT CANNOT BE FIXED THE OBVIOUS WAY. Adding a correlation_id to the dependency rows
was tried. cmdb_rel_ci HAS NO SUCH COLUMN, the insert accepted the field and silently
discarded it, and a query filtering on it returned all 40,709 rows because ServiceNow
ignores a filter on a column that does not exist. The idempotency key for a relationship
has to be the relationship itself: parent, type and child are real columns and they
discriminate. See load_servicenow.py, load_relationships.

⛔ SCOPED BY OUR OWN SYS_IDS. A developer instance ships with its own demo CMDB, and
deleting by relationship type alone would take those rows out too. Only rows whose parent
is a configuration item this project loaded are touched.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from load_servicenow import (  # noqa: E402
    DATASET,
    STATE,
    Target,
    call,
    load_sysids,
    read_env,
)

# Four of these were written with parent and child the wrong way round: the containment
# types, where the generator put the container first. "Owns::Owned by", "Depends on::Used
# by", "Runs on::Runs" and "IP Connection" were already correct, because the owner really
# is the subject of "Owns" and a service really does depend on its application.
#
# All eight are removed anyway. Nothing on a cmdb_rel_ci row identifies which dataset row
# produced it, and nothing can, so there is no way to delete only the four broken types
# and be sure the remaining rows are the correct ones rather than survivors of an earlier
# run. Clearing every edge this project loaded and writing them all once is the only state
# that can be reasoned about afterwards. It costs a few thousand extra deletes.
INVERTED = [
    "Hosted on::Hosts",
    "In Rack::Rack contains",
    "Located in Zone::Zone contains",
    "Managed by::Manages",
    "Owns::Owned by",
    "Depends on::Used by",
    "Runs on::Runs",
    "IP Connection::IP Connection",
]

PAGE = 1000


def resolve_types(t: Target, names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        q = urllib.parse.urlencode({"sysparm_query": f"name={name}",
                                    "sysparm_fields": "sys_id,name",
                                    "sysparm_limit": 1})
        found = call(t, f"/api/now/table/cmdb_rel_type?{q}").get("result", [])
        if found:
            out[name] = found[0]["sys_id"]
    missing = [n for n in names if n not in out]
    if missing:
        raise SystemExit(f"  these relationship types are not on the instance: {missing}")
    return out


def find_rows(t: Target, type_ids: dict[str, str], ours: set[str]) -> list[str]:
    """Every cmdb_rel_ci row of an inverted type whose parent is one of our items."""
    wanted = set(type_ids.values())
    sys_ids: list[str] = []
    offset = 0
    scanned = 0
    while True:
        q = urllib.parse.urlencode({
            "sysparm_query": "typeIN" + ",".join(sorted(wanted)),
            "sysparm_fields": "sys_id,parent,child,type",
            "sysparm_limit": PAGE,
            "sysparm_offset": offset,
        })
        page = call(t, f"/api/now/table/cmdb_rel_ci?{q}").get("result", [])
        if not page:
            break
        scanned += len(page)
        for row in page:
            parent = (row.get("parent") or {})
            pid = parent.get("value") if isinstance(parent, dict) else parent
            if pid in ours:
                sys_ids.append(row["sys_id"])
        offset += PAGE
        print(f"    scanned {scanned:,}, ours so far {len(sys_ids):,}", flush=True)
    return sys_ids


def delete_rows(t: Target, sys_ids: list[str], workers: int) -> int:
    done = 0
    t0 = time.time()

    def kill(sid: str) -> int:
        try:
            call(t, f"/api/now/table/cmdb_rel_ci/{sid}", method="DELETE")
            return 1
        except Exception:
            return 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(sys_ids), workers * 20):
            batch = sys_ids[i:i + workers * 20]
            done += sum(pool.map(kill, batch))
            rate = done / max(0.1, time.time() - t0)
            print(f"    deleted {done:>7,}/{len(sys_ids):,}  {rate:>5.1f} rec/s",
                  flush=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true",
                    help="count what would be deleted and stop")
    args = ap.parse_args()

    target = read_env()
    ours = set(load_sysids().values())
    print(f"  configuration items this project loaded : {len(ours):,}")

    type_ids = resolve_types(target, INVERTED)
    print(f"  inverted relationship types resolved    : {len(type_ids)}")

    print("\n  finding the rows that point the wrong way:")
    doomed = find_rows(target, type_ids, ours)
    print(f"\n  {len(doomed):,} rows to remove")

    if args.dry_run:
        print("  dry run, nothing deleted")
        return

    if not doomed:
        print("  nothing to do")
        return

    removed = delete_rows(target, doomed, args.workers)
    print(f"\n  removed {removed:,} of {len(doomed):,}")

    # Reset the phase so the loader writes all of them again, this time with a
    # correlation_id on every row.
    if STATE.exists():
        state = json.loads(STATE.read_text())
        state.pop("relationships", None)
        STATE.write_text(json.dumps(state, indent=2) + "\n")
        print("  relationships progress cleared. Re-run:")
        print("    python3 generator/load_servicenow.py --only relationships")


if __name__ == "__main__":
    main()
