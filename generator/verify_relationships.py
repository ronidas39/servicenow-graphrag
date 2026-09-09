"""Prove the dependency rows in ServiceNow point the way their type names say.

The direction defect was found in the generated files and fixed there, then the instance
was repaired and reloaded. None of that is evidence about the INSTANCE. This asks the
instance itself, because the graph in Part 7 is built by reading ServiceNow, so ServiceNow
is what has to be right.

Three checks, and the second is the one that matters:

  every type resolves            a type name that does not exist means rows were skipped
  a shared cluster is depended
  ON, not depending              the symptom the whole defect produced
  correlation_id is a phantom
  on this table                  proven, so nobody writes a check against it again

⛔ THIS ASKS THE SERVER, NOT THE FILES. Checking the files again would only re-prove what
the generator tests already prove.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import pathlib
import sys
import urllib.parse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from load_servicenow import Target, call, load_sysids, read_env  # noqa: E402

IMPACT = {"Depends on::Used by", "Runs on::Runs", "Hosted on::Hosts"}


def count(t: Target, table: str, query: str = "") -> int:
    q = urllib.parse.urlencode({"sysparm_query": query, "sysparm_count": "true"})
    return int(call(t, f"/api/now/stats/{table}?{q}")["result"]["stats"]["count"])


def type_ids(t: Target) -> dict[str, str]:
    out = {}
    for name in sorted(IMPACT):
        q = urllib.parse.urlencode({"sysparm_query": f"name={name}",
                                    "sysparm_fields": "sys_id", "sysparm_limit": 1})
        found = call(t, f"/api/now/table/cmdb_rel_type?{q}").get("result", [])
        if found:
            out[name] = found[0]["sys_id"]
    return out


def main() -> int:
    t = read_env()
    ours = load_sysids()
    problems: list[str] = []

    total = count(t, "cmdb_rel_ci")
    print(f"  cmdb_rel_ci rows            : {total:,}")

    # ⛔ THE FIRST VERSION OF THIS CHECK PRINTED A NUMBER THAT MEANT NOTHING. It counted
    # rows matching correlation_idISNOTEMPTY and reported 40,709 of 40,709 as "written by
    # this project", which was flatly wrong: 12,015 of those belong to the instance's own
    # demo CMDB. cmdb_rel_ci HAS NO correlation_id COLUMN, and ServiceNow answers a query
    # on a column that does not exist by ignoring the filter and returning everything.
    #
    # So the check is now the check: send a value nothing could hold. A real field matches
    # none of it. A phantom field matches every row in the table.
    impossible = count(t, "cmdb_rel_ci",
                       "correlation_id=zzz-this-value-cannot-exist-zzz")
    if impossible:
        print(f"    ⛔ a filter on correlation_id matched {impossible:,} rows, so that "
              f"column does not exist here")
        print(f"       and any check written against it is measuring nothing")

    types = type_ids(t)
    missing = [n for n in IMPACT if n not in types]
    if missing:
        problems.append(f"these impact types do not exist on the instance: {missing}")
    print(f"  impact types resolved       : {len(types)} of {len(IMPACT)}")

    # ⛔ THE CHECK THAT WOULD HAVE CAUGHT THE ORIGINAL DEFECT. A shared cluster is
    # depended ON by hundreds of hosts and depends on almost nothing. When the edges were
    # backwards it had 950 dependencies and zero dependents, and every blast radius
    # answer starting below it came back empty while looking perfectly healthy.
    cluster_sys_id = ours.get("cluster-us-east-01")
    if not cluster_sys_id:
        problems.append("cluster-us-east-01 is not in the sys_id map")
    else:
        hosted = types.get("Hosted on::Hosts", "")
        as_parent = count(t, "cmdb_rel_ci",
                          f"parent={cluster_sys_id}^type={hosted}")
        as_child = count(t, "cmdb_rel_ci",
                         f"child={cluster_sys_id}^type={hosted}")
        print(f"\n  cluster-us-east-01, 'Hosted on::Hosts'")
        print(f"    as PARENT (it is hosted on something) : {as_parent:,}")
        print(f"    as CHILD  (things are hosted on it)   : {as_child:,}")
        if as_child < 100:
            problems.append(
                f"the shared cluster has only {as_child} things hosted on it. A cluster "
                f"carrying hundreds of servers should be the CHILD of that many rows, "
                f"because the parent is the thing that is 'Hosted on' the other.")
        if as_parent > as_child:
            problems.append(
                f"the cluster is the parent of more rows ({as_parent:,}) than it is the "
                f"child of ({as_child:,}). That is the direction defect: it reads as the "
                f"cluster being hosted on its own servers.")

    print()
    if problems:
        print(f"  ⛔ {len(problems)} problem(s):")
        for line in problems:
            print(f"    {line}")
        return 1
    print("  the dependency rows on the instance point the way their names say")
    return 0


if __name__ == "__main__":
    sys.exit(main())
