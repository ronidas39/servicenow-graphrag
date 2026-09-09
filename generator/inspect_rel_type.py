"""Print every column ServiceNow defines on `cmdb_rel_type`, from a live instance.

Part 6 section 55 rests a decision on this table: it says there is no column telling you
which relationship types propagate impact, so you have to decide that yourself and record
it. That sentence was in the article for a while with nothing behind it but my memory of
having looked. An audit lens asked where the evidence was and it was right to.

So this asks the instance. It reads `sys_dictionary`, which is where ServiceNow keeps the
definition of every column on every table, and prints what comes back.

⛔ THIS IS NOT A CLAIM THAT SERVICENOW CANNOT COMPUTE IMPACT. It can, and Part 0 section 2b
says so. Impact rules live in their own tables and in the Impact Analysis API, not as a
flag on the relationship type. The narrow claim, and the only one this script supports, is
that reading `cmdb_rel_type` will not tell you which types to traverse.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import pathlib
import sys

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from env import read_env  # noqa: E402

TABLE = "cmdb_rel_type"
TIMEOUT = 90


def main() -> int:
    env = read_env()
    base = env["SERVICENOW_INSTANCE"].rstrip("/")
    auth = (env["SERVICENOW_USER"], env["SERVICENOW_PASSWORD"])

    response = requests.get(
        f"{base}/api/now/table/sys_dictionary",
        params={
            "sysparm_query": f"name={TABLE}^ORDERBYelement",
            "sysparm_fields": "element,internal_type,column_label",
            "sysparm_limit": "100",
        },
        auth=auth,
        headers={"Accept": "application/json"},
        timeout=TIMEOUT,
    )
    if response.status_code != 200:
        print(f"  ServiceNow returned {response.status_code} for {TABLE}")
        return 1

    # The dictionary carries one row with an empty `element` describing the table itself.
    # It is not a column, and counting it would make the table look one wider than it is.
    columns = [r for r in response.json().get("result", []) if r.get("element")]

    print(f"  columns defined on {TABLE}: {len(columns)}\n")
    for row in columns:
        kind = row.get("internal_type")
        kind = kind.get("value") if isinstance(kind, dict) else kind
        print(f"    {row['element']:26s} {str(kind):18s} {row.get('column_label')}")

    # ⛔ THE CHECK IS THE POINT, NOT THE LISTING. A reader running this a year from now on
    # a newer release needs to be told if the answer has changed, and a printed table does
    # not do that on its own.
    impactish = [r["element"] for r in columns
                 if any(word in r["element"].lower() for word in ("impact", "propagat"))]
    print()
    if impactish:
        print(f"  ⛔ this instance DOES define {impactish}, so section 55 is out of date "
              f"for this release.")
        return 1
    print("  No column here names impact or propagation. The decision about which types "
          "propagate\n  is yours to make and yours to record, which is what section 55 "
          "does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
