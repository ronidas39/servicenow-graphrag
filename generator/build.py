"""Build the whole dataset and write it to disk.

One command produces everything the article needs. The output is what readers download,
so it is written as plain JSON Lines: readable in any language, diffable, and streamable
without loading the whole file.

The manifest is the part that matters most. It records every rate this generator was
asked for and every rate it actually produced, because the article makes claims about
this data and those claims have to be checkable against the file rather than against my
description of it.

Author: Roni Das
Created: 2026-09-08
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
from datetime import datetime, timezone

from estate import build_estate
from incidents import generate as gen_incidents
from records import generate_changes, generate_knowledge, generate_problems

OUT = pathlib.Path(__file__).resolve().parent.parent / "dataset"


def _encode(obj: object) -> object:
    if isinstance(obj, datetime):
        # Always UTC, always explicit. The article spends a whole section on why a
        # timestamp without a zone is the most expensive kind of missing information.
        return obj.astimezone(timezone.utc).isoformat()
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def write_jsonl(rows: list, path: pathlib.Path) -> tuple[int, str]:
    """Write rows and return the count and a checksum of the file."""
    h = hashlib.sha256()
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(dataclasses.asdict(row), default=_encode,
                              ensure_ascii=False, sort_keys=True)
            f.write(line + "\n")
            h.update(line.encode("utf-8"))
    return len(rows), h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the ServiceNow GraphRAG dataset.")
    ap.add_argument("--services", type=int, default=2200)
    ap.add_argument("--incidents", type=int, default=60_000)
    ap.add_argument("--changes", type=int, default=8_000)
    ap.add_argument("--problems", type=int, default=900)
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    estate = build_estate(services=args.services, seed=args.seed)
    incidents = gen_incidents(estate, count=args.incidents, seed=args.seed)
    changes = generate_changes(estate, incidents, count=args.changes, seed=args.seed)
    problems = generate_problems(incidents, count=args.problems, seed=args.seed)
    knowledge = generate_knowledge(problems, seed=args.seed)

    files: dict[str, dict] = {}
    for name, rows in [
        ("configuration_items", estate.cis),
        ("relationships", estate.rels),
        ("incidents", incidents),
        ("changes", changes),
        ("problems", problems),
        ("knowledge", knowledge),
    ]:
        n, digest = write_jsonl(rows, OUT / f"{name}.jsonl")
        size = (OUT / f"{name}.jsonl").stat().st_size
        files[name] = {"rows": n, "bytes": size, "sha256_16": digest}
        print(f"  {name:22s} {n:>8,} rows  {size / 1e6:>7.1f} MB  {digest}")

    # Every rate the article will quote, measured from what was actually written.
    n = len(incidents)
    measured = {
        "incidents_repeating_an_earlier_ticket":
            round(sum(1 for i in incidents if i.duplicate_of) / n, 4),
        "incidents_naming_another_ticket":
            round(sum(1 for i in incidents if "related to INC" in i.description) / n, 4),
        "incidents_with_pasted_output":
            round(sum(1 for i in incidents if "```" in i.description) / n, 4),
        "incidents_with_no_configuration_item":
            round(sum(1 for i in incidents if not i.ci_key) / n, 4),
        "incidents_naming_a_one_hop_neighbour": None,   # filled below, needs the graph
        "changes_raised_after_their_incident":
            round(sum(1 for c in changes if c.raised_after_incident) / max(1, len(changes)), 4),
        "dependency_edges_over_a_year_old": round(sum(
            1 for r in estate.rels if r.last_discovered
            and (datetime.now(timezone.utc) - r.last_discovered).days > 365
        ) / max(1, len(estate.rels)), 4),
    }

    # The share of tickets whose notes name something one hop away. This is the number
    # that decides whether a text baseline has any chance at a multi hop question, so it
    # is published rather than described.
    adjacency: dict[str, set[str]] = {}
    for rel in estate.rels:
        adjacency.setdefault(rel.parent_key, set()).add(rel.child_key)
        adjacency.setdefault(rel.child_key, set()).add(rel.parent_key)
    names = {c.key: c.name for c in estate.cis}
    linked = [i for i in incidents if i.ci_key]
    crossing = sum(
        1 for i in linked
        if any(names.get(nb, "\0") in " ".join(n[2] for n in i.work_notes)
               for nb in adjacency.get(i.ci_key, ()))
    )
    measured["incidents_naming_a_one_hop_neighbour"] = round(crossing / max(1, len(linked)), 4)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "note": (
            "The instance, the tables, the field behaviour and the API this loads into "
            "are all real. The estate and the words inside the tickets are written by "
            "this generator. Real incident text is what makes ITSM data confidential, "
            "which is why no company publishes it, and why the article self hosts a "
            "model rather than sending the text to a vendor. "
            "Configuration items are named the way a real estate names them: a service "
            "carries a business name because people talk about it, a host carries an "
            "infrastructure name because nobody does. No token runs down a stack, so "
            "finding a service by name does not find its hosts. That is deliberate. An "
            "earlier version shared one token across every item in a stack and a plain "
            "substring search recovered the whole stack at 78 percent recall, which "
            "would have decided this article's comparison before any retriever ran."
        ),
        "files": files,
        "totals": {
            "nodes": sum(files[k]["rows"] for k in
                         ("configuration_items", "incidents", "changes",
                          "problems", "knowledge")),
            "dependency_edges": files["relationships"]["rows"],
            "work_notes": sum(len(i.work_notes) for i in incidents),
        },
        "measured_rates": measured,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n  manifest.json written. Rates measured from the files, not asserted:")
    for k, v in measured.items():
        print(f"    {k:44s} {v:.1%}")
    print(f"\n  nodes for the graph      : {manifest['totals']['nodes']:,}")
    print(f"  dependency edges         : {manifest['totals']['dependency_edges']:,}")
    print(f"  work notes               : {manifest['totals']['work_notes']:,}")


if __name__ == "__main__":
    main()
