"""Damage the graph by degrees and re-run the arms, to find where the graph stops winning.

Section 114b promises this and the article did not have it. Section 114 measures what
damage does to the SHAPE of a blast radius answer, which is a property of the estate.
This measures something else: at what point does damaging the graph take away the two
cells the graph arms win? That is the number a shop with a known-stale CMDB actually
needs, because it tells them whether their data is good enough to bother.

⛔ THIS MUTATES THE LIVE GRAPH AND PUTS IT BACK. Every deleted edge is held in memory with
its properties and re-created after the level is measured, and the edge count is checked
against the baseline before the next level starts. If a restore ever comes up short the
run stops rather than carrying a damaged graph into the next measurement, because a sweep
whose later rows sit on the earlier rows' damage measures nothing.

⛔ AND IT REFUSES TO START ON A GRAPH THAT IS NOT THE PUBLISHED ONE. If the baseline edge
count is not what Part 7 loads, the numbers would not be comparable with the rest of the
article.

⛔ EDGES ARE REMOVED, NOT REWIRED, for the reason degraded.py gives: a stale CMDB fails by
omission far more often than by pointing somewhere wrong.

Run: python3 retrieval/damage_sweep.py

Author: Roni Das
Created: 2026-09-10
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "generator"))
sys.path.insert(0, str(HERE.parent / "questions"))

from neo4j import GraphDatabase  # noqa: E402

from arms import BM25Only, GraphOnly, HybridCypher, VectorCypher  # noqa: E402
from env import env_path  # noqa: E402
from evaluate import score_one  # noqa: E402
from gold import World, build as build_gold  # noqa: E402
from questions import QUESTIONS  # noqa: E402
from run import build_corpus  # noqa: E402

FRACTIONS = [0.0, 0.10, 0.20, 0.40, 0.60]
SEEDS = [20260910, 20260911, 20260912]
BASELINE_EDGES = 28694
OUT = HERE.parent / "results" / "captures" / "damage-sweep.json"


def env() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in env_path().read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def edge_count(session) -> int:
    with session() as s:
        return s.run("MATCH ()-[r:SUPPORTS]->() RETURN count(r) AS n").single()["n"]


def all_edges(session) -> list[dict]:
    with session() as s:
        return [dict(r) for r in s.run(
            "MATCH (a:ConfigurationItem)-[r:SUPPORTS]->(b:ConfigurationItem) "
            "RETURN a.key AS parent, b.key AS child, r.type_name AS type_name, "
            "r.carries_impact AS carries_impact, "
            "toString(r.last_discovered) AS last_discovered")]


def drop(session, rows: list[dict]) -> None:
    with session() as s:
        for i in range(0, len(rows), 1000):
            s.run("""
                UNWIND $rows AS row
                MATCH (a:ConfigurationItem {key: row.parent})
                      -[r:SUPPORTS {type_name: row.type_name}]->
                      (b:ConfigurationItem {key: row.child})
                DELETE r
            """, rows=rows[i:i + 1000])


def restore(session, rows: list[dict]) -> None:
    with session() as s:
        for i in range(0, len(rows), 1000):
            s.run("""
                UNWIND $rows AS row
                MATCH (a:ConfigurationItem {key: row.parent})
                MATCH (b:ConfigurationItem {key: row.child})
                MERGE (a)-[r:SUPPORTS {type_name: row.type_name}]->(b)
                  SET r.carries_impact = row.carries_impact,
                      r.last_discovered = CASE WHEN row.last_discovered IS NULL
                                          THEN NULL
                                          ELSE datetime(row.last_discovered) END
            """, rows=rows[i:i + 1000])


def main() -> int:
    e = env()
    driver = GraphDatabase.driver(e["NEO4J_URI"],
                                  auth=(e["NEO4J_USERNAME"], e["NEO4J_PASSWORD"]))
    session = driver.session

    base = edge_count(session)
    if base != BASELINE_EDGES:
        print(f"  ⛔ the graph holds {base:,} SUPPORTS edges and the published graph holds "
              f"{BASELINE_EDGES:,}. Reload with generator/load_neo4j.py --wipe before "
              "measuring, or these numbers cannot be compared with the article's.")
        driver.close()
        return 2
    print(f"  baseline: {base:,} SUPPORTS edges\n")

    world = World()
    gold = build_gold(world)
    chunks = build_corpus(world)
    names = {c["name"]: k for k, c in world.cis.items() if c.get("name")}
    pairs = [(c.chunk_id, c.text) for c in chunks]

    # ⛔ THE QUESTIONS ARE THE ONES THE GRAPH ARMS ACTUALLY SCORE ON. Sweeping damage
    # against questions no graph arm can answer would measure nothing moving.
    ids = [q.id for q in QUESTIONS
           if q.id in gold and gold[q.id].supporting and q.kind in {"multi_hop",
                                                                    "aggregation"}]
    asked = [q for q in QUESTIONS if q.id in ids]
    print(f"  {len(asked)} questions, the multi hop and aggregation ones the graph wins\n")

    bm = BM25Only(pairs)
    impact_rows = [r for r in all_edges(session) if r["carries_impact"]]
    print(f"  {len(impact_rows):,} of them carry impact, and those are the ones damaged\n")

    results: dict[str, dict] = {}
    print(f"  {'damage':>7} {'keyword':>9} {'bare walk':>11} {'sim+walk':>10} "
          f"{'both+walk':>11}")
    for frac in FRACTIONS:
        per_seed: dict[str, list[float]] = {"bm25": [], "graph_only": [],
                                            "vector_cypher": [], "hybrid_cypher": []}
        for seed in SEEDS:
            rng = random.Random(seed)
            victims = (rng.sample(impact_rows, int(len(impact_rows) * frac))
                       if frac else [])
            if victims:
                drop(session, victims)
                left = edge_count(session)
                if left != base - len(victims):
                    print(f"\n  ⛔ expected {base - len(victims):,} edges after the "
                          f"delete and found {left:,}. Stopping with the graph damaged; "
                          "reload it with generator/load_neo4j.py --wipe.")
                    driver.close()
                    return 2
            try:
                arms = {"bm25": bm, "graph_only": GraphOnly(session, names),
                        "vector_cypher": VectorCypher(session)}
                arms["hybrid_cypher"] = HybridCypher(arms["vector_cypher"], session)
                for name, arm in arms.items():
                    got = []
                    for q in asked:
                        r = arm.retrieve(q.text, q.id)
                        s = score_one(r, q, gold[q.id])
                        if s.recall_at_k is not None:
                            got.append(s.recall_at_k)
                    per_seed[name].append(sum(got) / len(got) if got else 0.0)
            finally:
                if victims:
                    restore(session, victims)
                    back = edge_count(session)
                    if back != base:
                        print(f"\n  ⛔ restore left {back:,} edges, not {base:,}. "
                              "Reload with generator/load_neo4j.py --wipe.")
                        driver.close()
                        return 2
            if not frac:
                break        # undamaged is the same every seed
        row = {k: sum(v) / len(v) for k, v in per_seed.items()}
        results[f"{frac:.2f}"] = row
        print(f"  {frac * 100:>6.0f}% {row['bm25']:>9.2f} {row['graph_only']:>11.2f} "
              f"{row['vector_cypher']:>10.2f} {row['hybrid_cypher']:>11.2f}")

    # ⛔ THE CROSSING POINT IS COMPUTED, NOT EYEBALLED. It is the first damage level at
    # which no graph arm still beats keyword search on these questions.
    crossing = None
    for frac in FRACTIONS:
        row = results[f"{frac:.2f}"]
        best_graph = max(row["graph_only"], row["vector_cypher"], row["hybrid_cypher"])
        if best_graph <= row["bm25"]:
            crossing = frac
            break

    print()
    if crossing is None:
        print(f"  No crossing point inside this range. The best graph arm still beats "
              f"keyword search at {FRACTIONS[-1] * 100:.0f}% of impact edges missing.")
    elif crossing == 0.0:
        print("  The graph arms do not beat keyword search on these questions even "
              "undamaged, so there is no crossing point to find.")
    else:
        print(f"  Crossing point: {crossing * 100:.0f}% of impact-carrying edges missing. "
              "Past that, no graph arm still beats keyword search on these questions.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "what": "recall against how much of the dependency graph is missing",
        "source": f"retrieval/damage_sweep.py, {time.strftime('%Y-%m-%d')}",
        "questions": ids,
        "impact_edges": len(impact_rows),
        "seeds": SEEDS,
        "fractions": FRACTIONS,
        "recall": results,
        "crossing_point": crossing,
    }, indent=2) + "\n")
    print(f"\n  wrote {OUT}")
    print(f"  graph restored to {edge_count(session):,} edges")
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
