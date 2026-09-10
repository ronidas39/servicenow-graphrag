"""Put the chunks and their vectors into the graph, joined to the records they came from.

Part 9 section 98. Everything before this kept the text and the graph apart, because the
first four retrievers did not need them together. The three graph retrievers do: a chunk
with no edge back to its record is an island, and a traversal has nowhere to start.

⛔ FIVE QUERIES, NOT ONE, AND GETTING IT WRONG IS SILENT. The label and the key differ per
record kind, and Cypher will not take a label from a parameter without APOC. Match on
`:Incident` alone and the other four kinds find nothing: `MERGE` never runs and the rows
are skipped with no error. On this corpus that is 22,296 of 82,296 chunks gone, and every
configuration item is among them, which is exactly what a graph retriever needs most.
That is why this finishes by counting.

⛔ THE GRAPH AND THE CORPUS HAVE TO BE THE SAME ESTATE, and for a while they were not.
The graph had been loaded by reading a real ServiceNow developer instance, and that
instance holds its own demo CMDB: `INC0011485`, `Starry Night Pro`. The corpus comes from
`dataset/*.jsonl`: `INC2000000`, `cluster-us-east-01`. Measured, 0 of 60,000 incidents and
21 of 11,891 items were shared. Every MERGE here would have matched nothing and reported
success. Part 10 scores the file corpus, so the graph these chunks attach to is built from
the same files by `load_neo4j.py`.

Author: Roni Das
Created: 2026-09-10
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
from neo4j import GraphDatabase

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "retrieval"))
sys.path.insert(0, str(HERE.parent / "questions"))

from chunking import (change_chunks, ci_chunks, knowledge_chunks,  # noqa: E402
                      per_record, problem_chunks)
from embed import MODEL, build as build_vectors  # noqa: E402
from env import env_path  # noqa: E402
from gold import World  # noqa: E402

# The five kinds, each with the label and the key its chunks point at. This table IS
# section 98's table, and it exists in one place so the two cannot drift.
KINDS = [
    ("incidents", "Incident", "number"),
    ("configuration_items", "ConfigurationItem", "key"),
    ("changes", "Change", "number"),
    ("problems", "Problem", "number"),
    ("knowledge", "KnowledgeArticle", "number"),
]

INDEX_NAME = "chunk_embedding"


def read_env() -> dict[str, str]:
    import re
    env: dict[str, str] = {}
    for line in env_path().read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return env


def corpus(world: World) -> dict[str, list]:
    names = {k: c["name"] for k, c in world.cis.items()}
    return {
        "incidents": per_record(world.incidents),
        "configuration_items": ci_chunks(world.cis.values(), world.depends_on,
                                         world.supports, names),
        "changes": change_chunks(world.changes, names),
        "problems": problem_chunks(world.problems),
        "knowledge": knowledge_chunks(world.knowledge),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--wipe", action="store_true",
                    help="delete existing :Chunk nodes first")
    args = ap.parse_args()

    env = read_env()
    world = World()
    groups = corpus(world)
    ordered = [c for kind, *_ in KINDS for c in groups[kind]]
    total = len(ordered)
    print(f"  corpus: {total:,} chunks across {len(KINDS)} kinds")

    # ⛔ THE VECTORS COME FROM THE SAME CACHE PART 10 SCORES, keyed on a hash of the text.
    # Embedding them again here would risk a second array under a second fingerprint, and
    # then the graph retrievers would be searching a different space from the other four.
    vectors = build_vectors([(c.chunk_id, c.text) for c in ordered], quiet=False)
    if vectors.shape[0] != total:
        print(f"  ⛔ {vectors.shape[0]:,} vectors for {total:,} chunks")
        return 2
    width = int(vectors.shape[1])
    print(f"  vectors: {vectors.shape[0]:,} x {width} from the {MODEL} cache")

    driver = GraphDatabase.driver(env["NEO4J_URI"],
                                  auth=(env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"]))
    with driver.session() as s:
        if args.wipe:
            n = s.run("MATCH (c:Chunk) DETACH DELETE c "
                      "RETURN count(*) AS n").single()["n"]
            print(f"  wiped {n:,} existing chunks")

        s.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS "
              "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")

        at = 0
        for kind, label, key in KINDS:
            chunks = groups[kind]
            cypher = f"""
            UNWIND $rows AS row
            MATCH (r:{label} {{{key}: row.source_id}})
            MERGE (c:Chunk {{chunk_id: row.chunk_id}})
              SET c.text = row.text, c.embedding = row.embedding, c.kind = $kind
            MERGE (c)-[:CHUNK_OF]->(r)
            """
            t0, done = time.time(), 0
            for i in range(0, len(chunks), args.batch):
                window = chunks[i:i + args.batch]
                rows = [{"chunk_id": c.chunk_id, "source_id": c.source_id,
                         "text": c.text,
                         "embedding": vectors[at + i + j].astype(float).tolist()}
                        for j, c in enumerate(window)]
                s.run(cypher, rows=rows, kind=kind)
                done += len(window)
                print(f"    {kind:22s} {done:>7,}/{len(chunks):,}", end="\r", flush=True)
            at += len(chunks)
            print(f"    {kind:22s} {done:>7,} in {time.time() - t0:.0f}s")

        # ⛔ THE INDEX IS CREATED AFTER THE WRITE, and its width comes from the array
        # rather than from a constant. A vector index declared at the wrong width accepts
        # the create and then refuses every node, quietly, one at a time.
        s.run(f"""
            CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{ indexConfig: {{
              `vector.dimensions`: {width},
              `vector.similarity_function`: 'cosine' }} }}
        """)
        print(f"  vector index {INDEX_NAME} on :Chunk(embedding), {width} dims, cosine")

        # ⛔ THE COUNT IS THE CHECK, and it is the one section 98 names. Anything short
        # means a label or a key was wrong and the MERGE matched nothing.
        got = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
        joined = s.run("MATCH (:Chunk)-[:CHUNK_OF]->() RETURN count(*) AS n").single()["n"]
        print(f"\n  what landed:")
        print(f"    Chunk               {got:>8,}  of {total:,}")
        print(f"    CHUNK_OF            {joined:>8,}")
        for kind, label, _ in KINDS:
            n = s.run("MATCH (c:Chunk {kind: $k})-[:CHUNK_OF]->(:" + label + ") "
                      "RETURN count(c) AS n", k=kind).single()["n"]
            print(f"      {kind:20s} {n:>8,}  of {len(groups[kind]):,}")
        if got != total or joined != total:
            print(f"\n  ⛔ SHORT. {total - got:,} chunks never landed and {total - joined:,} "
                  "have no edge back to a record. A label or a key is wrong, and the "
                  "MERGE matched nothing without raising.")
            driver.close()
            return 1

    driver.close()
    print("\n  every chunk is in the graph and joined to its record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
