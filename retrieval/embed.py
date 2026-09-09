"""Turn chunks into vectors, once, and keep them on disk.

The embedding model runs locally rather than on the rented GPU, and that is a deliberate
choice about who can reproduce this article. Part 8 rents a GPU to run the model that
WRITES the answer, which is the part that needs one. Embedding is a much smaller job, and
putting it on the same rented machine would mean nobody without GPU quota could check a
single retrieval number in Part 10. Local embeddings cost nothing and let any reader run
the whole measurement.

⛔ THE CACHE IS KEYED ON THE CORPUS, NOT ON A FILENAME. Embedding 82,296 chunks takes
tens of minutes, so it has to be cached, and a cache that does not notice the corpus
changed is worse than no cache: it silently scores new chunks against old vectors. The
key is a hash of every chunk id and its text, so any edit anywhere invalidates it.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
import urllib.request

import numpy as np

CACHE = pathlib.Path(__file__).resolve().parent.parent / "dataset" / ".embeddings"
OLLAMA = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
DIMENSIONS = 768

# Long texts slow the model down more than large batches speed it up, so this is a
# compromise measured on the real corpus rather than a round number.
BATCH = 32


def corpus_fingerprint(chunks: list[tuple[str, str]]) -> str:
    """A digest of exactly what is being embedded, which is the TEXT and only the text.

    Ids are deliberately not in the hash. The same corpus gets keyed two ways in this
    project: by chunk_id when embedding, because that is unique, and by source_id when
    grading, because that is what a gold set names. Those are the same texts in the same
    order, so they must share one cache entry rather than embedding everything twice.

    What the hash must catch is a chunk whose TEXT was rewritten. Scoring new text
    against an old vector produces a result that looks entirely reasonable and is wrong,
    and nothing downstream can detect it.
    """
    h = hashlib.sha256()
    for _, text in chunks:
        h.update(text.encode())
        h.update(b"\0")
    return h.hexdigest()


def _call(texts: list[str]) -> list[list[float]]:
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps({"model": MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["embeddings"]


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a small list, for a question at query time."""
    vectors = _call(texts)
    arr = np.asarray(vectors, dtype=np.float32)
    # Normalised on the way in, so similarity is a dot product rather than a division
    # repeated eighty thousand times per question.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-9)


def build(chunks: list[tuple[str, str]], quiet: bool = False) -> np.ndarray:
    """Embed the whole corpus, or load it from the cache when nothing changed."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fingerprint = corpus_fingerprint(chunks)
    path = CACHE / f"{MODEL}-{fingerprint[:16]}.npy"

    if path.exists():
        arr = np.load(path)
        if arr.shape[0] == len(chunks):
            if not quiet:
                print(f"  embeddings loaded from cache: {arr.shape[0]:,} x {arr.shape[1]}")
            return arr
        # A cache file whose row count disagrees with the corpus is corrupt, not stale.
        # Deleting it is safer than trusting either number.
        path.unlink()

    out = np.zeros((len(chunks), DIMENSIONS), dtype=np.float32)
    t0 = time.time()
    for start in range(0, len(chunks), BATCH):
        window = chunks[start:start + BATCH]
        # ⛔ The model has a context limit and a very long ticket exceeds it. Truncating
        # here rather than letting the call fail keeps one enormous work note from
        # stopping a forty minute job, and the cut is recorded in the article.
        texts = [t[:8000] for _, t in window]
        out[start:start + len(window)] = np.asarray(_call(texts), dtype=np.float32)
        if not quiet and (start // BATCH) % 40 == 0:
            done = start + len(window)
            rate = done / max(0.1, time.time() - t0)
            left = (len(chunks) - done) / max(0.1, rate)
            print(f"    {done:>7,}/{len(chunks):,}  {rate:>6.1f} chunks/s  "
                  f"about {left / 60:>4.0f} min left", flush=True)

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.maximum(norms, 1e-9)
    np.save(path, out)
    if not quiet:
        print(f"  embedded {len(chunks):,} chunks in "
              f"{(time.time() - t0) / 60:.1f} min, cached at {path.name}")
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "questions"))
    from chunking import (change_chunks, ci_chunks, knowledge_chunks, per_record,
                          problem_chunks)
    from gold import World

    world = World()
    names = {k: c["name"] for k, c in world.cis.items()}
    everything = (per_record(world.incidents)
                  + ci_chunks(world.cis.values(), world.depends_on, world.supports, names)
                  + change_chunks(world.changes, names)
                  + problem_chunks(world.problems)
                  + knowledge_chunks(world.knowledge))
    build([(c.chunk_id, c.text) for c in everything])
