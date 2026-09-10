"""Turn chunks into vectors on the GPU server Part 8 built, and keep them on disk.

The embedding model is served by vLLM, on the same rented card as the model that writes
the answer, and it is reached through the OpenAI-compatible `/v1/embeddings` route. That
matters for the article's whole argument: Part 3 says the ticket text never leaves the
company, and an embedding call sends the ticket text. Sending it to a hosted embedding
API would break that promise just as thoroughly as sending it to a hosted chat API, and
it is the easier mistake to make because embeddings feel like plumbing rather than like
AI.

⛔ THIS USED TO CALL OLLAMA ON THE LAPTOP, AND THAT WAS WRONG TWICE. Once because the
article stands up a perfectly good embedding server in Part 8 and then did not use it,
so Part 8 read as a detour. And once because a laptop is not where an estate's ticket
text gets embedded when the corpus is 82,296 chunks. The old docstring defended the
laptop on reproducibility grounds, which was a real concern answered the wrong way: the
answer is that `/v1/embeddings` is a standard route, so any server that speaks it works
here, including one running on a reader's own machine.

⛔ THE CACHE IS KEYED ON THE CORPUS AND ON THE MODEL, NOT ON A FILENAME. Embedding 82,296
chunks takes real time and money, so it has to be cached, and a cache that does not
notice the corpus changed is worse than no cache: it silently scores new chunks against
old vectors. The key is a hash of every chunk's text, so any edit anywhere invalidates
it, and the model name is in the filename so two models never share an entry.

Author: Roni Das
Created: 2026-09-09
Modified: Roni Das, 2026-09-10
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

CACHE = pathlib.Path(__file__).resolve().parent.parent / "dataset" / ".embeddings"

# Where the vLLM embedding server is. Defaults to the loopback address, which is right
# when this runs on the GPU box itself; set it to the server's address to drive it from
# a laptop, which is what section 86 does.
BASE_URL = os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
ENDPOINT = f"{BASE_URL}/v1/embeddings"

# The name 03-serve.sh passes to --served-model-name, so the client does not have to
# repeat the Hugging Face path. MODEL is what the cache filename is built from, so it
# carries the real model name instead.
SERVED_AS = os.environ.get("EMBED_SERVED_AS", "embed")
MODEL = os.environ.get("EMBED_MODEL", "Qwen3-Embedding-0.6B")

# ⛔ BATCH AND CONCURRENCY ARE DIFFERENT KNOBS AND BOTH MATTER. A batch is how many texts
# ride in one HTTP request, and it stops being useful once the server is saturated.
# Concurrency is how many requests are in flight, and it is what keeps a GPU busy across
# the network round trip. Measured on an L4 over the public internet, batch alone left
# the card idle most of the time.
BATCH = int(os.environ.get("EMBED_BATCH", "32"))
CONCURRENCY = int(os.environ.get("EMBED_CONCURRENCY", "8"))

# The model's context. A very long work note is cut rather than allowed to fail the whole
# request, and the cut is recorded in the article rather than hidden.
MAX_CHARS = int(os.environ.get("EMBED_MAX_CHARS", "8000"))

# ⛔ THE QUERY GETS A PREFIX AND THE DOCUMENT DOES NOT, AND THAT ASYMMETRY IS THE MODEL'S
# DOCUMENTED USE RATHER THAN A TRICK. Qwen3-Embedding is trained so that a query is
# introduced by an instruction and a passage is embedded bare. Send both sides bare, as
# the obvious code does, and it still works: vectors come back, similarity ranks them,
# nothing errors.
#
# Measured on this corpus over the 19 questions with a gradable gold set, recall@20 went
# from 0.002 to 0.016 when the prefix was added, and the number of questions retrieving
# anything at all went from 5 to 7. Eight times better and still close to zero, which is
# the honest way to report it: the query format mattered more than the choice of model,
# and neither rescued similarity search on a question set full of record numbers.
QUERY_PREFIX = os.environ.get(
    "EMBED_QUERY_PREFIX",
    "Instruct: Given a search query, retrieve relevant IT service management records "
    "that answer the query\nQuery: ")


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


def _call(texts: list[str], attempts: int = 4) -> list[list[float]]:
    """One /v1/embeddings request, retried on the failures that are worth retrying.

    ⛔ THE RESPONSE IS NOT NECESSARILY IN THE ORDER YOU SENT IT. The OpenAI schema gives
    every item an `index` for exactly this reason, and a server that batches internally
    is entitled to use it. Sorting on `index` costs nothing and the alternative is
    vectors quietly attached to the wrong chunks, which no test downstream can see.
    """
    body = json.dumps({"model": SERVED_AS, "input": texts}).encode()
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                ENDPOINT, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                payload = json.load(resp)
            rows = sorted(payload["data"], key=lambda d: d["index"])
            return [r["embedding"] for r in rows]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last = exc
            if attempt == attempts:
                break
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"{ENDPOINT} failed after {attempts} attempts: {last}")


def _normalise(arr: np.ndarray) -> np.ndarray:
    # Normalised on the way in, so similarity is a dot product rather than a division
    # repeated eighty thousand times per question.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-9)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a small list, for a question at query time.

    ⛔ QUERY VECTORS ARE CACHED ON DISK AND THE CORPUS CACHE DOES NOT COVER THEM. The
    corpus is embedded once into one array. A question is embedded every time an arm
    runs, and there are seven arms over a frozen question set, so the same handful of
    questions gets sent to the server dozens of times per evaluation.

    That matters for more than speed. The GPU in Part 8 is rented by the hour and gets
    torn down, and without this an evaluation could only ever be re-run while the server
    happened to be up. With it, the questions are embedded once and every later run of
    Part 10 works with no GPU at all.

    Keyed on a hash of the model name and the text, so changing either one misses the
    cache rather than returning a vector from the wrong model.
    """
    cache_path = CACHE / f"{MODEL}-queries.json"
    store: dict[str, list[float]] = {}
    if cache_path.exists():
        store = json.loads(cache_path.read_text())

    cut = [QUERY_PREFIX + t[:MAX_CHARS] for t in texts]
    keys = [hashlib.sha256(f"{MODEL}\0{t}".encode()).hexdigest() for t in cut]
    missing = [i for i, k in enumerate(keys) if k not in store]
    if missing:
        fresh = _call([cut[i] for i in missing])
        for i, vec in zip(missing, fresh):
            store[keys[i]] = vec
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(store))

    arr = np.asarray([store[k] for k in keys], dtype=np.float32)
    return _normalise(arr)


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

    windows = [(start, [t[:MAX_CHARS] for _, t in chunks[start:start + BATCH]])
               for start in range(0, len(chunks), BATCH)]

    # ⛔ THE WIDTH COMES FROM THE SERVER, NOT FROM A CONSTANT. This file used to hard code
    # 768 for nomic-embed-text. Point it at a model with a different width and a hard
    # coded array silently truncates every vector, similarity still returns a ranked
    # list, and every number in Part 10 is wrong with nothing to show for it.
    first = np.asarray(_call(windows[0][1]), dtype=np.float32)
    width = first.shape[1]
    out = np.zeros((len(chunks), width), dtype=np.float32)
    out[0:first.shape[0]] = first
    if not quiet:
        print(f"  {MODEL} returned {width} dimensions, embedding {len(chunks):,} chunks "
              f"at {BATCH} per request, {CONCURRENCY} in flight")

    t0, done = time.time(), first.shape[0]
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        def one(job: tuple[int, list[str]]) -> tuple[int, np.ndarray]:
            start, texts = job
            return start, np.asarray(_call(texts), dtype=np.float32)

        for start, block in pool.map(one, windows[1:]):
            out[start:start + block.shape[0]] = block
            done += block.shape[0]
            if not quiet and done % (BATCH * 40) < BATCH:
                rate = done / max(0.1, time.time() - t0)
                left = (len(chunks) - done) / max(0.1, rate)
                print(f"    {done:>7,}/{len(chunks):,}  {rate:>6.1f} chunks/s  "
                      f"about {left / 60:>4.0f} min left", flush=True)

    out = _normalise(out)
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
