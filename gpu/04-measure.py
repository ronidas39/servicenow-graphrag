"""Measure what the rented GPU actually delivers, and what a million tokens costs.

Part 8 section 87. Every number Part 8 publishes about speed and money is produced here,
on the machine, against the two servers 03-serve.sh started. Nothing is quoted from a
vendor page and nothing is scaled from somebody else's card.

⛔ ONE STREAM AND MANY STREAMS ARE DIFFERENT NUMBERS AND BOTH MATTER. A single request
measures what one person waiting at a keyboard feels. Sixteen at once measures what the
card is worth, and on a batching server it is several times higher. Publishing only the
first makes the hardware look bad; publishing only the second makes the wait look short.

⛔ COST IS DERIVED FROM THE RENTAL PRICE, NOT FROM A PRICE LIST. The rate is passed in
from the launch script, which read it from the AWS pricing API at launch time, so the
cost per million moves when AWS moves and cannot quietly go stale.

⛔ THE INPUT IS REAL TICKET TEXT where the corpus is available, because throughput
depends on token length and a lorem prompt measures a fiction.

Author: Roni Das
Created: 2026-09-10
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = (
    "You are an incident assistant for an IT service desk. Using only the notes below, "
    "explain in plain English what most likely caused this outage and what to check "
    "first.\n\nNotes: The payments API began returning 502 for roughly one request in "
    "four at 09:14. The load balancer health check stayed green. The database showed no "
    "slow queries. A configuration change to the ingress controller was applied at 09:11 "
    "by the platform team.\n\nAnswer:"
)


def post(url: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def one_generation(base: str, max_tokens: int) -> tuple[float, int, int]:
    """One request, start to finish. Returns seconds, prompt tokens, output tokens."""
    t0 = time.time()
    out = post(f"{base}/v1/chat/completions", {
        "model": "chat",
        "messages": [{"role": "user", "content": PROMPT}],
        # Temperature 0 so repeated runs measure the machine rather than the sampler.
        "max_tokens": max_tokens, "temperature": 0,
        # ⛔ Without this the model stops early on an easy prompt and the run measures
        # a shorter generation than the one it claims to.
        "ignore_eos": True,
    })
    dt = time.time() - t0
    u = out["usage"]
    return dt, u["prompt_tokens"], u["completion_tokens"]


def measure_generation(base: str, price_per_hour: float, max_tokens: int,
                       concurrencies: list[int], repeats: int) -> dict:
    results = {}

    print("\n  one request at a time")
    singles = [one_generation(base, max_tokens) for _ in range(repeats)]
    rates = [n / dt for dt, _, n in singles]
    results["single"] = {
        "runs": repeats,
        "output_tokens": singles[0][2],
        "prompt_tokens": singles[0][1],
        "seconds_median": round(statistics.median(dt for dt, _, _ in singles), 3),
        "tokens_per_second": round(statistics.median(rates), 1),
    }
    print(f"    {results['single']['tokens_per_second']} output tokens a second, "
          f"median of {repeats} runs")

    for c in concurrencies:
        print(f"\n  {c} requests at once")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=c) as pool:
            batch = list(pool.map(lambda _: one_generation(base, max_tokens), range(c)))
        wall = time.time() - t0
        total_out = sum(n for _, _, n in batch)
        per_request = statistics.median(dt for dt, _, _ in batch)
        results[f"concurrent_{c}"] = {
            "requests": c,
            "wall_seconds": round(wall, 3),
            "total_output_tokens": total_out,
            "tokens_per_second": round(total_out / wall, 1),
            "median_request_seconds": round(per_request, 3),
        }
        print(f"    {round(total_out / wall, 1)} output tokens a second across the card, "
              f"each request took {per_request:.1f}s")

    # ⛔ COST USES THE BEST THROUGHPUT THE CARD REACHED, and says so, because that is the
    # number a batch job pays. A single stream costs several times more per token and the
    # article prints both rather than picking the flattering one.
    best = max(v["tokens_per_second"] for k, v in results.items() if k != "single")
    per_second = price_per_hour / 3600.0
    results["cost"] = {
        "price_per_hour_usd": price_per_hour,
        "single_stream_usd_per_million_output_tokens":
            round(per_second / results["single"]["tokens_per_second"] * 1_000_000, 2),
        "batched_usd_per_million_output_tokens":
            round(per_second / best * 1_000_000, 2),
        "batched_tokens_per_second": best,
    }
    return results


def measure_embedding(base: str, price_per_hour: float, texts: list[str],
                      batch: int) -> dict:
    print(f"\n  embedding {len(texts):,} real chunks in batches of {batch}")
    t0, dims, done = time.time(), None, 0
    for i in range(0, len(texts), batch):
        out = post(f"{base}/v1/embeddings",
                   {"model": "embed", "input": texts[i:i + batch]})
        dims = len(out["data"][0]["embedding"])
        done += len(out["data"])
        if done % (batch * 10) == 0:
            print(f"    {done:,} of {len(texts):,}", end="\r", flush=True)
    wall = time.time() - t0
    chars = sum(len(t) for t in texts)
    per_second = price_per_hour / 3600.0
    res = {
        "texts": len(texts),
        "dimensions": dims,
        "batch_size": batch,
        "wall_seconds": round(wall, 2),
        "texts_per_second": round(len(texts) / wall, 1),
        "characters": chars,
        # A chunk is not a token, so this reports what was actually sent and leaves the
        # token estimate to the reader rather than inventing a ratio.
        "usd_per_million_texts": round(per_second / (len(texts) / wall) * 1_000_000, 2),
    }
    print(f"    {res['texts_per_second']} texts a second at {dims} dimensions")
    return res


def real_chunks(n: int) -> list[str] | None:
    """Real ticket text if the corpus was copied up, otherwise nothing."""
    p = pathlib.Path.home() / "corpus-sample.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", default="http://127.0.0.1:8000")
    ap.add_argument("--embed", default="http://127.0.0.1:8001")
    ap.add_argument("--price-per-hour", type=float, required=True)
    ap.add_argument("--instance-type", default="")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--concurrency", default="4,16,32")
    ap.add_argument("--embed-texts", type=int, default=512)
    ap.add_argument("--embed-batch", type=int, default=32)
    ap.add_argument("--out", default=str(pathlib.Path.home() / "gpu-measurements.json"))
    a = ap.parse_args()

    with urllib.request.urlopen(f"{a.chat}/v1/models", timeout=30) as r:
        chat_model = json.loads(r.read())["data"][0]["id"]
    with urllib.request.urlopen(f"{a.embed}/v1/models", timeout=30) as r:
        embed_model = json.loads(r.read())["data"][0]["id"]

    print(f"measuring {a.instance_type} at ${a.price_per_hour}/hour")
    concurrencies = [int(x) for x in a.concurrency.split(",")]
    gen = measure_generation(a.chat, a.price_per_hour, a.max_tokens,
                             concurrencies, a.repeats)

    texts = real_chunks(a.embed_texts)
    source = "real corpus chunks"
    if texts is None:
        texts = [PROMPT[:400] + f" case {i}" for i in range(a.embed_texts)]
        source = "synthetic text, the corpus was not copied up"
    emb = measure_embedding(a.embed, a.price_per_hour, texts, a.embed_batch)
    emb["source"] = source

    out = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "instance_type": a.instance_type,
        "price_per_hour_usd": a.price_per_hour,
        "chat_model": chat_model,
        "embedding_model": embed_model,
        "generation": gen,
        "embedding": emb,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    print(json.dumps(out["generation"]["cost"], indent=2))


if __name__ == "__main__":
    main()
