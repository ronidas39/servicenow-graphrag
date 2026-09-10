"""Grade the ANSWER, not the retrieval, and check the grader before believing it.

Section 108b has said "not measured here" since the article was drafted. Everything in
Part 10 measures whether the right records were found. Nobody deploys retrieval; they
deploy an answer, and an arm can hand over every supporting record and still produce a
wrong sentence.

⛔ A JUDGE NOBODY CHECKED IS NOT A MEASUREMENT, so this runs the checks first and reports
them beside the scores rather than underneath them:

  1. A PLANTED CONTROL. Before grading anything real, the judge is shown an answer built
     from the gold records and an answer built from records chosen at random. If it
     cannot separate those two, its opinion about the arms is worthless and this stops.
  2. SELF CONSISTENCY. Every answer is graded twice. The disagreement rate is published.
     A grader that contradicts itself is measuring its own temperature.
  3. AGREEMENT WITH THE MECHANICAL GOLD. On the questions where correctness is decidable
     from the data, the judge's verdict is compared with what the gold set says. That is
     a harder test than agreeing with a person, because the gold cannot be talked round.

⛔ THE MODEL AND THE PROMPT ARE PUBLISHED, both here and in the article, because a grade
from an unnamed model behind an unnamed prompt is an opinion wearing a number.

⛔ IT GRADES ON THE CONTEXT THE ARM ACTUALLY RETRIEVED, not on the whole corpus. The
question is what the arm made possible, so the answer is generated from its context and
nothing else, and "I cannot answer from this" is a valid and correct output when the
context does not hold the answer.

Run: CHAT_BASE_URL=http://127.0.0.1:8000 python3 retrieval/judge.py

Author: Roni Das
Created: 2026-09-10
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import re
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "generator"))
sys.path.insert(0, str(HERE.parent / "questions"))

from gold import World, build as build_gold  # noqa: E402
from questions import QUESTIONS  # noqa: E402
from run import build_corpus  # noqa: E402

CHAT = os.environ.get("CHAT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("CHAT_SERVED_AS", "chat")
OUT = HERE.parent / "results" / "captures" / "answer-quality.json"
SEED = 20260910

# ⛔ THE PROMPTS ARE CONSTANTS SO THEY CAN BE PRINTED IN THE ARTICLE VERBATIM.
ANSWER_PROMPT = (
    "You are answering a question about an IT service management estate.\n"
    "Use ONLY the records below. Do not use anything you know from elsewhere.\n"
    "If the records do not contain the answer, reply exactly: CANNOT ANSWER FROM THESE "
    "RECORDS.\n"
    "Answer in at most three sentences.\n\n"
    "RECORDS:\n{context}\n\nQUESTION: {question}\nANSWER:"
)

JUDGE_PROMPT = (
    "You are grading one answer against a list of the records that a correct answer "
    "must be based on.\n"
    "Reply with one word and nothing else.\n"
    "CORRECT   the answer is supported by the records and addresses the question\n"
    "WRONG     the answer states something the records do not support\n"
    "REFUSED   the answer declines to answer\n\n"
    "QUESTION: {question}\n\n"
    "RECORDS A CORRECT ANSWER RESTS ON:\n{gold}\n\n"
    "ANSWER TO GRADE:\n{answer}\n\nVERDICT:"
)

VERDICTS = ("CORRECT", "WRONG", "REFUSED")


def ask(prompt: str, temperature: float = 0.0, max_tokens: int = 220) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(f"{CHAT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                payload = json.load(r)
            return payload["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:
            if attempt == 2:
                raise RuntimeError(f"the chat server did not answer: {exc}") from exc
            time.sleep(3)
    return ""


def verdict(question: str, gold_text: str, answer: str) -> str:
    """One word out of the judge, or UNPARSED, which is counted rather than guessed."""
    raw = ask(JUDGE_PROMPT.format(question=question, gold=gold_text, answer=answer),
              max_tokens=8).upper()
    for v in VERDICTS:
        if v in raw:
            return v
    return "UNPARSED"


def main() -> int:
    world = World()
    gold = build_gold(world)
    chunks = build_corpus(world)
    by_id: dict[str, str] = {}
    for c in chunks:
        by_id.setdefault(c.source_id, c.text)

    # ⛔ THE ARMS ARE RE-RUN HERE RATHER THAN READ FROM scores.json, because that file
    # keeps the score and not the context. Grading an answer needs the exact records the
    # arm handed over, so the arms run again and the answer is generated from that.
    from neo4j import GraphDatabase
    from arms import (BM25Only, GraphOnly, Hybrid, HybridCypher, NoRetrieval,
                      Text2Cypher, VectorCypher, VectorOnly)
    from env import env_path

    env = {}
    for line in env_path().read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    driver = GraphDatabase.driver(env["NEO4J_URI"],
                                  auth=(env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"]))
    from embed import build as build_vectors, corpus_fingerprint

    pairs = [(c.chunk_id, c.text) for c in chunks]
    names = {c["name"]: k for k, c in world.cis.items() if c.get("name")}
    vectors = build_vectors(pairs, quiet=True)
    fingerprint = corpus_fingerprint(pairs)
    vec = VectorOnly(pairs, vectors, fingerprint=fingerprint)
    bm = BM25Only(pairs)
    hyb = Hybrid(bm, vec, chunks=pairs)
    vc = VectorCypher(driver.session)
    arms = {"no_retrieval": NoRetrieval(pairs), "bm25": bm, "vector": vec,
            "hybrid": hyb, "graph_only": GraphOnly(driver.session, names),
            "vector_cypher": vc, "hybrid_cypher": HybridCypher(hyb, driver.session),
            "text2cypher": Text2Cypher(driver.session, CHAT)}

    asked = {q.id: q for q in QUESTIONS}
    gradable = [q for q in QUESTIONS
                if gold.get(q.id) and gold[q.id].supporting
                and len(gold[q.id].supporting) <= 6]
    rows = []
    for name, arm in arms.items():
        for q in gradable:
            r = arm.retrieve(q.text, q.id)
            if r.error or not r.context:
                continue
            rows.append({"arm": name, "question_id": q.id, "context": r.context,
                         "recall_at_k": (len(set(r.record_ids) & gold[q.id].supporting)
                                         / len(gold[q.id].supporting))})
    print(f"  {len(rows)} answers to grade, over {len(gradable)} questions "
          f"and {len(arms)} arms\n")

    print(f"  judge model: {MODEL} at {CHAT.split('//')[-1]}, temperature 0\n")

    # ── check 1: can it tell a good answer from a bad one at all? ────────────────
    print("  CONTROL, before anything real is graded")
    rng = random.Random(SEED)
    controls: list[tuple[str, str]] = []
    for qid in list(gold)[:60]:
        g = gold[qid]
        if not g or not g.supporting or len(g.supporting) > 6:
            continue
        q = asked.get(qid)
        if not q:
            continue
        good_ids = sorted(g.supporting)
        gold_text = "\n".join(f"- {i}" for i in good_ids)
        good = "\n\n".join(by_id.get(i, i) for i in good_ids)[:4000]
        bad_ids = rng.sample([c.source_id for c in chunks], len(good_ids))
        bad = "\n\n".join(by_id.get(i, i) for i in bad_ids)[:4000]
        controls.append((verdict(q.text, gold_text,
                                 ask(ANSWER_PROMPT.format(context=good,
                                                          question=q.text))),
                         verdict(q.text, gold_text,
                                 ask(ANSWER_PROMPT.format(context=bad,
                                                          question=q.text)))))
        if len(controls) >= 6:
            break

    on_gold = sum(1 for a, _ in controls if a == "CORRECT")
    on_random = sum(1 for _, b in controls if b == "CORRECT")
    print(f"    answers built from the gold records:   {on_gold} of {len(controls)} CORRECT")
    print(f"    answers built from random records:     {on_random} of {len(controls)} CORRECT")
    if not controls or on_gold <= on_random:
        print("\n  ⛔ THE JUDGE CANNOT SEPARATE A GOOD ANSWER FROM A RANDOM ONE. Its "
              "opinion about the arms would be noise, so nothing else is reported.")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"usable": False, "on_gold": on_gold,
                                   "on_random": on_random,
                                   "controls": len(controls)}, indent=2) + "\n")
        return 1
    print(f"    it separates them, so the grades below are worth reading\n")

    # ── grade every arm on every question it was scored on ──────────────────────
    per_arm: dict[str, list[str]] = {}
    flips = 0
    graded = 0
    agree_gold = 0
    comparable = 0
    detail = []
    for r in rows:
        q = asked.get(r["question_id"])
        g = gold.get(r["question_id"])
        if not q or not g or not g.supporting:
            continue
        ctx = r.get("context") or ""
        if not ctx:
            continue
        gold_text = "\n".join(f"- {i}" for i in sorted(g.supporting))[:2000]
        answer = ask(ANSWER_PROMPT.format(context=ctx[:6000], question=q.text))
        v1 = verdict(q.text, gold_text, answer)
        v2 = verdict(q.text, gold_text, answer)     # check 2: self consistency
        graded += 1
        if v1 != v2:
            flips += 1
        per_arm.setdefault(r["arm"], []).append(v1)
        # check 3: does the verdict line up with what the gold already knows?
        if r["recall_at_k"] is not None:
            comparable += 1
            found_everything = r["recall_at_k"] >= 0.999
            says_correct = v1 == "CORRECT"
            if found_everything == says_correct:
                agree_gold += 1
        detail.append({"arm": r["arm"], "question": r["question_id"],
                       "recall": r["recall_at_k"], "verdict": v1,
                       "second_verdict": v2, "answer": answer[:400]})
        print(f"    {r['arm']:16s} {r['question_id']:5s} recall "
              f"{r['recall_at_k']:.2f}  ->  {v1}")

    print(f"\n  RELIABILITY OF THE JUDGE")
    print(f"    graded twice, disagreed with itself on {flips} of {graded} "
          f"({100 * flips / max(1, graded):.0f}%)")
    print(f"    agreed with the mechanical gold on {agree_gold} of {comparable} "
          f"({100 * agree_gold / max(1, comparable):.0f}%)")

    print(f"\n  ANSWER QUALITY BY ARM")
    print(f"    {'arm':16s} {'correct':>8s} {'wrong':>7s} {'refused':>8s} {'graded':>7s}")
    summary = {}
    for arm, vs in sorted(per_arm.items(),
                          key=lambda kv: -sum(1 for v in kv[1] if v == "CORRECT")):
        c = sum(1 for v in vs if v == "CORRECT")
        w = sum(1 for v in vs if v == "WRONG")
        rf = sum(1 for v in vs if v == "REFUSED")
        summary[arm] = {"correct": c, "wrong": w, "refused": rf, "graded": len(vs)}
        print(f"    {arm:16s} {c:>8d} {w:>7d} {rf:>8d} {len(vs):>7d}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "usable": True,
        "model": MODEL,
        "temperature": 0.0,
        "answer_prompt": ANSWER_PROMPT,
        "judge_prompt": JUDGE_PROMPT,
        "control": {"cases": len(controls), "on_gold": on_gold, "on_random": on_random},
        "self_consistency": {"graded": graded, "disagreed": flips},
        "agreement_with_gold": {"comparable": comparable, "agreed": agree_gold},
        "by_arm": summary,
        "detail": detail,
    }, indent=2) + "\n")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
