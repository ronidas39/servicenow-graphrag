"""Tests for the three chunking strategies.

An audit called chunking the dominant variable in a retrieval comparison, and warned that
reporting one configuration is indefensible: a loss on one chunking may say nothing about
a vector index and everything about how the text was cut.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import pathlib
import sys

import pytest

R = pathlib.Path(__file__).resolve().parent.parent / "retrieval"
sys.path.insert(0, str(R))

from chunking import (  # noqa: E402
    change_chunks,
    ci_chunks,
    graph_denormalised,
    knowledge_chunks,
    per_field,
    per_record,
    problem_chunks,
)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "questions"))
from gold import World  # noqa: E402


@pytest.fixture
def incident():
    return {
        "number": "INC2000001",
        "short_description": "lnx0525: disk on the primary is nearly full",
        "description": "Disk on the primary is nearly full.\n\n```\n$ df -h\n97%\n```",
        "close_notes": "Cleared the old logs.",
        "ci_key": "host-payments-prd-042-0",
        "work_notes": [["2026-03-01T09:00:00+00:00", "user0001", "looking now"],
                       ["2026-03-01T10:00:00+00:00", "user0002", "restarted, watching"]],
    }


@pytest.fixture
def world():
    return {
        "ci_by_key": {
            "host-payments-prd-042-0": {
                "name": "lnx0525", "sys_class_name": "cmdb_ci_linux_server",
                "environment": "prd", "region": "eu-west", "domain": "payments"},
            "db-payments-prd-042": {"name": "pg0071", "sys_class_name": "cmdb_ci_server",
                                    "environment": "prd", "region": "eu-west",
                                    "domain": "payments"},
            "svc-payments-prd-042": {"name": "payments service 042 (prd)",
                                     "sys_class_name": "cmdb_ci_service",
                                     "environment": "prd", "region": "eu-west",
                                     "domain": "payments"},
        },
        "depends_on": {"host-payments-prd-042-0": ["db-payments-prd-042"]},
        "supports": {"host-payments-prd-042-0": ["svc-payments-prd-042"]},
        "changes_by_ci": {"host-payments-prd-042-0": [
            {"short_description": "Resize the volume on lnx0525"}]},
    }


def test_every_strategy_keeps_the_symptom(incident):
    """Whatever the cut, the sentence that says what went wrong must survive. A chunking
    that loses it is not a configuration choice, it is a bug."""
    for chunks in (per_record([incident]), per_field([incident])):
        joined = " ".join(c.text for c in chunks)
        assert "disk on the primary is nearly full" in joined.lower()


def test_per_field_isolates_a_pasted_block(incident):
    """A stack trace averaged into a paragraph is diluted. On its own it is a strong
    signal, which is why the question set has questions about pasted output."""
    chunks = per_field([incident])
    assert len(chunks) > len(per_record([incident]))
    assert any("df -h" in c.text for c in chunks)


def test_every_chunk_can_be_traced_back_to_its_record(incident):
    for chunks in (per_record([incident]), per_field([incident])):
        for c in chunks:
            assert c.source_id == "INC2000001"
            assert c.source_kind == "incident"
            assert c.chunk_id.startswith("INC2000001#")


def test_chunk_ids_are_unique_within_a_strategy(incident):
    for chunks in (per_record([incident]), per_field([incident])):
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


def test_the_denormalised_chunk_writes_the_graph_into_the_text(incident, world):
    """THE EXPERIMENT. Everything a one hop traversal would find is spelled out in
    words. If similarity search then answers a question it was never supposed to answer,
    the finding is that the graph was needed to BUILD the index, not to query it."""
    chunk = graph_denormalised([incident], **world)[0]
    assert "pg0071" in chunk.text, "what the item depends on is missing"
    assert "payments service 042 (prd)" in chunk.text, "the blast radius is missing"
    assert "payments team" in chunk.text, "who owns it is missing"
    assert "Resize the volume" in chunk.text, "recent work on it is missing"


def test_the_denormalised_chunk_is_strictly_bigger(incident, world):
    """It buys reach with tokens, and the article reports that cost rather than hiding
    it behind an accuracy number."""
    plain = per_record([incident])[0]
    rich = graph_denormalised([incident], **world)[0]
    assert rich.tokens > plain.tokens
    assert plain.text in rich.text, "denormalising must add, never replace"


def test_an_incident_with_no_item_still_produces_a_chunk(world):
    """Seventeen percent of tickets have no configuration item. They must still be
    retrievable, or the vector arm silently loses a sixth of the corpus."""
    orphan = {"number": "INC2000002", "short_description": "one of the payments boxes: slow",
              "description": "Slow.", "close_notes": "", "ci_key": None, "work_notes": []}
    chunks = graph_denormalised([orphan], **world)
    assert len(chunks) == 1
    assert chunks[0].text.strip()
    assert chunks[0].ci_key is None


def test_a_knowledge_article_is_split_on_its_headings():
    art = {"number": "KB9001", "short_description": "How to handle capacity faults",
           "text": "## Symptom\n\nDisk full.\n\n## What to do now\n\nClear the logs.\n\n"
                   "## Permanent fix\n\nResize."}
    chunks = knowledge_chunks([art])
    assert len(chunks) >= 3, "each heading should become its own retrievable section"
    for c in chunks:
        # The stated intent is that every fragment carries its context. It used to be
        # written as startswith, which was stricter than the intent and broke when the
        # article number was added to the front. Both facts are now required, which is
        # more than the original checked, not less.
        assert "How to handle capacity faults" in c.text, (
            "every section must carry the article title, or a retrieved fragment has no "
            "context at all")
        assert "KB9001" in c.text, (
            "every section must carry the article number, or it cannot be retrieved by it")


def test_every_chunk_carries_its_own_record_number():
    """⛔ AN INDEX CANNOT RETURN A RECORD BY AN IDENTIFIER IT WAS NEVER GIVEN.

    Q01 asks what the state of INC2000042 is. That is the simplest retrieval there is, a
    record named directly, and every arm scored zero on it. The cause was not retrieval:
    the chunk for INC2000042 did not contain the string INC2000042 anywhere, so nothing
    could ever match it. Changes and configuration items happened to include their ids;
    incidents, problems and knowledge articles did not.

    This is the fourth time in this project that a scoring failure turned out to be a
    decision about what went into the corpus rather than anything about the retriever.
    """
    world = World()
    names = {k: c["name"] for k, c in world.cis.items()}
    families = {
        "incident": per_record(world.incidents[:50]),
        "problem": problem_chunks(world.problems[:20]),
        "knowledge": knowledge_chunks(world.knowledge[:20]),
        "change": change_chunks(world.changes[:20], names),
        "configuration_item": ci_chunks(list(world.cis.values())[:20], world.depends_on,
                                        world.supports, names),
    }
    for kind, chunks in families.items():
        assert chunks, f"no {kind} chunks were produced, so nothing was checked"
        missing = [c.source_id for c in chunks if c.source_id not in c.text]
        assert not missing, (
            f"{len(missing)} {kind} chunk(s) do not contain their own identifier, so no "
            f"retriever can find them by it. First: {missing[0]}")


def test_the_corpus_is_byte_identical_across_processes():
    """⛔ THE CHUNK TEXT CHANGED ON EVERY BUILD, AND IT COST 75 MINUTES OF EMBEDDING.

    `ci_chunks` and `graph_denormalised` read neighbours out of a set and then keep the
    first eight. Python randomises string hashing per process, so the set iterated in a
    different order every run and a different eight neighbours went into the text. The
    corpus was therefore unreproducible, the embedding cache was invalidated the moment
    it was written, and a reader following the article would not get these numbers.

    ⛔ THIS TEST MUST FORK. Set iteration order is stable WITHIN one process, so checking
    twice in this one proves nothing at all. Only separate interpreters, each with their
    own hash seed, can see the fault.

    This is the same defect that was found and fixed in the dataset generator earlier, in
    a different file. Sorting is the whole fix.
    """
    import subprocess
    root = pathlib.Path(__file__).resolve().parent.parent
    script = (
        "import sys;"
        f"sys.path.insert(0, {str(root / 'retrieval')!r});"
        f"sys.path.insert(0, {str(root / 'questions')!r});"
        "from run import build_corpus;"
        "from embed import corpus_fingerprint;"
        "from gold import World;"
        "c = build_corpus(World());"
        "print(corpus_fingerprint([(x.chunk_id, x.text) for x in c]))"
    )
    seen = set()
    for _ in range(3):
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-500:]
        seen.add(out.stdout.strip())
    assert len(seen) == 1, (
        f"the corpus text differs between processes: {len(seen)} distinct fingerprints "
        f"from 3 runs. A set is being iterated somewhere without sorting it.")
