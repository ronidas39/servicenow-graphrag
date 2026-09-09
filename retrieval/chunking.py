"""Three ways to cut the same records into chunks, and why the choice decides the result.

An audit called chunking the dominant variable and said that comparing one configuration
was indefensible: if the vector arm loses a question with one chunk per incident, the
loss may be entirely a chunking artefact rather than anything inherent to a vector index.
So all three are built, all three are measured, and Part 10 reports all three.

The third one is the interesting one, and it is the experiment the article actually turns
on. It writes the graph INTO the text: the item's dependency path, the team that owns it,
the changes near it in time. If denormalising the graph into the chunk makes similarity
search answer a multi hop question, then the honest finding is not "graphs beat vectors".
It is "the graph was needed to build the index, not to query it", which is a more useful
sentence than the one this article set out to write.

⛔ NO EMBEDDING HAPPENS HERE. Chunking is a text decision and it costs nothing to run, so
it stays separate from the paid GPU step. Getting this wrong and finding out after paying
to embed sixty thousand records twice would be an expensive way to learn it.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Chunk:
    """One unit that will be embedded and retrieved."""

    chunk_id: str
    text: str
    strategy: str
    # What this chunk came from, so a hit can be traced back to a record.
    source_kind: str
    source_id: str
    # The item it concerns, when there is one. Used for grading, never for retrieval.
    ci_key: str | None = None
    tokens: int = 0


def approx_tokens(text: str) -> int:
    """Roughly four characters per token for English prose with identifiers in it.

    Deliberately approximate. The point is comparing chunk sizes between strategies, and
    a real tokenizer would tie this file to whichever model was chosen, which is a
    decision Part 8 has not made yet.
    """
    return max(1, len(text) // 4)


# The stored state values, spelled out. A corpus holding "6" cannot answer "what is the
# state of this ticket", and the reader of a retrieved chunk needs the word, not the code.
STATE_WORDS = {"1": "New", "2": "In Progress", "3": "On Hold",
               "6": "Resolved", "7": "Closed", "8": "Canceled"}


def _incident_text(inc: dict, include_notes: bool = True) -> str:
    # ⛔ THE STRUCTURED FIELDS BELONG IN THE TEXT TOO. Q01 asks what state a ticket is in
    # and who it was assigned to. Measured against the corpus, "state" and "assigned"
    # appeared in ZERO of 82,296 documents, so no retriever could have answered it at any
    # k. The index held the narrative and left out the fields, which is the fifth time in
    # this project that a scoring failure was a decision about the corpus.
    #
    # A real support system indexes both. Somebody searching for open payments incidents
    # assigned to a particular group is asking about fields, not about prose.
    # The dataset carries no state column, the same way many real extracts do not. It is
    # derived from whether the ticket was resolved, and the derivation is written down
    # here rather than assumed by a reader of the chunk.
    state = STATE_WORDS["6"] if inc.get("resolved_at") else STATE_WORDS["2"]
    facts = [f"State: {state}."]
    if inc.get("resolved_at"):
        facts.append(f"Resolved at {inc['resolved_at'][:16].replace('T', ' ')}.")
    if inc.get("assignment_group"):
        facts.append(f"Assigned to {inc['assignment_group']}.")
    if inc.get("category"):
        facts.append(f"Category {inc['category']}.")
    if inc.get("priority"):
        facts.append(f"Priority {inc['priority']}.")
    parts = [" ".join(facts), inc["short_description"], inc["description"]]
    if include_notes:
        parts += [note[2] for note in inc.get("work_notes", [])]
    if inc.get("close_notes"):
        parts.append(f"Resolution: {inc['close_notes']}")
    return "\n\n".join(p for p in parts if p)


# ── strategy one: the whole record, which is what most people build first ──────────

def per_record(incidents: Iterable[dict]) -> list[Chunk]:
    """One chunk per incident, everything it knows in one blob.

    The obvious choice, and the one most tutorials stop at. Its weakness is that a long
    ticket dilutes its own distinctive sentence: the symptom that matters ends up
    averaged with five work notes saying "looking now".
    """
    out = []
    for inc in incidents:
        # ⛔ THE RECORD NUMBER GOES IN THE TEXT, AND IT DID NOT. Q01 asks "what is the
        # state of INC2000042", which is the simplest retrieval there is, and every arm
        # scored zero on it. Not because retrieval failed: the chunk for INC2000042 did
        # not contain the string INC2000042 anywhere, so nothing could match it.
        #
        # That is the fourth time in this project that a scoring failure turned out to be
        # a corpus decision. An index cannot return a record by an identifier it was never
        # given, and a person asking about a ticket by number is the most ordinary request
        # a support system receives.
        text = f"{inc['number']}\n\n{_incident_text(inc)}"
        out.append(Chunk(
            chunk_id=f"{inc['number']}#whole",
            text=text, strategy="per_record",
            source_kind="incident", source_id=inc["number"],
            ci_key=inc.get("ci_key"), tokens=approx_tokens(text),
        ))
    return out


# ── strategy two: one chunk per field, so a short signal is not diluted ────────────

def per_field(incidents: Iterable[dict]) -> list[Chunk]:
    """Separate chunks for the symptom, the body, and each work note.

    A pasted stack trace becomes its own chunk instead of being averaged into a
    paragraph, which is why this should beat per_record on the questions about pasted
    output. It costs more chunks and more storage, which the article reports.
    """
    out = []
    for inc in incidents:
        n = inc["number"]
        fields = [("symptom", inc["short_description"]), ("body", inc["description"])]
        fields += [(f"note{i}", note[2]) for i, note in enumerate(inc.get("work_notes", []))]
        if inc.get("close_notes"):
            fields.append(("resolution", inc["close_notes"]))
        for label, text in fields:
            if not text or len(text.strip()) < 12:
                continue
            out.append(Chunk(
                chunk_id=f"{n}#{label}", text=text.strip(), strategy="per_field",
                source_kind="incident", source_id=n,
                ci_key=inc.get("ci_key"), tokens=approx_tokens(text),
            ))
    return out


# ── strategy three: write the graph into the text ─────────────────────────────────

def graph_denormalised(
    incidents: Iterable[dict],
    ci_by_key: dict[str, dict],
    depends_on: dict[str, list[str]],
    supports: dict[str, list[str]],
    changes_by_ci: dict[str, list[dict]],
) -> list[Chunk]:
    """One chunk per incident, with its neighbourhood spelled out in words.

    ⛔ THIS IS THE EXPERIMENT, NOT A BASELINE. Everything a traversal would find is
    written into the chunk as sentences: what the item runs on, what depends on it, who
    owns it, what changed near it. If similarity search then answers a multi hop question
    it was never supposed to answer, the finding is that the graph was needed to BUILD
    the index rather than to query it.

    It is also the honest limit of the technique, and the article must say so: this
    inlines the graph as it was on the day the index was built. The moment a dependency
    changes, every chunk touching it is stale, and there is nothing in a vector index
    that knows. A traversal reads the graph as it is now.
    """
    out = []
    for inc in incidents:
        key = inc.get("ci_key")
        # Starts from exactly what per_record produces, including the record number, so
        # this strategy stays a strict superset of the plain one. A test asserts that,
        # because "denormalising adds" is the claim the whole experiment rests on.
        lines = [f"{inc['number']}\n\n{_incident_text(inc)}"]

        if key and key in ci_by_key:
            ci = ci_by_key[key]
            lines.append(
                f"This happened on {ci['name']}, a {ci['sys_class_name']} in the "
                f"{ci['environment']} environment, {ci['region']} region, owned by the "
                f"{ci['domain']} team."
            )
            needs = [ci_by_key[k]["name"] for k in sorted(depends_on.get(key, ()))
                     if k in ci_by_key][:6]
            if needs:
                lines.append(f"{ci['name']} needs {', '.join(needs)} in order to work.")
            affected = [ci_by_key[k]["name"] for k in sorted(supports.get(key, ()))
                        if k in ci_by_key][:6]
            if affected:
                lines.append(
                    f"If {ci['name']} stops working, {', '.join(affected)} "
                    f"{'stop' if len(affected) > 1 else 'stops'} working too.")
            near = changes_by_ci.get(key, [])[:3]
            if near:
                lines.append("Recent work on it: "
                             + "; ".join(c["short_description"] for c in near) + ".")

        text = "\n\n".join(lines)
        out.append(Chunk(
            chunk_id=f"{inc['number']}#graph", text=text, strategy="graph_denormalised",
            source_kind="incident", source_id=inc["number"],
            ci_key=key, tokens=approx_tokens(text),
        ))
    return out


# ── the other record types, which every strategy needs ────────────────────────────

def knowledge_chunks(articles: Iterable[dict]) -> list[Chunk]:
    """Knowledge articles, split on their own headings.

    These are written for a reader rather than for a queue, which makes them the thing a
    text search should be best at, and the reason "has this happened before" is in the
    question set as a hybrid case rather than a graph one.
    """
    out = []
    for art in articles:
        sections = re.split(r"\n(?=## )", art["text"])
        for i, section in enumerate(sections):
            body = section.strip()
            if len(body) < 20:
                continue
            text = f"{art['number']}\n\n{art['short_description']}\n\n{body}"
            out.append(Chunk(
                chunk_id=f"{art['number']}#s{i}", text=text, strategy="any",
                source_kind="knowledge", source_id=art["number"],
                tokens=approx_tokens(text),
            ))
    return out


def problem_chunks(problems: Iterable[dict]) -> list[Chunk]:
    out = []
    for prb in problems:
        text = (f"{prb['number']}\n\n{prb['short_description']}\n\n"
                f"{prb['description']}\n\n{prb['cause_notes']}\n\n"
                f"Workaround: {prb['workaround']}")
        out.append(Chunk(
            chunk_id=f"{prb['number']}#whole", text=text, strategy="any",
            source_kind="problem", source_id=prb["number"],
            ci_key=prb.get("ci_key"), tokens=approx_tokens(text),
        ))
    return out


def report(chunks: list[Chunk]) -> str:
    """What each strategy costs, which is half of the comparison Part 10 reports."""
    from collections import Counter
    by = Counter(c.strategy for c in chunks)
    lines = ["strategy              chunks     tokens   mean  median    max"]
    for strategy in sorted(by):
        sizes = sorted(c.tokens for c in chunks if c.strategy == strategy)
        total = sum(sizes)
        lines.append(
            f"  {strategy:20s} {len(sizes):>7,} {total:>10,} "
            f"{total / len(sizes):>6.0f} {sizes[len(sizes) // 2]:>7,} {sizes[-1]:>6,}")
    return "\n".join(lines)


def ci_chunks(cis: Iterable[dict], depends_on: dict[str, list[str]],
              supports: dict[str, list[str]], names: dict[str, str]) -> list[Chunk]:
    """A retrievable document per configuration item.

    ⛔ AN INDEX OF TICKETS CANNOT ANSWER A QUESTION ABOUT A SERVER, and the first version
    of this corpus held only incidents. Every question about infrastructure scored zero
    for every arm, not because retrieval failed but because the answer was not in the
    index at all. The gold sets named items and the corpus contained tickets, so the two
    id spaces could never intersect.

    That is worth a paragraph in the article rather than a silent fix: what you put in
    the index decides which questions are answerable, before any retriever is chosen.
    """
    out = []
    for ci in cis:
        key = ci["key"]
        # ⛔ SORTED, BECAUSE THESE ARE SETS. Python randomises string hashing per
        # process, so iterating a set of keys gives a different order in every run, and
        # the [:8] then picks a different eight neighbours. The chunk text changed on
        # every build, which made the corpus unreproducible and silently invalidated the
        # embedding cache after 75 minutes of work.
        #
        # This is the same defect that was fixed in the dataset generator earlier, in a
        # different file. Sorting is the entire fix, and it is the reason the article can
        # claim a reader gets the same numbers.
        needs = [names[k] for k in sorted(depends_on.get(key, ())) if k in names][:8]
        holds_up = [names[k] for k in sorted(supports.get(key, ())) if k in names][:8]
        # ⛔ BOTH THE KEY AND THE NAME. An item is addressed two ways: engineers type the
        # name they see on a screen, lnx0525, while every gold set, every relationship row
        # and every join uses the key. A chunk holding only one of them is unreachable by
        # the other, and a spot check that happened to pick a cluster passed by accident
        # because for that one class the key and the name are the same string.
        lines = [
            f"{key} {ci['name']} is a {ci['sys_class_name']} in the "
            f"{ci['environment']} environment, {ci['region']} region, owned by the "
            f"{ci['domain']} team.",
        ]
        if needs:
            lines.append(f"{ci['name']} depends on {', '.join(needs)}.")
        if holds_up:
            lines.append(
                f"If {ci['name']} stops working, {', '.join(holds_up)} "
                f"{'stop' if len(holds_up) > 1 else 'stops'} working too.")
        if ci.get("last_discovered"):
            lines.append(f"Last confirmed by {ci['discovery_source']} on "
                         f"{ci['last_discovered'][:10]}.")
        text = " ".join(lines)
        out.append(Chunk(
            chunk_id=f"{key}#ci", text=text, strategy="any",
            source_kind="configuration_item", source_id=key,
            ci_key=key, tokens=approx_tokens(text),
        ))
    return out


def change_chunks(changes: Iterable[dict], names: dict[str, str]) -> list[Chunk]:
    """A retrievable document per change request.

    ⛔ THE THIRD TIME THE SAME LESSON LANDED. The corpus held incidents, then incidents
    and items, and questions about changes still scored zero for every arm because there
    were no changes in it. Nothing about the retrievers was wrong. What goes into the
    index decides which questions are answerable, and that is settled long before a
    retrieval strategy is chosen.
    """
    out = []
    for ch in changes:
        item = names.get(ch.get("ci_key") or "", "an unrecorded item")
        when = (ch.get("actual_end") or ch.get("planned_end") or "")[:16].replace("T", " ")
        text = (f"{ch['number']} {ch['change_type']} change on {item}: "
                f"{ch['short_description']}. {ch.get('description', '')} "
                f"Finished {when}. Outcome: {ch.get('close_code', 'unknown')}.")
        out.append(Chunk(
            chunk_id=f"{ch['number']}#chg", text=text.strip(), strategy="any",
            source_kind="change", source_id=ch["number"],
            ci_key=ch.get("ci_key"), tokens=approx_tokens(text),
        ))
    return out
