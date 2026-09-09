"""Tests for the ServiceNow loader.

An audit found this file had zero coverage, and it is the only code that decides what a
live instance actually receives. Three of the defects it found were one unit test each:

  · incidents and changes carried no cmdb_ci, so the instance held sixty thousand
    tickets attached to nothing and the graph could not be rebuilt from the platform
  · a resolved incident needs close_code, close_notes and resolved_at TOGETHER, and any
    subset makes the insert return null with no error at all
  · a retry replayed rows the server had already committed, because correlation_id was
    written on every row and never read

Nothing here touches a live instance. The transport is stubbed, which is the right level:
these are decisions about payload shape and retry behaviour, not questions about whether
ServiceNow is up.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

GEN = pathlib.Path(__file__).resolve().parent.parent / "generator"
sys.path.insert(0, str(GEN))

import load_servicenow as L  # noqa: E402


# ── fixtures ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def incident_row():
    return {
        "number": "INC2000001",
        "short_description": "lnx0341: disk on the primary is nearly full",
        "description": "Disk on the primary is nearly full.",
        "category": "capacity",
        "priority": 3,
        "opened_at": "2026-03-01T09:14:22+00:00",
        "resolved_at": "2026-03-01T15:02:00+00:00",
        "close_notes": "Cleared the old logs.",
        "ci_key": "host-payments-prd-042-0",
    }


@pytest.fixture
def sys_ids():
    return {"host-payments-prd-042-0": "aaaabbbbccccddddeeeeffff00001111"}


# ── the mappers decide what the instance receives ──────────────────────────────────

def test_an_incident_carries_its_configuration_item(incident_row, sys_ids):
    """The defect that made the whole premise impossible. Without cmdb_ci the instance
    holds tickets attached to nothing and no graph can be built from ServiceNow."""
    out = L.to_incident(incident_row, sys_ids)
    assert out["cmdb_ci"] == sys_ids["host-payments-prd-042-0"]


def test_an_incident_with_no_known_item_simply_omits_the_field(incident_row):
    """A ticket whose item was never loaded must not send an empty reference, which
    ServiceNow would reject for the whole batch."""
    out = L.to_incident(incident_row, {})
    assert "cmdb_ci" not in out


def test_an_unlinked_incident_omits_the_field(incident_row, sys_ids):
    row = {**incident_row, "ci_key": None}
    assert "cmdb_ci" not in L.to_incident(row, sys_ids)


def test_a_resolved_incident_sends_all_four_fields_or_none(incident_row, sys_ids):
    """Measured field by field against a live instance: state alone inserts 0 of 6,
    state with close_code inserts 0 of 6, all four together inserts 6 of 6. Nothing
    tells you which field is missing, so the shape is asserted here instead."""
    out = L.to_incident(incident_row, sys_ids)
    for field in ("state", "close_code", "close_notes", "resolved_at"):
        assert out.get(field), f"a resolved incident must send {field}"

    unresolved = L.to_incident({**incident_row, "resolved_at": None}, sys_ids)
    for field in ("state", "close_code", "close_notes", "resolved_at"):
        assert field not in unresolved, f"an open incident must not send {field}"


def test_timestamps_are_sent_in_the_format_servicenow_parses(incident_row, sys_ids):
    out = L.to_incident(incident_row, sys_ids)
    assert out["opened_at"] == "2026-03-01 09:14:22"
    assert "T" not in out["opened_at"] and "+" not in out["opened_at"]


def test_a_change_carries_its_configuration_item(sys_ids):
    row = {
        "number": "CHG100001", "short_description": "Resize the volume",
        "description": "Resize.", "change_type": "normal", "risk": "low",
        "planned_start": "2026-03-01T02:00:00+00:00",
        "planned_end": "2026-03-01T04:00:00+00:00",
        "actual_start": "2026-03-01T02:05:00+00:00",
        "actual_end": "2026-03-01T04:11:00+00:00",
        "ci_key": "host-payments-prd-042-0",
    }
    out = L.to_change(row, sys_ids)
    assert out["cmdb_ci"] == sys_ids["host-payments-prd-042-0"]
    # Planned and actual are different fields and the article turns on the difference.
    assert out["start_date"] != out["work_start"]


def test_a_knowledge_article_carries_the_mandatory_knowledge_base():
    """kb_knowledge_base is mandatory and nothing says so. Measured: 0 of 4 insert
    without it, 4 of 4 with it, and no error either way."""
    L.KB_BASE = "kbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkb"
    out = L.to_knowledge({"short_description": "How to clear the log volume",
                          "text": "## Symptom", "category": "capacity"})
    assert out["kb_knowledge_base"] == "kbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkb"
    assert out["short_description"]


def test_every_table_the_run_order_names_has_a_mapper():
    """The run order and the mapper table are edited separately, so a phase can be added
    to one and forgotten in the other. Knowledge was in the docstring and in neither."""
    for name in ("incidents", "changes", "problems", "knowledge"):
        assert name in L.MAPPERS, f"{name} is in the run order with no mapper"
        table, mapper = L.MAPPERS[name]
        assert table and callable(mapper)


# ── the retry must not insert the same rows twice ──────────────────────────────────

class StubTransport:
    """Stands in for ServiceNow. Records what it was asked to insert, and can be told
    to fail AFTER committing, which is the case a blind retry gets wrong."""

    def __init__(self, fail_after_commit: int = 0, phantom_field: bool = False):
        self.inserted: list[str] = []
        self.fail_after_commit = fail_after_commit
        self.calls = 0
        # ⛔ MODELS THE REAL SERVICENOW BEHAVIOUR. Two tables in this project have no
        # correlation_id column, and a filter on a column that does not exist is IGNORED
        # rather than rejected, so it matches every row in the table. With this set, the
        # stub answers the probe the way a phantom field really does.
        self.phantom_field = phantom_field

    def __call__(self, target, path, body=None, method="GET", timeout=300):
        if path == L.BULK_PATH:
            self.calls += 1
            numbers = [r.get("correlation_id") for r in body["rows"]]
            self.inserted.extend(n for n in numbers if n)
            if self.fail_after_commit > 0:
                self.fail_after_commit -= 1
                # Committed, then the gateway gave up. Indistinguishable from a failure
                # before the commit unless you ask what landed.
                raise TimeoutError("gateway timed out after the insert committed")
            return {"result": {"inserted": len(body["rows"])}}
        # The probe: an impossible value. A real field matches nothing. A phantom field
        # ignores the filter and reports the whole table.
        if "/api/now/stats/" in path and "cannot-exist" in path:
            n = 40709 if self.phantom_field else 0
            return {"result": {"stats": {"count": str(n)}}}
        # A read of correlation_id, which is how the retry learns what already landed.
        if "correlation_idIN" in path:
            wanted = path.split("correlation_idIN")[1].split("&")[0].split("%2C")
            have = [{"correlation_id": n} for n in self.inserted if n in wanted]
            return {"result": have}
        return {"result": []}


def test_a_retry_after_a_commit_does_not_insert_the_rows_again(monkeypatch, sys_ids):
    """The defect: three attempts, 150 rows inserted, 50 unique."""
    stub = StubTransport(fail_after_commit=2)
    monkeypatch.setattr(L, "call", stub)
    monkeypatch.setattr(L.time, "sleep", lambda *_: None)

    rows = [{
        "number": f"INC200{i:04d}", "short_description": "x", "description": "y",
        "category": "capacity", "priority": 3,
        "opened_at": "2026-03-01T09:00:00+00:00", "resolved_at": None,
        "ci_key": None,
    } for i in range(50)]

    state: dict = {}
    L.load_fast(object(), "incidents", rows, 0, batch=50, workers=1,
                state=state, ci_sys_id=sys_ids)

    # ⛔ ONE INSERT CALL IS THE CORRECT ANSWER, and a first version of this test asserted
    # three because it assumed the retry would resend. The point is the opposite: the
    # first call committed and then timed out, the retry asked what had landed, found all
    # fifty, and declined to send anything again.
    assert stub.calls == 1, f"the batch was sent {stub.calls} times"
    assert len(stub.inserted) == len(set(stub.inserted)), (
        f"{len(stub.inserted)} rows were sent for {len(set(stub.inserted))} distinct "
        "tickets, so the retry replayed rows the server had already committed"
    )
    assert set(stub.inserted) == {r["number"] for r in rows}, "not every ticket landed"
    assert state["incidents"] == 50, "progress must record the rows that actually landed"


def test_a_shortfall_stops_the_run_rather_than_recording_false_progress(monkeypatch):
    """A run that inserted 19 of 200 once recorded 200, and a restart skipped the 181
    that never arrived."""
    class ShortStub(StubTransport):
        def __call__(self, target, path, body=None, method="GET", timeout=300):
            if path == L.BULK_PATH:
                return {"result": {"inserted": 1}}
            # Everything else, including the correlation_id probe the loader now runs
            # BEFORE sending, is answered by the base stub.
            return super().__call__(target, path, body, method, timeout)

    monkeypatch.setattr(L, "call", ShortStub())
    rows = [{"number": f"INC300{i:04d}", "short_description": "x", "description": "y",
             "category": "capacity", "priority": 3,
             "opened_at": "2026-03-01T09:00:00+00:00", "resolved_at": None,
             "ci_key": None} for i in range(10)]
    with pytest.raises(SystemExit) as caught:
        L.load_fast(object(), "incidents", rows, 0, batch=10, workers=1, state={})
    assert "STOPPED" in str(caught.value)


def test_progress_is_written_so_a_restart_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "STATE", tmp_path / "progress.json")
    L.save_state({"incidents": 1234})
    assert L.load_state() == {"incidents": 1234}


def test_the_guard_is_what_prevents_the_duplicates(monkeypatch, sys_ids):
    """The contrast, so the previous test is not passing for some other reason.

    Blind the retry by making the lookup report that nothing landed, which is exactly
    the old behaviour, and the same run inserts every row three times.
    """
    stub = StubTransport(fail_after_commit=2)
    monkeypatch.setattr(L, "call", stub)
    monkeypatch.setattr(L.time, "sleep", lambda *_: None)
    monkeypatch.setattr(L, "already_there", lambda *_a, **_k: set())

    rows = [{"number": f"INC400{i:04d}", "short_description": "x", "description": "y",
             "category": "capacity", "priority": 3,
             "opened_at": "2026-03-01T09:00:00+00:00", "resolved_at": None,
             "ci_key": None} for i in range(50)]
    L.load_fast(object(), "incidents", rows, 0, batch=50, workers=1,
                state={}, ci_sys_id=sys_ids)

    assert stub.calls == 3, "blinded, the retry should have resent twice"
    assert len(stub.inserted) == 150, (
        f"blinded, 150 rows should have been sent; got {len(stub.inserted)}")
    assert len(set(stub.inserted)) == 50, "for only 50 distinct tickets"


def test_choice_fields_only_send_values_the_instance_accepts(incident_row, sys_ids):
    """A choice field silently accepts anything. The close code in use was
    "Solved (Permanently)", which is not on the instance's list at all, and every record
    took it without complaint. The record exists and the field means nothing."""
    VALID_CATEGORY = {"software", "hardware", "network", "password_reset",
                      "inquiry", "database"}
    VALID_CLOSE = {"Resolved by request", "No resolution provided", "Solution provided",
                   "Resolved by caller", "Workaround provided", "Duplicate",
                   "Resolved by change", "User error", "Known error",
                   "Resolved by problem"}
    for family in L.CATEGORY_FOR_FAMILY:
        out = L.to_incident({**incident_row, "category": family}, sys_ids)
        assert out["category"] in VALID_CATEGORY, (
            f"{family} maps to {out['category']!r}, which is not a choice on the instance")
        assert out["close_code"] in VALID_CLOSE


def test_the_loader_stops_when_its_idempotency_field_does_not_exist(monkeypatch, sys_ids):
    """⛔ A GUARD THAT ALWAYS SAYS "GO AHEAD" IS WORSE THAN NO GUARD.

    `cmdb_rel_ci` and `kb_knowledge` have no correlation_id column. ServiceNow does not
    reject a query on a column that does not exist, it ignores the filter and returns the
    whole table. So `already_there` read a field that was absent from every row, found
    nothing, and reported that none of the batch was loaded yet. On a retry after a
    committed insert that answer causes exactly the duplication the function exists to
    prevent, while the loader prints that it checked.

    It must stop instead, and name the table.
    """
    stub = StubTransport(phantom_field=True)
    monkeypatch.setattr(L, "call", stub)
    with pytest.raises(SystemExit, match="no usable correlation_id"):
        L.already_there(object(), "kb_knowledge", ["KB9000", "KB9001"])


def test_a_real_field_passes_the_probe(monkeypatch, sys_ids):
    """The same probe must not fire on a table where the column is genuine, or every
    phase stops."""
    stub = StubTransport(phantom_field=False)
    monkeypatch.setattr(L, "call", stub)
    assert L.already_there(object(), "incident", ["INC2000000"]) == set()


def test_running_the_loader_twice_does_not_insert_the_rows_twice(monkeypatch, sys_ids):
    """⛔ SAFE TO RESTART HAS TO MEAN SAFE TO RUN AGAIN, and it did not.

    The correlation_id check lived only in the retry path, so it fired after a gateway
    timeout and at no other time. Running the loader again over rows that had already
    landed raised no exception, never reached the retry, and inserted every one of them a
    second time.

    This was found the expensive way: a three row test run against a live instance that
    already held all 900 problems produced three duplicates. The loader printed success.
    """
    stub = StubTransport()
    monkeypatch.setattr(L, "call", stub)
    monkeypatch.setattr(L.time, "sleep", lambda *_: None)

    rows = [{
        "number": f"INC400{i:04d}", "short_description": "x", "description": "y",
        "category": "capacity", "priority": 3,
        "opened_at": "2026-03-01T09:00:00+00:00", "resolved_at": None, "ci_key": None,
    } for i in range(20)]

    L.load_fast(object(), "incidents", rows, 0, batch=10, workers=1, state={})
    after_first = list(stub.inserted)
    assert len(after_first) == 20, f"first run inserted {len(after_first)}"

    # The same call again, exactly as a person re-running the command would make it.
    L.load_fast(object(), "incidents", rows, 0, batch=10, workers=1, state={})

    assert stub.inserted == after_first, (
        f"the second run inserted {len(stub.inserted) - len(after_first)} rows that were "
        f"already there. {len(set(stub.inserted))} unique numbers across "
        f"{len(stub.inserted)} inserts.")
