"""The admin page shows how many different guests came in the last 7 days."""
from datetime import datetime, timedelta, timezone


def _days_ago(n):
    return (datetime.now(timezone.utc).date() - timedelta(days=n)).isoformat()


def test_guest_heartbeats_are_counted_once_per_day(app_module, client, admin_client):
    app_module._guest_seen_today.update(date=None, ids=set())
    for _ in range(5):
        client.post("/api/heartbeat")
    g = app_module.guest_visit_stats()
    assert g["today"] == 1 and g["last7Days"] == 1


def test_last_7_days_counts_different_guests(app_module):
    app_module.STORE.write("guest_days.json", {
        _days_ago(0): ["a", "b"], _days_ago(3): ["b", "c"], _days_ago(6): ["d"],
        _days_ago(7): ["old"], _days_ago(30): ["older"]})
    g = app_module.guest_visit_stats()
    assert g["last7Days"] == 4 and g["today"] == 2 and len(g["days"]) == 7


def test_only_8_days_are_kept(app_module):
    app_module.STORE.write("guest_days.json", {_days_ago(n): [f"g{n}"] for n in range(1, 12)})
    app_module._guest_seen_today.update(date=None, ids=set())
    app_module.note_guest_visit("new")
    assert len(app_module.STORE.read("guest_days.json", {})) == 8


def test_admin_stats_include_guests(app_module, admin_client):
    data = admin_client.get("/api/admin/stats").get_json()
    assert "last7Days" in data["guests"]


def test_logged_in_students_are_not_guests(app_module, client):
    from conftest import make_account
    app_module._guest_seen_today.update(date=None, ids=set())
    app_module.STORE.write("guest_days.json", {})
    make_account(client, "Real Kid", "9", "password1")
    client.post("/api/heartbeat")
    assert app_module.guest_visit_stats()["last7Days"] == 0


# ---------- from what time to what time ----------

def _reset_sessions(app_module):
    app_module._guest_sess.update(data=None, saved={})
    app_module.STORE.write("guest_sessions.json", {})


def test_heartbeats_make_one_session_with_start_and_end(app_module, monkeypatch):
    _reset_sessions(app_module)
    clock = [1_000_000.0]
    monkeypatch.setattr(app_module.time, "time", lambda: clock[0])
    for _ in range(6):                      # one minute of heartbeats
        app_module.note_guest_session("guestabc123")
        clock[0] += 10
    rows = app_module.guest_session_list()
    assert len(rows) == 1
    assert rows[0]["guest"] == "guesta"
    assert rows[0]["end"] - rows[0]["start"] == 50_000


def test_a_long_gap_starts_a_new_session(app_module, monkeypatch):
    _reset_sessions(app_module)
    clock = [2_000_000.0]
    monkeypatch.setattr(app_module.time, "time", lambda: clock[0])
    app_module.note_guest_session("g1")
    clock[0] += 600                          # came back 10 minutes later
    app_module.note_guest_session("g1")
    rows = app_module.guest_session_list()
    assert len(rows) == 2 and rows[0]["start"] > rows[1]["start"]   # newest first


def test_old_sessions_are_dropped_after_7_days(app_module, monkeypatch):
    _reset_sessions(app_module)
    clock = [3_000_000.0]
    monkeypatch.setattr(app_module.time, "time", lambda: clock[0])
    app_module.note_guest_session("old")
    clock[0] += 8 * 24 * 3600
    app_module.note_guest_session("new")
    assert [r["guest"] for r in app_module.guest_session_list()] == ["new"]
    assert "old" not in app_module.STORE.read("guest_sessions.json", {})


def test_admin_sees_guest_sessions(app_module, client, admin_client):
    _reset_sessions(app_module)
    app_module.note_guest_session("someguest")
    rows = admin_client.get("/api/admin/stats").get_json()["guestSessions"]
    assert rows and rows[0]["online"] is True and "start" in rows[0] and "end" in rows[0]


def test_real_heartbeat_records_a_guest_session(app_module, client):
    _reset_sessions(app_module)
    client.post("/api/heartbeat")
    assert len(app_module.guest_session_list()) == 1
