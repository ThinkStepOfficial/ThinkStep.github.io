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
