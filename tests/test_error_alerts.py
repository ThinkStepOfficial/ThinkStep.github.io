"""Error alerts: errors students actually see are logged (never their
question text), shown on the admin page, and 3 errors from one person in
an hour emails the operator — at most once per person every 20 minutes and at
most 20 emails a day."""
import pytest
import requests

from conftest import make_account


@pytest.fixture()
def sent(app_module, monkeypatch):
    """Captures every email instead of sending, and runs background work
    right away so the test can see it."""
    emails = []
    monkeypatch.setattr(app_module, "send_email",
                        lambda to, subject, body: emails.append((to, subject, body)) or True)
    monkeypatch.setattr(app_module, "_dispatch", lambda fn: fn())
    return emails


def _alerts(emails):
    return [e for e in emails if "hit 3 errors" in e[1]]


class _Resp:
    def __init__(self, status=200, lines=None):
        self.status_code = status
        self.text = ""
        self._lines = lines or []

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error", response=self)

    def iter_lines(self):
        for l in self._lines:
            yield l.encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ask(client, text="what is 7 x 8?"):
    return client.post("/api/chat", json={"messages": [{"role": "user", "content": text}]}).get_data(as_text=True)


def _groq_fails_with(app_module, monkeypatch, exc=None, status=None):
    def fake_post(url, headers=None, json=None, stream=None, timeout=None):
        if exc:
            raise exc
        return _Resp(status=status)
    monkeypatch.setattr(app_module.requests, "post", fake_post)
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")
    monkeypatch.setattr(app_module, "topic_gate_check", lambda h, k: {"blocked": False, "web": False})


# ---------- counting + email rules ----------

def test_two_errors_send_no_email(app_module, sent):
    for _ in range(2):
        app_module.note_error("guest:abc", "guest abc", "groq 429", "Groq said 429.")
    assert _alerts(sent) == []
    assert len(app_module._load_errors()) == 2


def test_three_errors_email_the_operator_once(app_module, sent):
    for _ in range(5):
        app_module.note_error("guest:abc", "guest abc", "groq 401", "Groq said 401.")
    alerts = _alerts(sent)
    assert len(alerts) == 1
    to, subject, body = alerts[0]
    assert to.lower() == "thinkstepv1.0@gmail.com"
    assert "a guest" in body and "API key" in body and "20 minutes" in body


def test_account_name_and_grade_are_in_the_email(app_module, client, sent, monkeypatch):
    make_account(client, "Error Ella", "9", "password1")
    _groq_fails_with(app_module, monkeypatch, exc=requests.exceptions.ReadTimeout("slow"))
    for _ in range(3):
        _ask(client)
    assert [e["who"] for e in app_module._load_errors()] == ["Error Ella"] * 3
    body = _alerts(sent)[0][2]
    assert "Error Ella (grade 9)" in body


def test_different_people_are_counted_separately(app_module, sent):
    for who in ("guest:a", "guest:b", "guest:a", "guest:b"):
        app_module.note_error(who, who, "groq timeout", "")
    assert _alerts(sent) == []
    app_module.note_error("guest:a", "guest:a", "groq timeout", "")
    assert len(_alerts(sent)) == 1


def test_another_alert_after_the_6_hour_cooldown(app_module, sent):
    for _ in range(3):
        app_module.note_error("guest:a", "a", "groq 429", "")
    # pretend the email went out 7 hours ago and the errors are old
    app_module._error_alerted["guest:a"] -= 7 * 3600
    for _ in range(3):
        app_module.note_error("guest:a", "a", "groq 429", "")
    assert len(_alerts(sent)) == 2


def test_daily_cap_on_alert_emails(app_module, sent):
    for i in range(app_module.ERROR_ALERTS_PER_DAY + 5):
        for _ in range(3):
            app_module.note_error(f"guest:{i}", f"g{i}", "groq 429", "")
    assert len(_alerts(sent)) == app_module.ERROR_ALERTS_PER_DAY


def test_log_keeps_only_the_newest_entries_and_short_details(app_module, sent):
    for i in range(app_module.ERRORS_KEEP + 10):
        app_module.note_error(f"guest:{i}", "g", "js", "x" * 500)
    log = app_module._load_errors()
    assert len(log) == app_module.ERRORS_KEEP
    assert all(len(e["detail"]) <= 200 for e in log)
    assert "errors.json" in app_module.BACKUP_FILES


# ---------- the chat route counts real errors ----------

def test_chat_connection_error_counts(app_module, client, sent, monkeypatch):
    _groq_fails_with(app_module, monkeypatch, exc=requests.exceptions.ConnectionError("down"))
    for _ in range(3):
        assert "Couldn't reach Groq" in _ask(client, "secret homework question")
    log = app_module._load_errors()
    assert [e["kind"] for e in log] == ["groq unreachable"] * 3
    assert "secret homework" not in str(log)
    assert len(_alerts(sent)) == 1
    assert "secret homework" not in _alerts(sent)[0][2]


def test_chat_429_and_401_are_named(app_module, client, sent, monkeypatch):
    _groq_fails_with(app_module, monkeypatch, status=429)
    _ask(client)
    _groq_fails_with(app_module, monkeypatch, status=401)
    _ask(client)
    assert [e["kind"] for e in app_module._load_errors()] == ["groq 429", "groq 401"]


def test_chat_without_api_key_counts(app_module, client, sent):
    _ask(client)  # the isolated test folder has no Groq key
    assert [e["kind"] for e in app_module._load_errors()] == ["no api key"]


def test_normal_notices_are_not_errors(app_module, client, unverified_client, sent, monkeypatch):
    _ask(unverified_client)  # age question, not an error
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")
    monkeypatch.setattr(app_module, "topic_gate_check", lambda h, k: {"blocked": True, "web": False})
    _ask(client, "tell me a joke")  # off-task refusal, not an error
    assert app_module._load_errors() == []


# ---------- crashes ----------

def test_server_crash_is_counted_and_friendly(app_module, sent):
    app_module.app.config["PROPAGATE_EXCEPTIONS"] = False

    @app_module.app.route("/api/test-boom")
    def boom():
        raise RuntimeError("kaboom")

    c = app_module.app.test_client()
    res = c.get("/api/test-boom")
    assert res.status_code == 500
    assert res.get_json() == {"error": "Something went wrong on our side."}
    log = app_module._load_errors()
    assert log[-1]["kind"] == "server crash" and "/api/test-boom" in log[-1]["detail"]


# ---------- /api/client-error ----------

def test_client_error_is_logged(app_module, client, sent):
    res = client.post("/api/client-error", json={"kind": "network", "detail": "y" * 999})
    assert res.status_code == 200 and res.get_json() == {"ok": True}
    e = app_module._load_errors()[-1]
    assert e["kind"] == "network" and len(e["detail"]) == 200


@pytest.mark.parametrize("body", [
    {"kind": "evil"}, {"kind": 5}, {}, [1, 2], "hello", None, {"kind": ["js"]},
    {"kind": "js", "detail": {"a": 1}},
])
def test_client_error_odd_input_never_crashes(client, sent, body):
    res = client.post("/api/client-error", json=body)
    assert res.status_code in (200, 400)


def test_client_error_broken_json_never_crashes(client, sent):
    res = client.post("/api/client-error", data="{not json", content_type="application/json")
    assert res.status_code == 400


def test_client_error_is_rate_limited(app_module, client, sent):
    codes = [client.post("/api/client-error", json={"kind": "js", "detail": str(i)}).status_code
             for i in range(25)]
    assert codes[:20] == [200] * 20 and codes[20:] == [429] * 5
    assert len(app_module._load_errors()) == 20


# ---------- admin page ----------

def test_admin_errors_needs_admin(client):
    assert client.get("/api/admin/errors").status_code == 401


def test_admin_errors_lists_newest_first_with_counts(app_module, admin_client, sent):
    app_module.note_error("guest:a", "guest a", "groq 429", "first")
    app_module.note_error("guest:b", "guest b", "js", "second")
    data = admin_client.get("/api/admin/errors").get_json()
    assert [e["detail"] for e in data["errors"]] == ["second", "first"]
    assert data["lastHour"] == 2 and data["last24h"] == 2
    assert admin_client.get("/api/admin/stats").get_json()["totalErrors24h"] == 2


def test_admin_page_has_errors_section():
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "static" / "admin.html").read_text(encoding="utf-8")
    assert "/api/admin/errors" in html and "errors (last 24h)" in html and "fetchErrors()" in html


def test_chat_page_reports_errors():
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")
    assert "function reportError" in html and "/api/client-error" in html
    assert "unhandledrejection" in html
