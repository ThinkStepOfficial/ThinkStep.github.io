"""October 2026 features: short replies + quiz mode, every account has an
email, progress emails (parents choose how often), and the read-aloud
voice endpoint."""
import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from conftest import grant_consent, make_account


class _Resp:
    def __init__(self, status=200, content=b"", text="", lines=None, headers=None):
        self.status_code = status
        self.content = content
        self.text = text
        self._lines = lines or []
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return {"choices": [{"message": {"content": '{"violation": 0, "web": 0}'}}]}

    def iter_lines(self):
        for l in self._lines:
            yield l.encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_tutor(app_module, monkeypatch, reply="Good thinking!"):
    seen = {}

    def fake_post(url, headers=None, json=None, stream=None, timeout=None):
        if not stream:
            return _Resp()
        seen["tutor"] = json
        line = "data: " + __import__("json").dumps(
            {"choices": [{"delta": {"content": reply}, "finish_reason": "stop"}]})
        return _Resp(lines=[line, "data: [DONE]"])

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")
    return seen


# ---------- short replies + quiz ----------

def test_tutor_is_told_to_keep_replies_short_and_knows_the_quiz_tags(app_module):
    p = app_module.SYSTEM_PROMPT
    assert "KEEP EVERY REPLY SHORT" in p and "ONE small step" in p
    for tag in ("[[QUIZ_OFFER]]", "[[RIGHT]]", "[[WRONG]]", "[[QUIZ_DONE]]"):
        assert tag in p


def test_running_quiz_tells_the_tutor_which_question_is_next(app_module, client, monkeypatch):
    seen = _fake_tutor(app_module, monkeypatch)
    client.post("/api/chat", json={"messages": [{"role": "user", "content": "12"}],
                                   "profile": {"quiz": {"total": 5, "answered": 2, "right": 1}}}).get_data()
    system = seen["tutor"]["messages"][0]["content"]
    assert "QUIZ MODE: a 5-question quiz" in system and "question 3 of 5" in system


@pytest.mark.parametrize("quiz", [
    {"total": 50, "answered": 0, "right": 0}, {"total": 3, "answered": 4, "right": 0},
    {"total": 3, "answered": 1, "right": 2}, {"total": "3", "answered": 0, "right": 0},
    {"total": True, "answered": 0, "right": 0}, "quiz please", [1, 2],
])
def test_odd_quiz_info_is_ignored(app_module, client, monkeypatch, quiz):
    seen = _fake_tutor(app_module, monkeypatch)
    res = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}],
                                         "profile": {"quiz": quiz}})
    assert res.status_code == 200
    res.get_data()
    assert "QUIZ MODE: a" not in seen["tutor"]["messages"][0]["content"]


# ---------- every account needs an email ----------

def test_13_plus_signup_needs_an_email(client):
    res = make_account(client, "No Email Kid", "9", "password1", recovery_email="")
    assert res.status_code == 400 and "email" in res.get_json()["error"].lower()
    res = make_account(client, "Bad Email Kid", "9", "password1", recovery_email="nope")
    assert res.status_code == 400


def test_old_account_without_email_is_asked_for_one(app_module, client, monkeypatch):
    make_account(client, "Old Timer", "9", "password1")
    users = app_module._load_users()
    users["old timer"].pop("recovery_email")
    app_module._save_users(users)
    me = client.get("/api/me").get_json()
    assert me["needsEmail"] is True
    _fake_tutor(app_module, monkeypatch)
    out = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}).get_data(as_text=True)
    assert out.startswith(app_module.NOTICE) and "email" in out
    assert client.post("/api/set-email", json={"email": "bad"}).status_code == 400
    assert client.post("/api/set-email", json={"email": "old@example.com"}).status_code == 200
    assert client.get("/api/me").get_json()["needsEmail"] is False


def test_under_13_accounts_use_the_parent_email(app_module, client):
    make_account(client, "Little Lu", "4", "password1", is13=False, parent_email="mom@example.com")
    grant_consent(app_module, client, "Little Lu")
    client.post("/api/login", json={"name": "Little Lu", "password": "password1"})
    me = client.get("/api/me").get_json()
    assert me["needsEmail"] is False and me["under13"] is True
    assert client.post("/api/set-email", json={"email": "kid@example.com"}).status_code == 403


# ---------- parents choose how often ----------

def test_approving_shows_the_email_choice_and_saving_it_works(app_module, client):
    make_account(client, "Sam Small", "3", "password1", is13=False, parent_email="dad@example.com")
    key = "sam small"
    tokens = app_module._load_consent_tokens()
    token = next(t for t, v in tokens.items() if v["user_key"] == key)
    page = client.post(f"/api/consent/{token}").get_data(as_text=True)
    assert "account approved" in page.lower()
    assert "every week" in page.lower() and "no emails" in page.lower()
    report_token = app_module._load_users()[key]["report_token"]
    assert f"/api/report-settings/{report_token}" in page

    # just visiting the link (email scanners do that) changes nothing
    assert client.get(f"/api/report-settings/{report_token}").status_code == 200
    assert "report_freq" not in app_module._load_users()[key]

    res = client.post(f"/api/report-settings/{report_token}", data={"freq": "weekly"})
    assert "every week" in res.get_data(as_text=True)
    assert app_module._load_users()[key]["report_freq"] == "weekly"
    assert client.post(f"/api/report-settings/{report_token}", data={"freq": "hourly"}).status_code == 200
    assert app_module._load_users()[key]["report_freq"] == "weekly"
    assert "link isn" in client.get("/api/report-settings/not-a-real-token").get_data(as_text=True)


def test_kids_under_13_cant_change_their_parents_choice(app_module, client):
    make_account(client, "Kit", "2", "password1", is13=False, parent_email="p@example.com")
    grant_consent(app_module, client, "Kit")
    client.post("/api/login", json={"name": "Kit", "password": "password1"})
    assert client.post("/api/my-report-settings", json={"freq": "never"}).status_code == 403


def test_progress_email_is_sent_when_due_with_counts_only(app_module, client, monkeypatch):
    make_account(client, "Rae", "3", "password1", is13=False, parent_email="rae.parent@example.com")
    grant_consent(app_module, client, "Rae")
    users = app_module._load_users()
    app_module._set_report_freq(users["rae"], "weekly")
    app_module._save_users(users)
    app_module.record_activity("rae", questions=7, photos=1)
    app_module.record_activity("rae", quiz={"total": 5, "right": 4})
    app_module.record_activity("rae", badge="quiz")

    sent = []
    monkeypatch.setattr(app_module, "send_email", lambda to, subj, body: sent.append((to, subj, body)) or True)
    now = datetime.now(timezone.utc)
    assert app_module.send_due_reports(now) == 0               # not a week yet
    assert app_module.send_due_reports(now + timedelta(days=7, minutes=1)) == 1
    to, subj, body = sent[0]
    assert to == "rae.parent@example.com" and "Rae" in subj
    assert "Questions worked on: 7" in body and "4 of 5" in body and "Quiz Whiz" in body
    assert "/api/report-settings/" in body
    assert app_module.send_due_reports(now + timedelta(days=8)) == 0   # not again so soon


def test_no_emails_when_turned_off_or_not_approved(app_module, client, monkeypatch):
    make_account(client, "Pending Pat", "3", "password1", is13=False, parent_email="x@example.com")
    users = app_module._load_users()
    users["pending pat"]["report_freq"] = "daily"
    users["pending pat"]["report_freq_set_at"] = "2020-01-01T00:00:00+00:00"
    app_module._save_users(users)
    sent = []
    monkeypatch.setattr(app_module, "send_email", lambda *a: sent.append(a) or True)
    app_module.send_due_reports()
    assert sent == []


def test_13_plus_students_choose_their_own_progress_emails(app_module, client):
    make_account(client, "Teen Tia", "9", "password1")
    assert client.post("/api/my-report-settings", json={"freq": "monthly"}).get_json()["reportFreq"] == "monthly"
    assert client.get("/api/me").get_json()["reportFreq"] == "monthly"
    assert client.post("/api/my-report-settings", json={"freq": "yearly"}).status_code == 400


def test_activity_is_counted_and_deleted_with_the_account(app_module, client, monkeypatch):
    make_account(client, "Count Me", "9", "password1")
    _fake_tutor(app_module, monkeypatch)
    client.post("/api/chat", json={"messages": [{"role": "user", "content": "what is 2+2"}]}).get_data()
    assert client.post("/api/activity", json={"kind": "badge", "id": "first"}).status_code == 200
    assert client.post("/api/activity", json={"kind": "badge", "id": "hacker"}).status_code == 400
    assert client.post("/api/activity", json={"kind": "quiz", "total": 3, "right": 9}).status_code == 400
    s = app_module.activity_summary("count me", 1)
    assert s["questions"] == 1 and s["badges"] == ["🌱 First Step"]
    client.post("/api/delete-account", json={"password": "password1"})
    assert "count me" not in (app_module.STORE.read("activity.json", {}) or {})


# ---------- read aloud ----------

def _fake_voice(app_module, monkeypatch, status=200, headers=None, text=""):
    calls = []

    def fake_post(url, headers=None, json=None, stream=None, timeout=None):
        calls.append(json)
        return _Resp(status=status, content=b"RIFFfakewav" if status == 200 else b"",
                     text=text, headers=headers_ or {})
    headers_ = headers
    monkeypatch.setattr(app_module.requests, "post", fake_post)
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")
    return calls


def test_speak_returns_audio_and_remembers_it(app_module, client, monkeypatch):
    calls = _fake_voice(app_module, monkeypatch)
    for _ in range(3):
        res = client.post("/api/speak", json={"text": "Nice work! What comes next?", "voice": "hannah"})
        assert res.status_code == 200 and res.mimetype == "audio/wav"
    assert len(calls) == 1, "the same words are only made once"
    assert calls[0]["model"] == "canopylabs/orpheus-v1-english" and calls[0]["voice"] == "hannah"


@pytest.mark.parametrize("body", [{"text": "hi", "voice": "robot"}, {"text": "", "voice": "troy"},
                                  {"text": "x" * 201, "voice": "troy"}, {"text": 5, "voice": "troy"}])
def test_speak_rejects_bad_requests(app_module, client, monkeypatch, body):
    _fake_voice(app_module, monkeypatch)
    assert client.post("/api/speak", json=body).status_code == 400


def test_speak_needs_the_age_check_first(app_module, unverified_client, monkeypatch):
    _fake_voice(app_module, monkeypatch)
    res = unverified_client.post("/api/speak", json={"text": "hi", "voice": "troy"})
    assert res.status_code == 403 and res.get_json()["fallback"] is True


def test_when_groq_says_enough_the_page_is_told_to_use_its_own_voice(app_module, client, monkeypatch):
    calls = _fake_voice(app_module, monkeypatch, status=429,
                        headers={"x-ratelimit-reset-requests": "2h3m"})
    res = client.post("/api/speak", json={"text": "one", "voice": "troy"})
    assert res.status_code == 503 and res.get_json() == {"fallback": True, "reason": "daily_limit"}
    client.post("/api/speak", json={"text": "two", "voice": "troy"})
    assert len(calls) == 1, "stops asking Groq until the limit resets"
    assert app_module._parse_groq_reset("2h3m") == 7380


def test_voice_terms_problem_shows_up_for_the_admin(app_module, client, monkeypatch):
    _fake_voice(app_module, monkeypatch, status=400,
                text='{"error":{"message":"The model requires terms acceptance"}}')
    res = client.post("/api/speak", json={"text": "hi", "voice": "troy"})
    assert res.get_json()["reason"] == "terms"
    assert any("orpheus" in w for w in app_module.setup_warnings())


def test_chat_page_has_the_new_controls():
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")
    for needle in ('id="themeToggle"', 'id="badgeShelf"', 'id="voiceSelect"', 'id="autoRead"',
                   'id="emailOverlay"', 'id="quizPill"', ':root[data-theme="dark"]', "kokoro-js@"):
        assert needle in html, needle
