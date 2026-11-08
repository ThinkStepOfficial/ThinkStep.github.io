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
    make_account(client, "Full Kid", "9", "password1")  # photos/voices need an email account
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

def test_13_plus_signup_without_email_works_but_is_basic(client):
    res = make_account(client, "No Email Kid", "9", "password1", recovery_email="")
    assert res.status_code == 200
    me = res.get_json()
    assert me["needsEmail"] is True and me["fullFeatures"] is False
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
    assert not out.startswith(app_module.NOTICE)  # basic questions still work without an email
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


# ---------- parent dashboard (replaced the progress emails) ----------

def test_approving_gives_the_parent_a_dashboard_link_and_one_email(app_module, client, monkeypatch):
    sent = []
    monkeypatch.setattr(app_module, "send_email", lambda to, subj, body: sent.append((to, subj, body)) or True)
    make_account(client, "Sam Small", "3", "password1", is13=False, parent_email="dad@example.com")
    key = "sam small"
    tokens = app_module._load_consent_tokens()
    token = next(t for t, v in tokens.items() if v["user_key"] == key)
    page = client.post(f"/api/consent/{token}").get_data(as_text=True)
    assert "account approved" in page.lower()
    report_token = app_module._load_users()[key]["report_token"]
    assert f"/progress/{report_token}" in page
    link_emails = [m for m in sent if m[0] == "dad@example.com" and "progress" in m[1].lower()]
    assert len(link_emails) == 1 and f"/progress/{report_token}" in link_emails[0][2]


def test_dashboard_shows_counts_only(app_module, client):
    make_account(client, "Rae", "3", "password1", is13=False, parent_email="rae.parent@example.com")
    grant_consent(app_module, client, "Rae")
    app_module.record_activity("rae", questions=7, photos=1, checks=1)
    app_module.record_activity("rae", quiz={"total": 5, "right": 4})
    app_module.record_activity("rae", badge="quiz")
    token = app_module._load_users()["rae"]["report_token"]
    page = client.get(f"/progress/{token}").get_data(as_text=True)
    assert "Rae&#x27;s progress" in page or "Rae's progress" in page
    assert ">7<" in page and "4/5" in page and "Quiz Whiz" in page and "work checks" in page
    assert 'name="robots" content="noindex' in page
    assert "link isn" in client.get("/progress/not-a-real-token").get_data(as_text=True)


def test_old_email_links_open_the_dashboard(client):
    res = client.get("/api/report-settings/abc123")
    assert res.status_code == 302 and res.headers["Location"].endswith("/progress/abc123")


def test_no_more_scheduled_progress_emails(app_module):
    assert not hasattr(app_module, "send_due_reports")
    assert not hasattr(app_module, "start_report_thread")


def test_13_plus_students_can_open_their_own_progress_page(app_module, client):
    make_account(client, "Teen Tia", "9", "password1")
    assert client.get("/api/me").get_json()["hasProgressPage"] is True
    url = client.get("/api/my-progress").get_json()["url"]
    assert client.get(url).status_code == 200


def test_kids_under_13_get_no_progress_link_their_parent_has_it(app_module, client):
    make_account(client, "Kit", "2", "password1", is13=False, parent_email="p@example.com")
    grant_consent(app_module, client, "Kit")
    client.post("/api/login", json={"name": "Kit", "password": "password1"})
    assert client.get("/api/me").get_json()["hasProgressPage"] is False
    assert client.get("/api/my-progress").status_code == 403


# ---------- step checker + explain simple ----------

def test_check_my_work_finds_the_first_wrong_step(app_module, client, monkeypatch):
    make_account(client, "Full Kid", "9", "password1")  # photos/voices need an email account
    reading = {"appropriate": True, "homework": True, "notes": "2/3 - 1/6",
               "problems": [{"label": "Problem 3", "box": [70, 160, 910, 330]}], "focus": 0,
               "steps": [{"text": "2/3 = 4/6", "box": [100, 200, 600, 240]},
                         {"text": "4/6 - 1/6 = 3/0", "box": [100, 250, 600, 290]},
                         {"text": "", "box": [1, 2, 3]}],
               "first_error": 1, "error_note": "subtracted the bottom numbers"}
    seen = {"vision": None, "tutor": None}

    def fake_post(url, headers=None, json=None, stream=None, timeout=None):
        msgs = (json or {}).get("messages", [])
        if any(isinstance(m.get("content"), list) for m in msgs):
            seen["vision"] = json
            r = _Resp(); r.json = lambda: {"choices": [{"message": {"content": __import__("json").dumps(reading)}}]}
            return r
        if not stream:
            return _Resp()
        seen["tutor"] = json
        line = "data: " + __import__("json").dumps({"choices": [{"delta": {"content": "Nice!"}, "finish_reason": "stop"}]})
        return _Resp(lines=[line, "data: [DONE]"])

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")
    img = "data:image/jpeg;base64," + __import__("base64").b64encode(b"\xff\xd8fake").decode()
    out = client.post("/api/chat", json={"messages": [{"role": "user", "content": "check my work"}],
                                         "image": img, "imageSize": [1000, 1300], "checkWork": True}).get_data(as_text=True)
    assert "first_error" in seen["vision"]["messages"][0]["content"]
    data = json.loads(out.split(app_module.PHOTO_MARK)[1])
    assert data["check"] is True and data["firstError"] == 1 and len(data["steps"]) == 2
    system = seen["tutor"]["messages"][0]["content"]
    assert "CHECK THEIR WORK" in system and "Step 2" in system and "subtracted the bottom" in system
    assert "error_note" not in data, "the private note never goes to the page"


def test_explain_simple_tells_the_tutor_to_simplify(app_module, client, monkeypatch):
    make_account(client, "Full Kid", "9", "password1")  # photos/voices need an email account
    seen = _fake_tutor(app_module, monkeypatch)
    client.post("/api/chat", json={"messages": [{"role": "user", "content": "Explain that more simply"}],
                                   "profile": {"simple": True}}).get_data()
    assert "EXPLAIN SIMPLE" in seen["tutor"]["messages"][0]["content"]
    client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}],
                                   "profile": {"simple": "yes"}}).get_data()
    assert "EXPLAIN SIMPLE" not in seen["tutor"]["messages"][0]["content"]


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
    make_account(client, "Full Kid", "9", "password1")  # photos/voices need an email account
    calls = _fake_voice(app_module, monkeypatch)
    for _ in range(3):
        res = client.post("/api/speak", json={"text": "Nice work! What comes next?", "voice": "hannah"})
        assert res.status_code == 200 and res.mimetype == "audio/wav"
    assert len(calls) == 1, "the same words are only made once"
    assert calls[0]["model"] == "canopylabs/orpheus-v1-english" and calls[0]["voice"] == "hannah"


@pytest.mark.parametrize("body", [{"text": "hi", "voice": "robot"}, {"text": "", "voice": "troy"},
                                  {"text": "x" * 201, "voice": "troy"}, {"text": 5, "voice": "troy"}])
def test_speak_rejects_bad_requests(app_module, client, monkeypatch, body):
    make_account(client, "Full Kid", "9", "password1")  # photos/voices need an email account
    _fake_voice(app_module, monkeypatch)
    assert client.post("/api/speak", json=body).status_code == 400


def test_speak_needs_an_email_account(app_module, client, monkeypatch):
    _fake_voice(app_module, monkeypatch)
    res = client.post("/api/speak", json={"text": "hi", "voice": "troy"})  # guest
    assert res.status_code == 403 and res.get_json()["reason"] == "needs_email"
    make_account(client, "Plain Kid", "9", "password1", recovery_email="")
    res = client.post("/api/speak", json={"text": "hi", "voice": "troy"})
    assert res.status_code == 403 and res.get_json()["fallback"] is False


def test_when_groq_says_enough_the_page_is_told_to_use_its_own_voice(app_module, client, monkeypatch):
    make_account(client, "Full Kid", "9", "password1")  # photos/voices need an email account
    calls = _fake_voice(app_module, monkeypatch, status=429,
                        headers={"x-ratelimit-reset-requests": "2h3m"})
    res = client.post("/api/speak", json={"text": "one", "voice": "troy"})
    assert res.status_code == 503 and res.get_json() == {"fallback": True, "reason": "daily_limit"}
    client.post("/api/speak", json={"text": "two", "voice": "troy"})
    assert len(calls) == 1, "stops asking Groq until the limit resets"
    assert app_module._parse_groq_reset("2h3m") == 7380


def test_voice_terms_problem_shows_up_for_the_admin(app_module, client, monkeypatch):
    make_account(client, "Full Kid", "9", "password1")  # photos/voices need an email account
    _fake_voice(app_module, monkeypatch, status=400,
                text='{"error":{"message":"The model requires terms acceptance"}}')
    res = client.post("/api/speak", json={"text": "hi", "voice": "troy"})
    assert res.get_json()["reason"] == "terms"
    assert any("orpheus" in w for w in app_module.setup_warnings())
    # once the terms are accepted and a reading works, the warning goes away
    app_module._tts_state["down_until"] = 0
    _fake_voice(app_module, monkeypatch)
    assert client.post("/api/speak", json={"text": "hello", "voice": "troy"}).status_code == 200
    assert not any("orpheus" in w for w in app_module.setup_warnings())


def test_chat_page_has_the_new_controls():
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")
    for needle in ('id="themeToggle"', 'id="badgeShelf"', 'id="voiceSelect"', 'id="autoRead"',
                   'id="emailOverlay"', 'id="quizPill"', ':root[data-theme="dark"]', "kokoro-js@"):
        assert needle in html, needle


# ---------- faster "who's online" ----------

def test_closing_the_tab_takes_you_off_the_online_list(app_module, client, admin_client):
    make_account(client, "Quick Quinn", "9", "password1")
    client.post("/api/heartbeat")
    names = [a["name"] for a in admin_client.get("/api/admin/online").get_json()["accounts"]]
    assert "Quick Quinn" in names
    roster = {a["name"]: a["online"] for a in admin_client.get("/api/admin/accounts").get_json()["accounts"]}
    assert roster["Quick Quinn"] is True
    client.post("/api/heartbeat?leaving=1")
    assert "Quick Quinn" not in [a["name"] for a in admin_client.get("/api/admin/online").get_json()["accounts"]]
    roster = {a["name"]: a["online"] for a in admin_client.get("/api/admin/accounts").get_json()["accounts"]}
    assert roster["Quick Quinn"] is False


def test_last_seen_is_not_saved_on_every_check_in(app_module, client, monkeypatch):
    make_account(client, "Saver Sam", "9", "password1")
    saves = []
    real = app_module._save_users
    monkeypatch.setattr(app_module, "_save_users", lambda u: saves.append(1) or real(u))
    for _ in range(5):
        client.post("/api/heartbeat")
    assert len(saves) <= 1
