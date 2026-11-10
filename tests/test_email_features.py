"""Accounts can be made without an email (13+), but then — like guests —
they only get the basic tutor. Photos, voices, quizzes, badges, explain
simple and the progress page need an account with an email."""
import json

import pytest
import requests

from conftest import make_account

PHOTO = "data:image/png;base64," + "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


class _Stream:
    status_code = 200
    text = ""
    def __init__(self, seen, body):
        seen.append(body)
    def raise_for_status(self): pass
    def iter_lines(self):
        yield ("data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]})).encode()
        yield b"data: [DONE]"
    def __enter__(self): return self
    def __exit__(self, *a): return False


@pytest.fixture()
def tutor(app_module, monkeypatch):
    seen = []
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")
    monkeypatch.setattr(app_module, "topic_gate_check", lambda h, k: {"blocked": False, "web": False})
    monkeypatch.setattr(app_module.requests, "post",
                        lambda url, headers=None, json=None, stream=None, timeout=None: _Stream(seen, json))
    return seen


def _chat(client, **extra):
    return client.post("/api/chat", json={"messages": [{"role": "user", "content": "what is 7 x 8"}], **extra}
                       ).get_data(as_text=True)


def test_signup_without_email_is_allowed_and_basic(client):
    r = make_account(client, "Plain Kid", "9", "password1", recovery_email="")
    assert r.status_code == 200
    me = client.get("/api/me").get_json()
    assert me["needsEmail"] is True and me["fullFeatures"] is False


def test_signup_with_email_gets_everything(client):
    make_account(client, "Mail Kid", "9", "password1")
    assert client.get("/api/me").get_json()["fullFeatures"] is True


def test_basic_tutor_still_works_without_email(client, tutor):
    make_account(client, "Plain Kid", "9", "password1", recovery_email="")
    assert "ok" in _chat(client)


@pytest.mark.parametrize("who", ["guest", "no-email"])
def test_photos_are_locked(client, tutor, who):
    if who == "no-email":
        make_account(client, "Plain Kid", "9", "password1", recovery_email="")
    out = _chat(client, image=PHOTO)
    assert "email" in out.lower() and tutor == []  # never reached the AI


@pytest.mark.parametrize("who", ["guest", "no-email"])
def test_quiz_and_explain_simple_are_ignored(client, tutor, who):
    if who == "no-email":
        make_account(client, "Plain Kid", "9", "password1", recovery_email="")
    _chat(client, profile={"simple": True, "quiz": {"total": 5, "answered": 1, "right": 1}})
    system = tutor[-1]["messages"][0]["content"]
    assert "the student pressed the Explain simple button" not in system
    assert "QUIZ MODE: a 5-question quiz" not in system


def test_voices_are_locked_for_no_email_account(client):
    make_account(client, "Plain Kid", "9", "password1", recovery_email="")
    r = client.post("/api/speak", json={"text": "hi", "voice": "troy"})
    assert r.status_code == 403 and r.get_json()["reason"] == "needs_email"


def test_progress_page_and_activity_are_locked(client):
    make_account(client, "Plain Kid", "9", "password1", recovery_email="")
    r = client.get("/api/my-progress")
    assert r.status_code == 403 and r.get_json()["needsEmail"] is True
    assert client.post("/api/activity", json={"kind": "badge", "id": "first"}).get_json()["ok"] is False


def test_adding_an_email_unlocks_everything(client, tutor):
    make_account(client, "Plain Kid", "9", "password1", recovery_email="")
    assert client.post("/api/set-email", json={"email": "plain@example.com"}).status_code == 200
    assert client.get("/api/me").get_json()["fullFeatures"] is True
    assert client.get("/api/my-progress").status_code == 200


def test_terms_and_privacy_explain_it(client):
    terms = client.get("/terms").get_data(as_text=True)
    assert "Features that need an email" in terms and "can't be recovered" in terms
    assert "only gets the basic tutor" in client.get("/privacy").get_data(as_text=True)


def test_an_existing_email_cant_be_swapped(client):
    make_account(client, "Mail Kid", "9", "password1")
    r = client.post("/api/set-email", json={"email": "thief@example.com"})
    assert r.status_code == 409
