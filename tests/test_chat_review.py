"""A small Cloudflare model reviews each finished chat and flags anything
off-topic, inappropriate or unsafe for the admin. It counts as Cloudflare
load, so the balancer sees it."""
import json as _json

from conftest import TEST_ADMIN_PASSWORD
from test_smarter_models import _Reply, _ask, _fake_groq


class _CF:
    def __init__(self, content, status=200):
        self.status_code = status
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _keys(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")


def _admin(app_module):
    c = app_module.app.test_client()
    assert c.post("/api/admin/login", json={"password": TEST_ADMIN_PASSWORD}).status_code == 200
    return c


def test_flagged_chat_shows_in_admin(app_module, monkeypatch):
    _keys(monkeypatch)
    sent = []

    def fake(url, headers=None, json=None, timeout=None, **k):
        sent.append((url, json))
        return _CF('{"verdict": "inappropriate", "reason": "mean words"}')
    monkeypatch.setattr(app_module.requests, "post", fake)
    assert app_module.review_chat("guest:abc", "Guest abc123", "you are dumb", "Let's be kind!") == "inappropriate"
    url, body = sent[0]
    assert "accounts/acct/ai/v1/chat/completions" in url and body["model"] == "@cf/openai/gpt-oss-20b"
    st = _admin(app_module).get("/api/admin/errors").get_json()["chatReview"]
    assert st["ready"] and st["checks"] == 1
    assert st["recent"][0]["kind"] == "inappropriate" and st["recent"][0]["q"] == "you are dumb"


def test_ok_chats_are_not_flagged(app_module, monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setattr(app_module.requests, "post", lambda *a, **k: _CF('{"verdict": "ok", "reason": "math"}'))
    assert app_module.review_chat("guest:a", "Guest a", "what is 2+2", "What do you think?") == "ok"
    assert app_module.chat_review_status()["recent"] == []


def test_review_counts_as_cloudflare_load(app_module, monkeypatch):
    _keys(monkeypatch)
    seen = {}

    def fake(*a, **k):
        seen["busy"] = app_module.provider_count("Cloudflare")
        return _CF('{"verdict": "ok"}')
    monkeypatch.setattr(app_module.requests, "post", fake)
    app_module.review_chat("guest:a", "Guest a", "q", "a")
    assert seen["busy"] == 1 and app_module.provider_count("Cloudflare") == 0
    assert "Cloudflare also reviews" in app_module.BALANCER_PROMPT


def test_broken_cloudflare_never_breaks_anything(app_module, monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setattr(app_module.requests, "post", lambda *a, **k: _CF("", status=500))
    assert app_module.review_chat("guest:a", "Guest a", "q", "a") is None
    assert app_module.chat_review_status()["resting"]


def test_no_keys_means_no_review(app_module, monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    assert app_module.review_chat("guest:a", "Guest a", "q", "a") is None
    assert not app_module.chat_review_status()["ready"]


def test_blocked_off_topic_question_is_flagged_without_asking(app_module, client, monkeypatch):
    monkeypatch.setattr(app_module, "topic_gate_check", lambda h, k: {"blocked": True, "web": False})
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")
    client.post("/api/chat", json={"messages": [{"role": "user", "content": "play a game with me"}]}).get_data()
    flags = app_module.chat_review_status()["recent"]
    assert flags[0]["kind"] == "off_topic" and flags[0]["q"] == "play a game with me"


def test_finished_chat_gets_reviewed(app_module, client, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "review_chat_later", lambda *a: calls.append(a))
    _fake_groq(app_module, monkeypatch, [_Reply(content=["Nice question!"])])
    _ask(client, "why is the sky blue?")
    assert calls and calls[0][2] == "why is the sky blue?" and calls[0][3] == "Nice question!"
