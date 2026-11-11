"""Uploaded profile pictures are checked once by an AI (Mistral first, Groq
if Mistral can't): no inappropriate stuff, no faces, no emoji pictures.
Unchecked pictures are never allowed. Never saved on the server."""
import json as _json

import pytest

from conftest import make_account

PIC = "data:image/jpeg;base64," + "A" * 400


class _R:
    def __init__(self, status, content):
        self.status_code = status
        self._c = content
        self.text = ""

    def json(self):
        return {"choices": [{"message": {"content": self._c}}]}


@pytest.fixture()
def ai(app_module, monkeypatch):
    calls = []
    replies = {"mistral": (200, '{"ok": 1, "reason": "Cute dog!"}'), "groq": (200, '{"ok": 1, "reason": "ok"}')}
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")
    app_module._backup_state.update(active={}, rest_until={}, answers={}, recent={})

    def fake_post(url, headers=None, json=None, timeout=None, **k):
        who = "mistral" if "mistral.ai" in url else "groq"
        calls.append((who, json))
        # while Mistral checks, the balancer must see it as busy
        if who == "mistral":
            assert app_module.provider_count("Mistral") == 1
        return _R(*replies[who])
    monkeypatch.setattr(app_module.requests, "post", fake_post)
    return calls, replies


def _check(client, image=PIC):
    return client.post("/api/check-avatar", json={"image": image})


def test_ok_picture_passes(client, ai):
    calls, _ = ai
    make_account(client, "Pic Kid", "9", "password1")
    r = _check(client)
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert calls[0][0] == "mistral"
    assert "EMOJI" in calls[0][1]["messages"][0]["content"]


def test_bad_picture_is_blocked(client, ai):
    calls, replies = ai
    replies["mistral"] = (200, '{"ok": 0, "reason": "Please don\'t use a photo of a face."}')
    make_account(client, "Pic Kid", "9", "password1")
    r = _check(client).get_json()
    assert r["ok"] is False and "face" in r["reason"]


def test_mistral_down_uses_groq(client, ai):
    calls, replies = ai
    replies["mistral"] = (500, "")
    make_account(client, "Pic Kid", "9", "password1")
    assert _check(client).get_json()["ok"] is True
    assert [c[0] for c in calls][:2] == ["mistral", "groq"]


def test_nobody_can_check_means_blocked(client, ai):
    calls, replies = ai
    replies["mistral"] = (500, "")
    replies["groq"] = (500, "")
    make_account(client, "Pic Kid", "9", "password1")
    r = _check(client)
    assert r.status_code == 503 and r.get_json()["ok"] is False


def test_garbled_answer_is_not_ok(client, ai):
    calls, replies = ai
    replies["mistral"] = (200, "sure!")
    replies["groq"] = (200, "looks fine")
    make_account(client, "Pic Kid", "9", "password1")
    assert _check(client).get_json()["ok"] is False


def test_needs_email_account(client, ai):
    assert _check(client).status_code == 403
    make_account(client, "Plain Kid", "9", "password1", recovery_email="")
    assert _check(client).status_code == 403


def test_bad_or_huge_images_refused(client, ai):
    make_account(client, "Pic Kid", "9", "password1")
    assert _check(client, "not a picture").status_code == 400
    assert _check(client, "data:image/jpeg;base64," + "A" * 300_000).status_code == 400


def test_mistral_count_goes_back_down_and_balancer_knows(app_module, client, ai):
    make_account(client, "Pic Kid", "9", "password1")
    _check(client)
    assert app_module.provider_count("Mistral") == 0
    assert "profile pictures" in app_module.BALANCER_PROMPT
