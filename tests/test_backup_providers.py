"""Backup AI providers: Groq answers up to 5 students at once, then
OpenRouter up to 4, then Mistral up to 4 — only providers with a key set.
Students are never told they were switched. The admin page shows who is
on which provider."""
import json as _json

import pytest
import requests


class _Resp:
    def __init__(self, status=200, words=None):
        self.status_code = status
        self.text = ""
        self._words = words or []

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error", response=self)

    def iter_lines(self):
        for w in self._words:
            yield ("data: " + _json.dumps({"choices": [{"delta": {"content": w}}]})).encode()
        yield b"data: [DONE]"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture()
def setup(app_module, monkeypatch):
    for p in app_module.BACKUP_PROVIDERS:
        monkeypatch.delenv(p["key_env"], raising=False)
    app_module._backup_state.update(active={}, rest_until={}, answers={}, recent={})
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")
    monkeypatch.setattr(app_module, "topic_gate_check", lambda h, k: {"blocked": False, "web": False})
    calls = []
    behaviour = {"groq": 200, "backup": 200}

    def fake_post(url, headers=None, json=None, stream=None, timeout=None):
        calls.append(url)
        if "groq.com" in url:
            return _Resp(behaviour["groq"], ["groq ", "answer"])
        return _Resp(behaviour["backup"], ["backup ", "answer"])
    monkeypatch.setattr(app_module.requests, "post", fake_post)
    yield calls, behaviour
    app_module._backup_state.update(active={}, rest_until={}, answers={}, recent={})


def _fill(app_module, name, n):
    for i in range(n):
        app_module._provider_start(name, f"busy kid {i}")


def _ask(client):
    return client.post("/api/chat", json={"messages": [{"role": "user", "content": "what is 7 x 8?"}]}).get_data(as_text=True)


def test_quiet_groq_answers_normally(app_module, client, setup, monkeypatch):
    calls, _ = setup
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    assert "groq answer" in _ask(client)
    assert not any("mistral" in c for c in calls)
    assert app_module.groq_active_count() == 0  # counter goes back down


def test_crowded_groq_sends_student_to_backup(app_module, client, setup, monkeypatch):
    calls, _ = setup
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    _fill(app_module, "Groq", app_module.GROQ_LIMIT)
    assert "backup answer" in _ask(client)
    assert any("mistral.ai" in c for c in calls)
    assert not any("groq.com/openai/v1/chat" in c for c in calls)


def test_crowded_but_no_keys_still_uses_groq(app_module, client, setup):
    _fill(app_module, "Groq", app_module.GROQ_LIMIT)
    assert "groq answer" in _ask(client)


def test_order_is_groq_then_openrouter_then_mistral(app_module, client, setup, monkeypatch):
    calls, _ = setup
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    _fill(app_module, "Groq", 5)
    _ask(client)
    assert "openrouter.ai" in calls[-1]
    _fill(app_module, "OpenRouter", 4)
    _ask(client)
    assert "mistral.ai" in calls[-1]
    _fill(app_module, "Mistral", 4)
    assert "groq answer" in _ask(client)  # everyone full -> Groq takes it


def test_limits_are_5_4_4(app_module):
    lim = {p["name"]: p["limit"] for p in app_module.BACKUP_PROVIDERS}
    assert app_module.GROQ_LIMIT == 5 and lim["OpenRouter"] == 4 and lim["Mistral"] == 4


def test_switch_is_silent(app_module, client, setup, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    _fill(app_module, "Groq", 5)
    out = _ask(client)
    assert out.strip("\x00").strip() == "backup answer"
    for word in ("OpenRouter", "Mistral", "switch", "backup provider", "busy"):
        assert word.lower() not in out.lower().replace("backup answer", "")


def test_groq_429_everywhere_falls_to_backup(app_module, client, setup, monkeypatch):
    calls, behaviour = setup
    behaviour["groq"] = 429
    monkeypatch.setenv("GEMINI_API_KEY", "z")
    assert "backup answer" in _ask(client)
    assert any("generativelanguage" in c for c in calls)


def test_failing_backup_rests_and_groq_answers(app_module, client, setup, monkeypatch):
    calls, behaviour = setup
    behaviour["backup"] = 500
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    _fill(app_module, "Groq", app_module.GROQ_LIMIT)
    assert "groq answer" in _ask(client)
    assert app_module.backups_ready() == []  # Mistral is resting


def test_admin_sees_who_is_on_which_provider(app_module, client, setup, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    _fill(app_module, "Groq", 5)
    _ask(client)
    st = {p["name"]: p for p in app_module.backup_status()["providers"]}
    assert len(st["Groq"]["now"]) == 5 and st["Groq"]["limit"] == 5
    assert st["OpenRouter"]["ready"] and not st["Mistral"]["ready"]
    assert st["OpenRouter"]["answers"] == 1 and st["OpenRouter"]["now"] == []
    assert len(st["OpenRouter"]["recent"]) == 1


def test_counters_go_back_down(app_module, client, setup):
    _ask(client)
    assert app_module.groq_active_count() == 0
    st = {p["name"]: p for p in app_module.backup_status()["providers"]}
    assert st["Groq"]["answers"] == 1 and len(st["Groq"]["recent"]) == 1
