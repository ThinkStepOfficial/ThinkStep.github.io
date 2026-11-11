"""AI providers: no limit on students. Questions go to Groq, OpenRouter or
Mistral (only providers with a key set), and the Cloudflare balancer spreads
them evenly — or, without it, the least busy AI answers. Students are never
told they were switched. The admin page shows who is on which AI."""
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


def _reset(app_module):
    app_module._backup_state.update(active={}, rest_until={}, answers={}, recent={})
    app_module._balance.update(asks={}, log=[], cf_checks=0, cf_rest_until=0.0)


@pytest.fixture()
def setup(app_module, monkeypatch):
    for k in [p["key_env"] for p in app_module.BACKUP_PROVIDERS] + ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"]:
        monkeypatch.delenv(k, raising=False)
    _reset(app_module)
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")
    monkeypatch.setattr(app_module, "topic_gate_check", lambda h, k: {"blocked": False, "web": False})
    calls = []
    behaviour = {"groq": 200, "backup": 200, "cloudflare": None}

    class CF:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": behaviour["cloudflare"]}}]}

    def fake_post(url, headers=None, json=None, stream=None, timeout=None):
        calls.append((url, json))
        if "cloudflare" in url:
            if behaviour["cloudflare"] is None:
                r = CF(); r.status_code = 500; return r
            return CF()
        if "groq.com" in url:
            return _Resp(behaviour["groq"], ["groq ", "answer"])
        return _Resp(behaviour["backup"], ["backup ", "answer"])
    monkeypatch.setattr(app_module.requests, "post", fake_post)
    yield calls, behaviour
    _reset(app_module)


def _urls(calls):
    return [u for u, _ in calls]


def _fill(app_module, name, n):
    for i in range(n):
        app_module._provider_start(name, f"busy kid {i}")


def _others(app_module, n=1):
    for i in range(n):
        app_module.note_question(f"other kid {i}")


def _ask(client):
    return client.post("/api/chat", json={"messages": [{"role": "user", "content": "what is 7 x 8?"}]}).get_data(as_text=True)


def _cloudflare_on(monkeypatch, behaviour, reply):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    behaviour["cloudflare"] = reply


# ---------- no limits, even spread ----------

def test_quiet_groq_answers_normally(app_module, client, setup, monkeypatch):
    calls, _ = setup
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    assert "groq answer" in _ask(client)
    assert app_module.groq_active_count() == 0  # counter goes back down


def test_no_backup_keys_means_groq_always(app_module, client, setup):
    _fill(app_module, "Groq", 50)
    assert "groq answer" in _ask(client)


def test_busy_groq_sends_student_to_least_busy_ai(app_module, client, setup, monkeypatch):
    calls, _ = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    _fill(app_module, "Groq", 3)
    _fill(app_module, "OpenRouter", 2)
    assert "backup answer" in _ask(client)
    assert "mistral.ai" in _urls(calls)[-1]


def test_there_is_no_limit_on_students(app_module, client, setup, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    for name in ("Groq", "OpenRouter", "Mistral"):
        _fill(app_module, name, 40)
    assert "answer" in _ask(client)  # 121st student still gets an answer
    assert not hasattr(app_module, "GROQ_LIMIT")
    assert all("limit" not in p for p in app_module.BACKUP_PROVIDERS)


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
    assert "generativelanguage" in _urls(calls)[-1]


def test_failing_backup_rests_and_groq_answers(app_module, client, setup, monkeypatch):
    calls, behaviour = setup
    behaviour["backup"] = 500
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    _fill(app_module, "Groq", 5)
    assert "groq answer" in _ask(client)
    assert app_module.backups_ready() == []  # Mistral is resting


def test_admin_sees_who_is_on_which_provider(app_module, client, setup, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    _fill(app_module, "Groq", 5)
    _ask(client)
    st = {p["name"]: p for p in app_module.backup_status()["providers"]}
    assert len(st["Groq"]["now"]) == 5 and "limit" not in st["Groq"]
    assert st["OpenRouter"]["ready"] and not st["Mistral"]["ready"]
    assert st["OpenRouter"]["answers"] == 1 and st["OpenRouter"]["now"] == []
    assert len(st["OpenRouter"]["recent"]) == 1


def test_counters_go_back_down(app_module, client, setup):
    _ask(client)
    assert app_module.groq_active_count() == 0
    st = {p["name"]: p for p in app_module.backup_status()["providers"]}
    assert st["Groq"]["answers"] == 1 and len(st["Groq"]["recent"]) == 1


# ---------- the Cloudflare balancer decides ----------

def test_cloudflare_picks_the_ai(app_module, client, setup, monkeypatch):
    calls, behaviour = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    _cloudflare_on(monkeypatch, behaviour, '{"send_to": "Mistral", "reason": "spread the load"}')
    _others(app_module)
    assert "backup answer" in _ask(client)
    assert "mistral.ai" in _urls(calls)[-1]  # where the model said, even though nobody was busy
    log = app_module.balancer_status()["recent"]
    assert log[0]["sendTo"] == "Mistral" and log[0]["reason"] == "spread the load"


def test_cloudflare_can_keep_a_student_on_groq(app_module, client, setup, monkeypatch):
    calls, behaviour = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    _cloudflare_on(monkeypatch, behaviour, '{"send_to": "Groq", "reason": "even"}')
    _fill(app_module, "Groq", 9)
    _others(app_module)
    assert "groq answer" in _ask(client)


def test_unusable_cloudflare_answer_means_least_busy(app_module, client, setup, monkeypatch):
    calls, behaviour = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    _cloudflare_on(monkeypatch, behaviour, '{"send_to": "SomethingElse"}')
    _fill(app_module, "Groq", 2)
    _others(app_module)
    assert "backup answer" in _ask(client)


def test_cloudflare_only_sees_numbers(app_module, setup, monkeypatch):
    calls, behaviour = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    _cloudflare_on(monkeypatch, behaviour, '{"send_to": "OpenRouter"}')
    for _ in range(6):
        app_module.note_question("Secret Name (grade 5)")
    app_module.note_question("Other Kid (grade 3)")
    assert app_module.choose_provider("Secret Name (grade 5)") == "OpenRouter"
    url, body = calls[0]
    assert "api.cloudflare.com" in url and body["model"] == "@cf/openai/gpt-oss-20b"
    dumped = _json.dumps(body)
    assert "Secret" not in dumped and "Other Kid" not in dumped and "slots" not in dumped


def test_one_student_alone_skips_cloudflare(app_module, setup, monkeypatch):
    calls, behaviour = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    _cloudflare_on(monkeypatch, behaviour, '{"send_to": "OpenRouter"}')
    app_module.note_question("only kid")
    assert app_module.choose_provider("only kid") is None and calls == []


def test_cloudflare_broken_means_least_busy(app_module, client, setup, monkeypatch):
    calls, behaviour = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")  # behaviour["cloudflare"] None -> 500
    _fill(app_module, "Groq", 4)
    _others(app_module)
    assert "backup answer" in _ask(client)
    assert app_module.balancer_status()["resting"]
