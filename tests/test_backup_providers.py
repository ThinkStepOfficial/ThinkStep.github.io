"""AI providers: no limit on students. Questions go to Groq, OpenRouter or
Mistral or Cloudflare (only providers with a key set), and the balancer
(Groq's small model) spreads them evenly — or, without it, the least busy AI answers. Students are never
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
    app_module._balance.update(asks={}, log=[], checks=0, rest_until=0.0)


@pytest.fixture()
def setup(app_module, monkeypatch):
    for k in [p["key_env"] for p in app_module.BACKUP_PROVIDERS] + ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"]:
        monkeypatch.delenv(k, raising=False)
    _reset(app_module)
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")
    monkeypatch.setattr(app_module, "topic_gate_check", lambda h, k: {"blocked": False, "web": False})
    calls = []
    behaviour = {"groq": 200, "backup": 200, "brain": None}

    class Brain:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": behaviour["brain"]}}]}

    def fake_post(url, headers=None, json=None, stream=None, timeout=None):
        calls.append((url, json))
        if "groq.com" in url and (json or {}).get("model") == app_module.BALANCER_MODEL:
            if behaviour["brain"] is None:
                r = Brain(); r.status_code = 500; return r
            return Brain()
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


def _brain(behaviour, reply):
    behaviour["brain"] = reply


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
    monkeypatch.setenv("MISTRAL_API_KEY", "z")
    assert "backup answer" in _ask(client)
    assert "mistral.ai" in _urls(calls)[-1]


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


# ---------- the balancer (Groq's small model) decides ----------

def test_balancer_picks_the_ai(app_module, client, setup, monkeypatch):
    calls, behaviour = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    _brain(behaviour, '{"send_to": "Mistral", "reason": "spread the load"}')
    _others(app_module)
    assert "backup answer" in _ask(client)
    assert "mistral.ai" in _urls(calls)[-1]  # where the model said, even though nobody was busy
    st = app_module.balancer_status()
    assert st["recent"][0]["sendTo"] == "Mistral" and st["recent"][0]["reason"] == "spread the load"
    assert st["checks"] == 1 and st["provider"] == "Groq" and st["model"] == "llama-3.1-8b-instant"


def test_cloudflare_is_not_the_balancer_anymore(app_module, setup, monkeypatch):
    calls, behaviour = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    _brain(behaviour, '{"send_to": "OpenRouter"}')
    _others(app_module)
    app_module.note_question("me")
    assert app_module.choose_provider("me") == "OpenRouter"
    assert all("cloudflare" not in u for u in _urls(calls)), "only Groq's small model decides"
    assert not hasattr(app_module, "_ask_cloudflare")


def test_balancer_can_keep_a_student_on_groq(app_module, client, setup, monkeypatch):
    calls, behaviour = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    _brain(behaviour, '{"send_to": "Groq", "reason": "even"}')
    _fill(app_module, "Groq", 9)
    _others(app_module)
    assert "groq answer" in _ask(client)


def test_unusable_balancer_answer_means_least_busy(app_module, client, setup, monkeypatch):
    calls, behaviour = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    _brain(behaviour, '{"send_to": "SomethingElse"}')
    _fill(app_module, "Groq", 2)
    _others(app_module)
    assert "backup answer" in _ask(client)


def test_balancer_only_sees_numbers(app_module, setup, monkeypatch):
    calls, behaviour = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    _brain(behaviour, '{"send_to": "OpenRouter"}')
    for _ in range(6):
        app_module.note_question("Secret Name (grade 5)")
    app_module.note_question("Other Kid (grade 3)")
    assert app_module.choose_provider("Secret Name (grade 5)") == "OpenRouter"
    url, body = calls[0]
    assert "groq.com" in url and body["model"] == "llama-3.1-8b-instant"
    dumped = _json.dumps(body)
    assert "Secret" not in dumped and "Other Kid" not in dumped and "slots" not in dumped


def test_one_student_alone_skips_the_balancer(app_module, setup, monkeypatch):
    calls, behaviour = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    _brain(behaviour, '{"send_to": "OpenRouter"}')
    app_module.note_question("only kid")
    assert app_module.choose_provider("only kid") is None and calls == []


def test_broken_balancer_means_least_busy(app_module, client, setup, monkeypatch):
    calls, behaviour = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")   # behaviour["brain"] None -> 500
    _fill(app_module, "Groq", 4)
    _others(app_module)
    assert "backup answer" in _ask(client)
    assert app_module.balancer_status()["resting"]


def test_no_groq_key_means_least_busy(app_module, setup, monkeypatch):
    calls, behaviour = setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: None)
    _brain(behaviour, '{"send_to": "Groq"}')
    _fill(app_module, "Groq", 3)
    _others(app_module)
    app_module.note_question("me")
    assert app_module.choose_provider("me") == "OpenRouter" and calls == []
    assert not app_module.balancer_status()["ready"]


def test_balancer_knows_groq_is_best_then_the_order(app_module, setup, monkeypatch):
    calls, behaviour = setup
    for k in ("OPENROUTER_API_KEY", "MISTRAL_API_KEY", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"):
        monkeypatch.setenv(k, "x")
    _brain(behaviour, '{"send_to": "Groq"}')
    _others(app_module)
    app_module.note_question("me")
    app_module.choose_provider("me")
    url, body = calls[0]
    ais = _json.loads(body["messages"][1]["content"])["ais"]
    assert [(a["name"], a["quality_rank"]) for a in ais] == [
        ("Groq", 1), ("OpenRouter", 2), ("Mistral", 3), ("Cloudflare", 4)]
    assert "Groq is the best" in body["messages"][0]["content"]


def test_without_a_decision_a_tie_goes_to_the_better_ai(app_module, setup, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    _fill(app_module, "Groq", 3)
    _fill(app_module, "OpenRouter", 1)
    _fill(app_module, "Mistral", 1)
    assert app_module.choose_provider("me") == "OpenRouter"


# ---------- Cloudflare also answers kids ----------

def test_cloudflare_answers_with_gpt_oss_120b(app_module, client, setup, monkeypatch):
    calls, _ = setup
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    _fill(app_module, "Groq", 3)
    assert "backup answer" in _ask(client)
    url, body = [c for c in calls if "/ai/v1/chat/completions" in c[0] and c[1].get("stream")][-1]
    assert "accounts/acct123/ai/v1/chat/completions" in url and body["model"] == "@cf/openai/gpt-oss-120b"


def test_cloudflare_needs_the_account_id_too(app_module, setup, monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    assert all(p["name"] != "Cloudflare" for p in app_module.backups_ready())


def test_cloudflare_falls_back_to_llama_if_model_missing(app_module, client, setup, monkeypatch):
    calls, _ = setup
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    real = app_module.requests.post

    def fake(url, headers=None, json=None, stream=None, timeout=None):
        if "cloudflare" in url and json and json.get("model") == "@cf/openai/gpt-oss-120b":
            calls.append((url, json))
            r = _Resp(404); r.close = lambda: None
            return r
        return real(url, headers=headers, json=json, stream=stream, timeout=timeout)
    monkeypatch.setattr(app_module.requests, "post", fake)
    _fill(app_module, "Groq", 3)
    assert "backup answer" in _ask(client)
    assert calls[-1][1]["model"] == "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


def test_gemini_is_gone_and_admin_shows_model_and_rank(app_module, setup):
    names = [p["name"] for p in app_module.BACKUP_PROVIDERS]
    assert "Gemini" not in names
    st = {p["name"]: p for p in app_module.backup_status()["providers"]}
    assert set(st) == {"Groq", "OpenRouter", "Mistral", "Cloudflare"}
    assert st["Groq"]["rank"] == 1 and st["Groq"]["model"] == app_module.TUTOR_MODEL
    assert st["Cloudflare"]["model"] == "@cf/openai/gpt-oss-120b"
