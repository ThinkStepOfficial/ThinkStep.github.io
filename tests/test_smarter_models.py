"""The tutor thinks harder (reasoning_effort "high"), and every way that can
go wrong falls back to something that still answers the student.

Groq is faked: each streaming call pops the next scripted reply, so a test
can say "the first attempt gets rejected, the second succeeds" and check
exactly what the app sent each time.
"""

import json

from conftest import make_account

SMART, QUICK = "openai/gpt-oss-120b", "openai/gpt-oss-20b"


class _Reply:
    """One scripted streaming reply from the fake Groq."""

    def __init__(self, status=200, content=(), finish="stop", reasoning=(), usage=None):
        self.status_code = status
        self._content = list(content)
        self._finish = finish
        self._reasoning = list(reasoning)
        self._usage = usage
        self.text = "" if status == 200 else '{"error": "scripted failure"}'
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(response=self)

    def iter_lines(self):
        def line(delta, finish=None, extra=None):
            chunk = {"choices": [{"delta": delta, "finish_reason": finish}]}
            if extra:
                chunk.update(extra)
            return b"data: " + json.dumps(chunk).encode()
        for r in self._reasoning:  # private thinking, in its own field
            yield line({"reasoning": r})
        for c in self._content:
            yield line({"content": c})
        yield line({}, self._finish, {"x_groq": {"usage": self._usage}} if self._usage else None)
        yield b"data: [DONE]"


class _Gate:
    status_code = 200
    headers = {}

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": '{"violation": 0}'}}]}


def _fake_groq(app_module, monkeypatch, script):
    sent = []

    def fake_post(url, headers=None, json=None, stream=None, timeout=None):
        if not stream:
            return _Gate()
        sent.append(json)
        return script.pop(0)

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")
    return sent


def _ask(client, text="what is 3/4 + 1/6?", model=QUICK):
    res = client.post("/api/chat", json={"model": model, "messages": [{"role": "user", "content": text}]})
    return res.get_data(as_text=True).replace("\x00", "").replace("\x02", "")


def _signed_in(client, name="Smart Kid", grade="7"):
    assert make_account(client, name, grade, "smart-pass-123").status_code == 200
    assert client.post("/api/login", json={"name": name, "password": "smart-pass-123"}).status_code == 200


# ---------------------------------------------------------------- thinking

def test_first_attempt_asks_for_high_effort_thinking(app_module, client, monkeypatch):
    sent = _fake_groq(app_module, monkeypatch, [_Reply(content=["What's 4 x 3?"])])
    reply = _ask(client)

    assert reply == "What's 4 x 3?"
    assert sent[0]["reasoning_effort"] == "high"
    assert sent[0]["include_reasoning"] is False, "we never show thinking, so don't download it"
    # the thinking counts toward this cap, so it has to be well above
    # Groq's 1,024 default or a hard think could leave no room to answer
    assert sent[0]["max_completion_tokens"] >= 4096


def test_private_thinking_never_reaches_the_student(app_module, client, monkeypatch):
    _fake_groq(app_module, monkeypatch, [
        _Reply(reasoning=["SECRET: the answer is 11/12"], content=["What do 4 and 6 both go into?"]),
    ])
    reply = _ask(client)
    assert "SECRET" not in reply and "11/12" not in reply
    assert reply == "What do 4 and 6 both go into?"


# ---------------------------------------------------------------- fallbacks

def test_if_groq_rejects_the_big_request_it_retries_exactly_like_before(app_module, client, monkeypatch):
    for status in (400, 413):
        sent = _fake_groq(app_module, monkeypatch, [
            _Reply(status=status), _Reply(content=["Let's look at the bottoms first."]),
        ])
        reply = _ask(client)

        assert reply == "Let's look at the bottoms first.", f"status {status}"
        retry = sent[1]
        for key in ("reasoning_effort", "max_completion_tokens", "include_reasoning"):
            assert key not in retry, f"the safe retry must be the old, known-good request ({key})"


def test_thinking_so_long_it_never_answers_triggers_a_lighter_retry(app_module, client, monkeypatch):
    sent = _fake_groq(app_module, monkeypatch, [
        _Reply(reasoning=["hmm " * 50], content=[], finish="length"),
        _Reply(content=["Try drawing it!"]),
    ])
    assert _ask(client) == "Try drawing it!"
    assert len(sent) == 2


def test_an_answer_cut_off_midway_says_so_instead_of_retrying(app_module, client, monkeypatch):
    sent = _fake_groq(app_module, monkeypatch, [_Reply(content=["First, find the"], finish="length")])
    reply = _ask(client)
    assert reply.startswith("First, find the")
    assert "keep going" in reply
    assert len(sent) == 1, "can't retry once words are already on the student's screen"


def test_busy_smartest_model_falls_back_to_the_quick_one(app_module, client, monkeypatch):
    _signed_in(client)
    sent = _fake_groq(app_module, monkeypatch, [
        _Reply(status=429), _Reply(status=429),       # 120b: high, then defaults
        _Reply(content=["Good question! What's 10% of 80?"]),  # 20b steps in
    ])
    reply = _ask(client, "what's 15% of 80?", model=SMART)

    assert reply == "Good question! What's 10% of 80?"
    assert [s["model"] for s in sent] == [SMART, SMART, QUICK]
    assert sent[2]["reasoning_effort"] == "high", "the stand-in should think hard too"


def test_if_both_models_are_busy_the_student_gets_the_friendly_message(app_module, client, monkeypatch):
    sent = _fake_groq(app_module, monkeypatch, [_Reply(status=429)] * 4)
    reply = _ask(client)

    assert [s["model"] for s in sent] == [SMART, SMART, QUICK, QUICK]
    assert "usage limit" in reply.lower(), "should show the friendly busy message, not crash"


# ---------------------------------------------------------------- one model, no daily cap

def test_everyone_gets_the_smart_model_even_if_an_old_page_asks_for_another(app_module, client, monkeypatch):
    # an old copy of the page cached in someone's browser may still send
    # the retired quick model's name — the server ignores it
    sent = _fake_groq(app_module, monkeypatch, [_Reply(content=["ok"]), _Reply(content=["ok"])])
    _ask(client, model=QUICK)
    _ask(client, model="something-made-up")
    assert [s["model"] for s in sent] == [SMART, SMART]


def test_guests_can_use_the_smart_model(app_module, client, monkeypatch):
    _fake_groq(app_module, monkeypatch, [_Reply(content=["What's 7 x 4 first?"])])
    reply = _ask(client, model=SMART)
    assert reply == "What's 7 x 4 first?", f"guests used to be blocked from this model: {reply!r}"


def test_accounts_have_no_daily_question_limit(app_module, client, monkeypatch):
    _signed_in(client)
    n = 30  # well past the old 20-per-day cap
    _fake_groq(app_module, monkeypatch, [_Reply(content=[f"hint {i}"]) for i in range(n)])
    replies = [_ask(client, f"question {i}", model=SMART) for i in range(n)]
    assert replies == [f"hint {i}" for i in range(n)], "every question should get a real answer"


def test_guests_get_5_free_questions_then_must_sign_up(app_module, client, monkeypatch):
    app_module.GUEST_QUESTION_LIMIT = app_module._real_guest_question_limit
    limit = app_module.GUEST_QUESTION_LIMIT
    assert limit == 5
    sent = _fake_groq(app_module, monkeypatch, [_Reply(content=["ok"]) for _ in range(limit)])
    for i in range(limit):
        assert _ask(client, f"q{i}") == "ok"
    final = _ask(client, "one more?")
    assert "account" in final.lower() and len(sent) == limit, "the 6th guest question should ask them to sign up"


def test_model_list_and_old_usage_endpoint(client):
    assert client.get("/api/models").get_json() == [SMART]
    assert client.get("/api/usage").status_code == 404, "the daily-cap endpoint is gone"


def test_chat_page_has_no_model_picker():
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")
    assert '<select id="model"' not in html
    assert "modelInfoBtn" not in html and "modelPopover" not in html
    assert f"const CHAT_MODEL = '{SMART}'" in html
    assert 'id="toolsToggle"' in html, "the 🛠 tools button lives next to where the picker was — keep it"


def test_usage_logging_handles_groqs_usage_block(app_module, client, monkeypatch, capsys):
    usage = {"prompt_tokens": 1800, "completion_tokens": 900,
             "prompt_tokens_details": {"cached_tokens": 1536}}
    _fake_groq(app_module, monkeypatch, [_Reply(content=["Hi!"], usage=usage)])
    _ask(client)
    out = capsys.readouterr().out
    assert "1536 cached" in out and "264 counted" in out


# ---------------------------------------------------------------- caching

def test_tutor_instructions_come_first_so_groq_can_cache_them(app_module, monkeypatch):
    """Groq only reuses (and doesn't charge rate limits for) the start of a
    request when it's identical to a recent one. Two different students
    must get the exact same opening — the long tutor instructions — with
    their own details only after."""
    systems = []
    # (grades 7+ so the plain 13-or-older signup is valid — younger grades
    # correctly require a parent's consent first)
    for name, grade in (("Ava Example", "8"), ("Ben Example", "11")):
        c = app_module.app.test_client()
        _signed_in(c, name, grade)
        sent = _fake_groq(app_module, monkeypatch, [_Reply(content=["ok"])])
        _ask(c)
        systems.append(sent[0]["messages"][0]["content"])

    for s in systems:
        assert s.startswith(app_module.SYSTEM_PROMPT), "instructions must be the very first thing"
    assert systems[0] != systems[1], "each student's own details still have to be there"
    assert "Grade: 8" in systems[0] and "Grade: 11" in systems[1]


# ---------------------------------------------------------------- gate

def test_gate_skips_the_safeguard_model_for_a_while_after_it_says_busy(app_module, client, monkeypatch):
    calls = []

    class Busy:
        status_code = 429
        headers = {"retry-after": "30"}
        text = ""

    def fake_post(url, headers=None, json=None, stream=None, timeout=None):
        if stream:
            return _Reply(content=["ok"])
        calls.append(json["model"])
        return Busy() if json["model"] == "openai/gpt-oss-safeguard-20b" else _Gate()

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")

    _ask(client, "why is the sky blue?")
    _ask(client, "what is photosynthesis?")
    assert calls == [
        "openai/gpt-oss-safeguard-20b", "openai/gpt-oss-20b",  # 1st: busy -> backup
        "openai/gpt-oss-20b",                                  # 2nd: straight to backup
    ]


def test_guest_status_shows_5_free_questions(app_module, client):
    app_module.GUEST_QUESTION_LIMIT = app_module._real_guest_question_limit
    data = client.get("/api/guest-status").get_json()
    assert data["limit"] == 5 and data["remaining"] == 5
