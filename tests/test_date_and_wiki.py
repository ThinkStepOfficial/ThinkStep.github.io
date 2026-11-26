"""The AIs are told today's real date (their own knowledge stops ~2024),
and Wikipedia backs up the web search when Groq's search fails."""
import json as _json
from datetime import datetime, timezone

from test_smarter_models import _Reply, _ask, _fake_groq


def test_today_line_has_the_real_date(app_module):
    line = app_module.today_line()
    assert str(datetime.now(timezone.utc).year) in line and "TODAY'S DATE" in line


def test_tutor_is_told_the_date_and_to_trust_web_notes(app_module, client, monkeypatch):
    sent = _fake_groq(app_module, monkeypatch, [_Reply(content=["hi"])])
    _ask(client, "what is 2+2")
    system = sent[0]["messages"][0]["content"]
    assert "TODAY'S DATE" in system and "trust them over what you remember" in system
    # the fixed instructions still come first (so Groq can reuse them)
    assert system.startswith(app_module.SYSTEM_PROMPT)


def test_gate_and_web_search_get_the_date(app_module, monkeypatch):
    seen = []

    class R:
        status_code = 200
        headers = {}
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": '{"violation": 0, "web": 1}'}}]}

    def fake(url, headers=None, json=None, **k):
        seen.append(json)
        return R()
    monkeypatch.setattr(app_module.requests, "post", fake)
    h = [{"role": "user", "content": "who won the 2026 world cup"}]
    assert app_module.topic_gate_check(h, "k")["web"] is True
    assert "TODAY'S DATE" in seen[0]["messages"][1]["content"]
    app_module.groq_web_research(h, "k")
    assert "TODAY'S DATE" in seen[-1]["messages"][1]["content"]
    assert "NEWEST" in seen[-1]["messages"][0]["content"]
    assert "about 2024, 2025, the current year" in app_module.TOPIC_GATE_POLICY


class _G:
    def __init__(self, data, status=200):
        self.status_code, self._d = status, data
    def json(self):
        return self._d


def test_wikipedia_backs_up_a_failed_web_search(app_module, monkeypatch):
    app_module.wikipedia_research = app_module._real_wikipedia_research
    monkeypatch.setattr(app_module, "groq_web_research", lambda h, k: None)
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(params)
        assert "ThinkStep" in headers["User-Agent"]
        if params.get("list") == "search":
            return _G({"query": {"search": [{"title": "Photosynthesis"}, {"title": "Chlorophyll"}]}})
        return _G({"query": {"pages": [
            {"title": "Photosynthesis", "extract": "Photosynthesis is how plants make food from light."},
            {"title": "Chlorophyll", "extract": "Chlorophyll is the green pigment."}]}})
    monkeypatch.setattr(app_module.requests, "get", fake_get)
    notes = app_module.web_research([{"role": "user", "content": "what is photosynthesis for my homework"}], "k")
    assert "plants make food" in notes and "(Wikipedia: Photosynthesis)" in notes
    assert calls[0]["srsearch"] == "photosynthesis"


def test_short_follow_up_searches_the_earlier_topic(app_module):
    h = [{"role": "user", "content": "how do volcanoes erupt"}, {"role": "assistant", "content": "..."},
         {"role": "user", "content": "search it up"}]
    assert "volcanoes" in app_module._wiki_query(h)


def test_wikipedia_trouble_never_breaks_anything(app_module, monkeypatch):
    app_module.wikipedia_research = app_module._real_wikipedia_research
    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(app_module.requests, "get", boom)
    assert app_module.wikipedia_research([{"role": "user", "content": "what is a volcano"}]) is None
    monkeypatch.setattr(app_module.requests, "get", lambda *a, **k: _G({}, status=503))
    assert app_module.wikipedia_research([{"role": "user", "content": "what is a volcano"}]) is None


def test_groq_search_wins_when_it_works(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "groq_web_research", lambda h, k: "- fact (Source)")
    monkeypatch.setattr(app_module, "wikipedia_research", lambda h: (_ for _ in ()).throw(AssertionError("not needed")))
    assert app_module.web_research([{"role": "user", "content": "x"}], "k") == "- fact (Source)"
