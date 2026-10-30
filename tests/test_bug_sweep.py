"""The "clean every bug" sweep (October 2026).

  - odd or broken request bodies get a polite 400, never a crash (500)
  - the AI health check doesn't ask Groq on every single poll
  - on Render, the sitemap lists only pages that really live there
  - the chat page explains "too big" (413) and "too fast" (429) kindly
"""
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "static"

ROUTES = ["/api/signup", "/api/login", "/api/feedback", "/api/forgot-password",
          "/api/guest-age", "/api/admin/login", "/api/chat"]
BODIES = ['[]', '"hello"', '123', 'null', 'not json at all',
          '{"name": 5, "password": [], "grade": {}, "email": 7, "review": null, '
          '"stars": "x", "messages": "hi", "profile": "x", "is13OrOlder": "yes"}']


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("body", BODIES)
def test_weird_bodies_never_crash_the_server(client, route, body):
    res = client.post(route, data=body, content_type="application/json")
    assert res.status_code < 500, f"{route} crashed on {body!r}"


def test_health_check_remembers_groqs_answer(app_module, client, monkeypatch):
    calls = []

    class OK:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key")
    monkeypatch.setattr(app_module.requests, "get", lambda *a, **k: calls.append(1) or OK())
    for _ in range(5):
        assert client.get("/api/health").get_json() == {"groq": "ok"}
    assert len(calls) == 1, "five polls in a row should only ask Groq once"


def test_render_sitemap_skips_forwarding_pages(client, monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    body = client.get("/sitemap.xml", headers={"Host": "x.onrender.com"}).get_data(as_text=True)
    assert "https://x.onrender.com/app" in body
    assert "<loc>https://x.onrender.com/</loc>" not in body
    assert "/privacy" not in body and "/terms" not in body


def test_chat_page_explains_too_big_and_too_fast():
    js = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "res.status === 413" in js and "res.status === 429" in js


def test_render_copies_of_privacy_and_terms_point_to_the_main_copies():
    for name in ("privacy", "terms"):
        html = (STATIC / f"{name}.html").read_text(encoding="utf-8")
        assert f'rel="canonical" href="https://thinkstepofficial.github.io/thinkstep-tutor/{name}.html"' in html
