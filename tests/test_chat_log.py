"""The admin can read every chat (kids and parents are told on the Privacy
page and by a notice in the app). Chats are kept 30 days, photos are never
saved, and deleting an account deletes its chats."""
from datetime import datetime, timedelta, timezone

from conftest import TEST_ADMIN_PASSWORD
from test_smarter_models import _Reply, _ask, _fake_groq, _signed_in


def _admin(app_module):
    c = app_module.app.test_client()
    assert c.post("/api/admin/login", json={"password": TEST_ADMIN_PASSWORD}).status_code == 200
    return c


def test_guest_chat_shows_up_for_the_admin(app_module, client, monkeypatch):
    _fake_groq(app_module, monkeypatch, [_Reply(content=["What do 4 and 6 both go into?"])])
    assert _ask(client, "what is 3/4 + 1/6?") == "What do 4 and 6 both go into?"
    admin = _admin(app_module)
    people = admin.get("/api/admin/chats").get_json()["people"]
    assert len(people) == 1 and people[0]["guest"] and people[0]["who"].startswith("Guest ")
    assert len(people[0]["who"]) == len("Guest ") + 6
    msgs = admin.get("/api/admin/chats", query_string={"person": people[0]["person"]}).get_json()["messages"]
    assert msgs[0]["q"] == "what is 3/4 + 1/6?" and msgs[0]["a"] == "What do 4 and 6 both go into?"


def test_account_chat_shows_name_and_grade(app_module, client, monkeypatch):
    _signed_in(client, "Mia Lopez", "8")
    _fake_groq(app_module, monkeypatch, [_Reply(content=["hint 1"]), _Reply(content=["hint 2"])])
    _ask(client, "first question")
    _ask(client, "second question")
    people = _admin(app_module).get("/api/admin/chats").get_json()["people"]
    assert people[0]["who"] == "Mia Lopez" and people[0]["grade"] == "8"
    assert people[0]["count"] == 2 and not people[0]["guest"]
    assert people[0]["lastQ"] == "second question"


def test_only_the_admin_can_read_chats(app_module, client, monkeypatch):
    _fake_groq(app_module, monkeypatch, [_Reply(content=["ok"])])
    _ask(client, "secret homework")
    assert client.get("/api/admin/chats").status_code == 401
    assert app_module.app.test_client().get("/api/admin/chats").status_code == 401


def test_failed_answers_are_not_saved(app_module, client, monkeypatch):
    _fake_groq(app_module, monkeypatch, [_Reply(status=500)] * 10)
    _ask(client, "this one breaks")
    assert _admin(app_module).get("/api/admin/chats").get_json()["people"] == []


def test_off_topic_questions_are_saved_as_blocked(app_module, client, monkeypatch):
    monkeypatch.setattr(app_module, "topic_gate_check", lambda h, k: {"blocked": True, "web": False})
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")
    _ask(client, "tell me a scary story")
    admin = _admin(app_module)
    person = admin.get("/api/admin/chats").get_json()["people"][0]["person"]
    msg = admin.get("/api/admin/chats", query_string={"person": person}).get_json()["messages"][0]
    assert msg["blocked"] and msg["q"] == "tell me a scary story"


def test_deleting_an_account_deletes_its_chats(app_module, client, monkeypatch):
    _signed_in(client, "Leaving Kid", "9")
    _fake_groq(app_module, monkeypatch, [_Reply(content=["ok"])])
    _ask(client, "bye")
    assert len(_admin(app_module).get("/api/admin/chats").get_json()["people"]) == 1
    assert client.post("/api/delete-account", json={"password": "smart-pass-123"}).status_code == 200
    assert _admin(app_module).get("/api/admin/chats").get_json()["people"] == []


def test_chats_older_than_30_days_are_deleted(app_module):
    old = (datetime.now(timezone.utc).date() - timedelta(days=31)).isoformat()
    app_module.STORE.write(f"chatlog-{old}.json", [{"person": "guest:x", "who": "Guest old",
                                                     "q": "old", "a": "old", "time": old + "T00:00:00+00:00"}])
    app_module.STORE.write("chatlog_days.json", [old])
    app_module.log_chat("guest:y", "Guest new", "", "new question", "new answer")
    assert app_module.STORE.read(f"chatlog-{old}.json") is None
    assert old not in app_module.STORE.read("chatlog_days.json")
    rows = app_module._chat_log_rows()
    assert [r["q"] for r in rows] == ["new question"]


def test_privacy_page_and_app_tell_people(client):
    privacy = client.get("/privacy").get_data(as_text=True)
    assert "30 days" in privacy and "can read them" in privacy
    assert "not on\n      our servers" not in privacy
    app_html = client.get("/app").get_data(as_text=True)
    assert 'id="chatNotice"' in app_html and "can read them to keep everyone safe" in app_html
