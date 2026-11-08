"""Homework photos: the photo is read once, the tutor gets the written-out
problems, and the page gets green-box data for the problem being asked about."""

import base64
import json

import pytest
from conftest import make_account

TINY_JPEG = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff\xe0fakejpeg").decode()


class _Resp:
    def __init__(self, status=200, content="", stream_lines=None):
        self.status_code = status
        self._content = content
        self._lines = stream_lines or []
        self.text = content
        self.headers = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}

    def iter_lines(self):
        for l in self._lines:
            yield l.encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _tutor_stream(text):
    return [f"data: {json.dumps({'choices': [{'delta': {'content': text}, 'finish_reason': 'stop'}]})}",
            "data: [DONE]"]


GOOD_READING = {
    "appropriate": True, "homework": True,
    "notes": "1) 1/2 + 1/4\n2) 3/4 + 1/8",
    "problems": [{"label": "Problem 1", "box": [70, 160, 910, 330]},
                 {"label": "Problem 2", "box": [70, 360, 910, 530]}],
    "focus": 1,
}


def _fake_groq(app_module, monkeypatch, reading):
    seen = {"vision": [], "tutor": None}

    def fake_post(url, headers=None, json=None, stream=None, timeout=None):
        msgs = (json or {}).get("messages", [])
        if any(isinstance(m.get("content"), list) for m in msgs):      # the photo reader
            seen["vision"].append(json)
            return _Resp(content=reading if isinstance(reading, str) else __import__("json").dumps(reading))
        if not stream:                                                 # the off-task gate
            return _Resp(content='{"violation": 0, "web": 0}')
        seen["tutor"] = json
        return _Resp(stream_lines=_tutor_stream("Look at the problem in the green box!"))

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")
    return seen


def _ask(client, image=TINY_JPEG, text="help with number 2", messages=None):
    body = {"messages": messages or [{"role": "user", "content": text}]}
    if image is not None:
        body["image"] = image
        body["imageSize"] = [1000, 1300]
    res = client.post("/api/chat", json=body)
    return res.get_data(as_text=True)


def test_photo_gives_green_box_data_then_the_tutor_reply(app_module, client, monkeypatch):
    make_account(client, "Full Kid", "9", "password1")  # photos/voices need an email account
    seen = _fake_groq(app_module, monkeypatch, GOOD_READING)
    out = _ask(client)
    assert out.startswith(app_module.PHOTO_MARK)
    data = json.loads(out.split(app_module.PHOTO_MARK)[1])
    assert data["focus"] == 1 and data["problems"][1]["label"] == "Problem 2"
    assert "green box" in out.split(app_module.PHOTO_MARK)[2]
    # the tutor itself never gets the picture, only the written-out problems
    system = seen["tutor"]["messages"][0]["content"]
    assert "3/4 + 1/8" in system and "Problem 2" in system
    assert not any(isinstance(m["content"], list) for m in seen["tutor"]["messages"])
    assert seen["tutor"]["model"] == app_module.TUTOR_MODEL


def test_not_homework_photo_gets_a_friendly_notice(app_module, client, monkeypatch):
    make_account(client, "Full Kid", "9", "password1")  # photos/voices need an email account
    seen = _fake_groq(app_module, monkeypatch, {"appropriate": True, "homework": False, "notes": "a dog"})
    out = _ask(client)
    assert out.startswith(app_module.NOTICE) and "schoolwork" in out
    assert seen["tutor"] is None


def test_inappropriate_photo_is_blocked(app_module, client, monkeypatch):
    make_account(client, "Full Kid", "9", "password1")  # photos/voices need an email account
    seen = _fake_groq(app_module, monkeypatch, {"appropriate": False, "homework": True, "notes": "x"})
    out = _ask(client)
    assert out.startswith(app_module.NOTICE) and "can't help with that picture" in out
    assert seen["tutor"] is None


def test_unreadable_reply_means_a_retry_and_then_a_notice(app_module, client, monkeypatch):
    make_account(client, "Full Kid", "9", "password1")  # photos/voices need an email account
    seen = _fake_groq(app_module, monkeypatch, "sorry I can't")
    out = _ask(client)
    assert out.startswith(app_module.NOTICE) and "couldn't read that photo" in out
    assert len(seen["vision"]) >= 2


@pytest.mark.parametrize("bad", ["not a data url", "data:image/gif;base64,AAAA",
                                 "data:text/html;base64,PGgxPg==", 12345])
def test_bad_images_are_refused_before_calling_groq(app_module, client, monkeypatch, bad):
    make_account(client, "Full Kid", "9", "password1")  # photos/voices need an email account
    seen = _fake_groq(app_module, monkeypatch, GOOD_READING)
    out = _ask(client, image=bad)
    assert out.startswith(app_module.NOTICE) and seen["vision"] == [] and seen["tutor"] is None


def test_earlier_photo_notes_reach_the_tutor_as_text(app_module, client, monkeypatch):
    seen = _fake_groq(app_module, monkeypatch, GOOD_READING)
    _ask(client, image=None, messages=[
        {"role": "user", "content": "help", "photoNotes": "2) 3/4 + 1/8"},
        {"role": "assistant", "content": "Which one?"},
        {"role": "user", "content": "number 2"},
    ])
    first = seen["tutor"]["messages"][1]["content"]
    assert "3/4 + 1/8" in first and "photo of homework" in first


def test_box_cleaning():
    from app import _clean_box, _clean_photo_reading
    assert _clean_box([0.1, 0.2, 0.5, 0.4]) == [100, 200, 500, 400]        # fractions
    assert _clean_box([500, 400, 100, 200]) == [100, 200, 500, 400]        # flipped corners
    assert _clean_box([200, 260, 1000, 1300], (2000, 2600)) == [100, 100, 500, 500]  # pixels
    assert _clean_box([2000, 1, 3000, 2]) is None                          # pixels, no size
    assert _clean_box([10, 10, 12, 500]) is None                           # too thin
    assert _clean_box("nope") is None
    r = _clean_photo_reading({"homework": True, "problems": [{"label": "Q1", "box": [1, 1, 900, 900]}], "focus": 7})
    assert r["focus"] == 0                                                 # only one problem
    r = _clean_photo_reading({"homework": False, "problems": [{"box": [1, 1, 900, 900]}]})
    assert r["problems"] == [] and r["focus"] == -1


def test_photo_limit_per_network(app_module, client, monkeypatch):
    make_account(client, "Full Kid", "9", "password1")  # photos/voices need an email account
    seen = _fake_groq(app_module, monkeypatch, GOOD_READING)
    monkeypatch.setattr(app_module, "_photo_limiter", app_module.SlidingWindowLimiter(1, 3600))
    _ask(client)
    out = _ask(client)
    assert out.startswith(app_module.NOTICE) and "a lot of photos" in out
