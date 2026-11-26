"""Voice typing: the browser records audio and Groq's Whisper writes it down.
Works in every browser; audio is never saved."""
import io


class _W:
    def __init__(self, text="what is seven times eight", status=200):
        self.status_code = status
        self._t = text

    def json(self):
        return {"text": self._t}


def _post(client, data=b"x" * 3000, kind="audio/webm"):
    return client.post("/api/transcribe", data={"audio": (io.BytesIO(data), "speech.webm", kind)},
                       content_type="multipart/form-data")


def test_speech_becomes_text(app_module, client, monkeypatch):
    sent = {}
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "test-key-not-real")

    def fake(url, headers=None, files=None, data=None, timeout=None, **k):
        sent.update(url=url, files=files, data=data)
        return _W(" what is  seven times eight ")
    monkeypatch.setattr(app_module.requests, "post", fake)
    res = _post(client)
    assert res.status_code == 200 and res.get_json()["text"] == "what is seven times eight"
    assert sent["url"].endswith("/audio/transcriptions")
    assert sent["data"]["model"] == "whisper-large-v3-turbo"


def test_silence_ghost_words_count_as_nothing(app_module, client, monkeypatch):
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "k")
    monkeypatch.setattr(app_module.requests, "post", lambda *a, **k: _W("Thank you."))
    assert _post(client).get_json()["text"] == ""


def test_guests_must_answer_age_question_first(app_module, unverified_client):
    assert _post(unverified_client).status_code == 403


def test_bad_audio_is_refused(app_module, client, monkeypatch):
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "k")
    assert _post(client, kind="text/plain").status_code == 400
    assert _post(client, data=b"x" * (app_module.TRANSCRIBE_MAX_BYTES + 10)).status_code == 400
    assert client.post("/api/transcribe").status_code == 400


def test_groq_trouble_gives_a_friendly_error(app_module, client, monkeypatch):
    monkeypatch.setattr(app_module, "_load_groq_api_key", lambda: "k")
    monkeypatch.setattr(app_module.requests, "post", lambda *a, **k: _W(status=500))
    res = _post(client)
    assert res.status_code == 502 and "type it" in res.get_json()["error"]


def test_page_uses_the_real_microphone_and_explains_blocked_mics(client):
    html = client.get("/app").get_data(as_text=True)
    assert "getUserMedia" in html and "/api/transcribe" in html
    assert "Your microphone is blocked" in html
    assert "webkitSpeechRecognition" not in html
