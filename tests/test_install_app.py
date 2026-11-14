"""ThinkStep can be installed to the home screen like an app (no app store)."""
import json
from pathlib import Path

from PIL import Image

STATIC = Path(__file__).resolve().parent.parent / "static"


def test_manifest_is_served_with_the_right_type(client):
    res = client.get("/manifest.webmanifest")
    assert res.status_code == 200
    assert res.mimetype == "application/manifest+json"
    data = json.loads(res.get_data(as_text=True))
    assert data["short_name"] == "ThinkStep"
    assert data["display"] == "standalone"
    assert data["start_url"].startswith("/app")
    sizes = {i["sizes"] for i in data["icons"]}
    assert {"192x192", "512x512"} <= sizes
    assert any(i.get("purpose") == "maskable" for i in data["icons"])


def test_every_manifest_icon_exists_and_has_the_right_size(client):
    data = json.loads(client.get("/manifest.webmanifest").get_data(as_text=True))
    for icon in data["icons"]:
        res = client.get(icon["src"])
        assert res.status_code == 200, icon["src"]
        w, h = (int(n) for n in icon["sizes"].split("x"))
        assert Image.open(STATIC / Path(icon["src"]).name).size == (w, h)


def test_service_worker_is_served_from_the_top_of_the_site(client):
    res = client.get("/sw.js")
    assert res.status_code == 200
    assert "javascript" in res.mimetype
    body = res.get_data(as_text=True)
    assert "offline.html" in body
    # it must never cache chats or anything from the API
    assert "/api/" not in body.replace("anything from /api/", "")


def test_offline_page_exists(client):
    res = client.get("/static/offline.html")
    assert res.status_code == 200
    assert "offline" in res.get_data(as_text=True).lower()


def test_chat_page_links_the_manifest_and_has_the_install_button(client):
    html = client.get("/app").get_data(as_text=True)
    assert 'rel="manifest" href="/manifest.webmanifest"' in html
    assert 'id="installBtn"' in html
    assert "serviceWorker.register('/sw.js')" in html
    assert "beforeinstallprompt" in html
