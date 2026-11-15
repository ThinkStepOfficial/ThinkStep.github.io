"""The "Get the app" (install to home screen) feature was removed. Browsers
that already installed its helper get it switched off cleanly."""


def test_chat_page_has_no_install_button_or_manifest(client):
    html = client.get("/app").get_data(as_text=True)
    assert 'id="installBtn"' not in html
    assert 'rel="manifest"' not in html
    assert "beforeinstallprompt" not in html
    assert "serviceWorker.register(" not in html
    assert "unregister()" in html  # old helpers are switched off


def test_manifest_route_is_gone(client):
    assert client.get("/manifest.webmanifest").status_code == 404


def test_old_helper_switches_itself_off(client):
    res = client.get("/sw.js")
    assert res.status_code == 200 and "javascript" in res.mimetype
    body = res.get_data(as_text=True)
    assert "registration.unregister()" in body and "caches.delete" in body
    assert "addEventListener('fetch'" not in body  # never gets between kids and the site
