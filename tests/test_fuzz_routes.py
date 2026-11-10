"""Hammers every route with odd inputs as a guest, a no-email account, an
email account and the admin: nothing may ever crash (500)."""
import json, itertools
from conftest import make_account

BODIES = [None, "", "null", "[]", "5", '"x"', "{}", '{"a":1}',
          json.dumps({"messages": "hi", "profile": [], "image": 5, "imageSize": "x", "checkWork": "yes",
                      "text": ["a"], "voice": 3, "email": {"a": 1}, "name": None, "kind": "quiz",
                      "total": "3", "id": [], "stars": "9", "password": 5, "grade": 13, "token": {}}),
          json.dumps({"messages": [{"role": "user", "content": {"x": 1}}, 5, None, {"role": 7}], "profile": {"quiz": {"total": 1e9}, "simple": 1}}),
          "\x00\xff garbage"]

def test_fuzz_everything(app_module, client, monkeypatch):
    monkeypatch.setattr(app_module.requests, "post", lambda *a, **k: (_ for _ in ()).throw(app_module.requests.exceptions.ConnectionError("no net")))
    monkeypatch.setattr(app_module.requests, "get", lambda *a, **k: (_ for _ in ()).throw(app_module.requests.exceptions.ConnectionError("no net")))
    rules = [r for r in app_module.app.url_map.iter_rules() if "static" not in r.endpoint]
    crashes = []
    for who in ["admin", "guest", "noemail", "email"]:
        if who == "admin":
            from conftest import TEST_ADMIN_PASSWORD
            assert client.post("/api/admin/login", json={"password": TEST_ADMIN_PASSWORD}).status_code == 200
        if who == "noemail":
            make_account(client, "Fuzz Plain", "9", "password1", recovery_email="")
        if who == "email":
            client.post("/api/logout")
            make_account(client, "Fuzz Mail", "9", "password1")
        for r in rules:
            path = r.rule
            for arg in r.arguments:
                path = path.replace(f"<{arg}>", "x").replace(f"<path:{arg}>", "x")
            path = __import__("re").sub(r"<[^>]+>", "x", path)
            if "logout" in path or "delete" in path or path == "/api/health" or (who == "admin" and "admin/login" in path):
                continue
            for m in r.methods - {"HEAD", "OPTIONS"}:
                for b in (BODIES if m == "POST" else [None]):
                    res = client.open(path, method=m, data=b, content_type="application/json")
                    try:
                        res.get_data()
                    except Exception as e:
                        crashes.append((who, m, path, b, repr(e)))
                        continue
                    if res.status_code >= 500:
                        crashes.append((who, m, path, (b or "")[:60], res.status_code))
    print("\n".join(map(str, crashes)))
    assert not crashes
