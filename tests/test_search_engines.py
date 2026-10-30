"""What search engines need to find and index ThinkStep: robots.txt, a
sitemap, a cheap keep-awake route, and correct homepage metadata.

Mistakes here are silent — Google doesn't report a broken sitemap or a
JSON-LD typo to you, it just quietly ignores it — so they're pinned down
by tests instead.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "static"
LIVE = "https://thinkstep-github-io-1.onrender.com/"
# the homepage itself lives on GitHub Pages; Render's "/" forwards there
HOME = "https://thinkstepofficial.github.io/thinkstep-tutor/"


def test_robots_txt_allows_crawling_and_points_to_the_sitemap(client):
    res = client.get("/robots.txt", headers={"Host": "thinkstep-github-io-1.onrender.com"})
    body = res.get_data(as_text=True)
    assert res.status_code == 200 and res.mimetype == "text/plain"
    assert "Allow: /" in body
    assert "Disallow: /api/" in body
    assert "Disallow: /admin" in body, "the admin page must never show up in search"
    assert "Sitemap: https://thinkstep-github-io-1.onrender.com/sitemap.xml" in body


def test_sitemap_is_valid_and_every_page_in_it_actually_loads(client):
    res = client.get("/sitemap.xml", headers={"Host": "thinkstep-github-io-1.onrender.com"})
    assert res.status_code == 200
    root = ET.fromstring(res.get_data())  # raises if it's not valid XML
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [e.text for e in root.findall("s:url/s:loc", ns)]
    assert LIVE in locs, "the homepage must be in the sitemap"
    for loc in locs:
        assert loc.startswith("https://"), f"must advertise https: {loc}"
        path = loc.replace("https://thinkstep-github-io-1.onrender.com", "") or "/"
        assert client.get(path).status_code == 200, f"sitemap lists a broken page: {path}"
    assert not any("/admin" in l or "/api/" in l for l in locs)


def test_ping_is_cheap_and_never_touches_the_database_or_ai(app_module, client, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("/ping must not touch storage or call out to the AI")

    monkeypatch.setattr(app_module.STORE, "read", boom)
    monkeypatch.setattr(app_module.STORE, "write", boom)
    monkeypatch.setattr(app_module.requests, "post", boom)
    monkeypatch.setattr(app_module.requests, "get", boom)
    res = client.get("/ping")
    assert res.status_code == 200 and res.get_data(as_text=True) == "ok"


def _home():
    return (STATIC / "home.html").read_text(encoding="utf-8")


def test_homepage_title_and_description_lead_with_the_brand():
    html = _home()
    title = re.search(r"<title>(.*?)</title>", html).group(1)
    desc = re.search(r'<meta name="description" content="([^"]+)"', html).group(1)
    assert title.startswith("ThinkStep")
    assert "tutor" in title.lower()
    assert "ThinkStep" in desc
    # Google cuts titles off around 60 characters and descriptions ~160
    assert len(title) <= 70, f"title too long ({len(title)}): {title}"
    assert len(desc) <= 170, f"description too long ({len(desc)})"


def test_homepage_structured_data_is_valid_json_naming_thinkstep():
    blocks = re.findall(r'<script type="application/ld\+json">([\s\S]*?)</script>', _home())
    assert blocks, "structured data block missing"
    data = json.loads(blocks[0])  # a single typo here would make Google ignore it all
    names = [node.get("name") for node in data["@graph"]]
    assert names.count("ThinkStep") == len(names)
    assert all(node["url"] == HOME for node in data["@graph"])


def test_homepage_has_canonical_and_absolute_preview_image():
    html = _home()
    assert f'<link rel="canonical" href="{HOME}">' in html
    assert f'content="{LIVE}static/og-image.png"' in html, "link previews need a full image URL"
    assert (STATIC / "og-image.png").exists()


def test_google_search_console_verification_tag_is_still_there():
    # removing this would silently un-verify the site in Search Console
    assert 'name="google-site-verification"' in _home()


def test_admin_page_tells_search_engines_to_stay_away():
    assert '<meta name="robots" content="noindex, nofollow">' in (STATIC / "admin.html").read_text(encoding="utf-8")


def test_on_render_the_homepage_moves_to_github_pages(client, monkeypatch):
    import app as app_module
    monkeypatch.setenv("RENDER", "true")
    res = client.get("/")
    assert res.status_code == 301
    assert res.headers["Location"] == app_module.PUBLIC_HOME_URL
    # the tutor itself stays right here
    assert client.get("/app").status_code == 200


def test_locally_the_homepage_is_still_served_here(client, monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    assert client.get("/").status_code == 200
