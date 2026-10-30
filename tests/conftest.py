"""
Shared pytest fixtures for the ThinkStep test suite.

The key idea: every single test gets a completely fresh, isolated copy of
app.py, pointed at an empty temp folder for all its data files (users.json,
feedback.json, tokens, etc). This means:

  - Tests NEVER touch your real data (the users.json with real accounts,
    real feedback, etc. next to app.py is never opened by these tests).
  - Tests can't leak state into each other (login lockouts, admin lockouts,
    presence tracking, tokens — all of it resets between every test), which
    was a real source of flaky/order-dependent failures in the old
    script-based compliance_test.py.
  - No real server or network port is used — everything runs through
    Flask's built-in test client, which calls straight into the app in the
    same process. Fast, and nothing to accidentally leave running.

By default everything is saved to files in a temp folder. Set
THINKSTEP_TEST_DATABASE_URL to a throwaway Postgres database to run the
same tests against the database backend (see _wipe_test_database below).

Nothing here sends real email or reaches Groq — email sending will just
fail quietly (no network in most environments), and Groq calls will fail
(no API key configured in the isolated test temp dir), both of which the
app already handles gracefully.
"""

import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path

import pytest
from werkzeug.security import generate_password_hash

APP_PY_PATH = Path(__file__).resolve().parent.parent / "app.py"

# A fixed password used only inside this test suite's isolated temp
# folders — never written anywhere near your real admin_config.json.
TEST_ADMIN_PASSWORD = "pytest-only-admin-password-not-real"

# ---- Database mode ----
# Set THINKSTEP_TEST_DATABASE_URL to a THROWAWAY Postgres database to run
# this entire suite against the database backend instead of files. The
# tables are wiped before every test, so never point this at real data.
TEST_DATABASE_URL = os.environ.get("THINKSTEP_TEST_DATABASE_URL", "").strip()
DB_MODE = bool(TEST_DATABASE_URL)

_module_counter = itertools.count()


def _wipe_test_database():
    import psycopg
    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS thinkstep_store, thinkstep_backups")


def _import_fresh_app():
    """Loads a brand-new copy of app.py under a unique module name —
    equivalent to the server starting up from scratch."""
    module_name = f"thinkstep_app_under_test_{next(_module_counter)}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.app.testing = True
    return module


@pytest.fixture()
def app_factory(tmp_path, monkeypatch):
    """Returns a function that starts a fresh copy of the app. Calling it a
    second time simulates a server restart: new process state, same saved
    data (same temp folder, or same database)."""
    monkeypatch.setenv("THINKSTEP_DATA_DIR", str(tmp_path))
    # Relaxes SESSION_COOKIE_SECURE so Flask's test client (which talks
    # plain http, not https) can keep a login session across requests.
    monkeypatch.setenv("THINKSTEP_DEBUG", "1")
    # Never let a developer's real settings leak into a test run.
    for var in ("DATABASE_URL", "ADMIN_PASSWORD", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    if DB_MODE:
        _wipe_test_database()
        monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
        # There's no admin_config.json on Render; the password comes from
        # this setting instead — so that's the path exercised here.
        monkeypatch.setenv("ADMIN_PASSWORD", TEST_ADMIN_PASSWORD)
    else:
        admin_hash = generate_password_hash(TEST_ADMIN_PASSWORD)
        (tmp_path / "admin_config.json").write_text(json.dumps({"password_hash": admin_hash}))

    started = []

    def start():
        module = _import_fresh_app()
        relax_spam_limits(module)
        started.append(module)
        return module

    yield start

    for module in started:
        module.STORE.close()
        sys.modules.pop(module.__name__, None)


@pytest.fixture()
def app_module(app_factory):
    """A brand-new copy of app.py, isolated from every other test.

    Using importlib to load app.py under a unique module name (instead of a
    plain `import app`) means every test gets its own fresh Flask app
    object and fresh copies of every in-memory global (login lockout
    counters, presence tracker, etc.) — true test isolation, not just data
    isolation.
    """
    return app_factory()


def relax_spam_limits(module):
    """Most tests fire lots of requests from one 'visitor' far faster than a
    kid could, so the spam limits are raised for them. Tests about the
    limits themselves call real_spam_limits() to put the real values back."""
    for name in ("_signup_limiter", "_feedback_limiter", "_reset_email_limiter",
                 "_chat_burst_limiter", "_chat_visitor_limiter"):
        getattr(module, name).limit = 10_000
    module._real_guest_daily_limit = module.GUEST_DAILY_LIMIT_PER_VISITOR
    module.GUEST_DAILY_LIMIT_PER_VISITOR = 10_000


def real_spam_limits(module):
    module._signup_limiter.limit = module.SIGNUP_LIMIT_PER_HOUR
    module._feedback_limiter.limit = module.FEEDBACK_LIMIT_PER_HOUR
    module._reset_email_limiter.limit = 5
    module._chat_burst_limiter.limit = module.CHAT_BURST_PER_MINUTE
    module._chat_visitor_limiter.limit = module.CHAT_LIMIT_PER_VISITOR_HOUR
    module.GUEST_DAILY_LIMIT_PER_VISITOR = module._real_guest_daily_limit


def strip_markers(text):
    """Chat replies may contain heartbeat bytes and the notice marker."""
    return text.replace("\x00", "").replace("\x02", "")


@pytest.fixture()
def unverified_client(app_module):
    """A brand-new guest who hasn't answered the age question yet."""
    return app_module.app.test_client()


@pytest.fixture()
def client(app_module):
    """An anonymous guest who has already answered the age question (13+),
    which every guest has to do before chatting."""
    c = app_module.app.test_client()
    assert c.post("/api/guest-age", json={"age": 14}).get_json()["status"] == "ok"
    return c


@pytest.fixture()
def admin_client(client):
    """A test client that's already logged in as the admin."""
    res = client.post("/api/admin/login", json={"password": TEST_ADMIN_PASSWORD})
    assert res.status_code == 200, "Admin login fixture failed — check ADMIN password wiring"
    return client


_NO_EMAIL = object()


def make_account(client, name, grade, password, is13=True, parent_email="", recovery_email=_NO_EMAIL):
    """Convenience helper: signs up an account via the real /api/signup
    route (so every test exercises real validation) and returns the
    response. For under-13 signups, the account is created but stays
    pending — callers that need it active should use make_and_activate_account
    or grant_consent() below.

    Every 13+ account needs an email, so one is made up unless the test
    passes its own (pass recovery_email="" to test leaving it out)."""
    if recovery_email is _NO_EMAIL:
        recovery_email = ("student-" + "".join(c for c in name.lower() if c.isalnum())
                          + "@example.com") if is13 else ""
    return client.post("/api/signup", json={
        "name": name,
        "grade": grade,
        "password": password,
        "is13OrOlder": is13,
        "parentEmail": parent_email,
        "recoveryEmail": recovery_email,
    })


def grant_consent(app_module, client, name):
    """Finds the pending consent token for `name` directly from disk (the
    same shortcut a parent's email link would provide) and completes the
    two-step approve flow, exactly like clicking the emailed link and then
    the confirm button."""
    key = " ".join(name.split()).strip().lower()
    tokens = app_module._load_consent_tokens()
    token = next((t for t, v in tokens.items() if v.get("user_key") == key), None)
    assert token, f"No pending consent token found for {name!r}"
    res = client.post(f"/api/consent/{token}")
    assert res.status_code == 200
    return token
