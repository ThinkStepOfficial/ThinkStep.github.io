# ThinkStep — running the test suite

This checks that the app's code *behaves the way it's supposed to* — signups,
the parental consent flow, password reset, account deletion, the admin
dashboard, safety-flag alerting, and more. Like the old `compliance_test.py`,
it does **not** and **cannot** certify that ThinkStep is legally compliant —
that's a question for a real lawyer, not a test script.

## One-time setup

```
pip install -r requirements-test.txt
```

(Node.js is optional — only needed for the frontend JS syntax checks. If it's
not installed, those specific checks are skipped automatically, everything
else still runs.)

## Running it

From the same folder as `app.py`:

```
pytest
```

That's it. It'll run every test and print a summary. A few useful variations:

```
pytest -v                        # verbose — show every individual test name
pytest tests/test_admin_dashboard.py   # just one file
pytest -k "safety"                # just tests with "safety" in the name
pytest -x                         # stop at the first failure instead of running everything
```

## Is it safe to run? Will it touch my real data?

Yes, safe. Every test gets its own throwaway copy of the app pointed at an
empty temp folder — your real `users.json`, `feedback.json`, etc. next to
`app.py` are never opened by these tests. No real email gets sent (there's no
real `email_config.json` in the temp folder) and no real server or network
port is used — tests call directly into the app in-process, so there's
nothing to accidentally leave running afterward.

## What's covered

- `test_pages.py` — Privacy Policy, Terms of Service, home page, favicon/OG
  tags, no-cache headers.
- `test_signup_and_consent.py` — signup validation, the under-13 parental
  consent flow (including the grade-vs-checkbox loophole fix, the GET-request
  auto-approve vulnerability fix, and XSS escaping).
- `test_login_security.py` — wrong passwords, and the brute-force lockout.
- `test_password_reset.py` — the forgot-password flow, including the
  no-account-enumeration guarantee.
- `test_account_deletion.py` — self-service account deletion.
- `test_feedback.py` — the feedback form's validation and local-save
  guarantee.
- `test_admin_dashboard.py` — the private `/admin` view: login, lockout,
  who's online, the full account roster (including offline accounts), and
  the summary stats.
- `test_safety_flags.py` — the keyword-based safety net that flags and
  logs concerning messages.
- `test_frontend_js.py` — a basic syntax check on every page's JavaScript,
  so a typo doesn't silently break a whole page for every visitor.

## What this suite does NOT cover

It doesn't click through the actual UI in a real browser (no Playwright/
Selenium) — so a button that's wired to the wrong element, or a CSS layout
bug, wouldn't be caught here. It also can't test real email delivery or a
real Ollama response, since those need actual network access this suite
deliberately avoids. If either of those becomes worth the extra setup, both
are addable later.

## The old compliance_test.py

Still there, still works the same way it always did (run it against a live
`python3 app.py` on a throwaway data copy). This new suite is the faster,
more reliable way to check things going forward, but nothing was deleted.
