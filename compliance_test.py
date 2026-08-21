"""
ThinkStep compliance test suite.

IMPORTANT: This checks that the CODE behaves the way it's supposed to —
it does NOT and CANNOT certify that ThinkStep is "legally compliant."
Whether these technical safeguards satisfy COPPA (or any other law) for
your specific situation is a judgment call for a real lawyer, not
something a test script can determine. Treat every PASS below as
"this safeguard works as designed," not "this is legal."

Run against a throwaway copy of the app (a fresh, empty users.json etc.)
— never against your real data.
"""

import json
import re
import time
import requests

BASE = "http://localhost:5000"
# Must match whatever password admin_config.json was set up with on this
# particular test copy of the app (see the setup commands used to run this
# suite) — the real production password is never written into this file.
ADMIN_PASSWORD = "95cca6db-988a-4cc3-93ba-0ad7356805e0"
results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail and status == "FAIL" else ""))


def main():
    # ---- Privacy Policy page ----
    r = requests.get(f"{BASE}/privacy", timeout=5)
    check("Privacy Policy page loads (200 OK)", r.status_code == 200, f"got {r.status_code}")
    body = r.text
    check("Privacy Policy mentions children/COPPA", "COPPA" in body or "children" in body.lower())
    check("Privacy Policy explains parental rights (review/delete)",
          "delete" in body.lower() and ("parent" in body.lower() or "guardian" in body.lower()))
    check("Privacy Policy states no data is sold",
          "sell" in body.lower() or "not sell" in body.lower())
    contact_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', body)
    check("Privacy Policy lists a contact email", bool(contact_match),
          "no email address found in page")

    # ---- Under-13 signup requires parent email ----
    r = requests.post(f"{BASE}/api/signup", json={
        "name": "NoParentEmail", "grade": "3", "password": "password123",
        "is13OrOlder": False, "parentEmail": ""
    }, timeout=5)
    check("Signup rejects under-13 without a parent email", r.status_code == 400)

    # ---- The grade-vs-checkbox loophole: a young grade + "I'm 13+"
    # checked (with no parent email given) must NOT bypass the gate. ----
    r = requests.post(f"{BASE}/api/signup", json={
        "name": f"SneakyGrade4{int(time.time())}", "grade": "4", "password": "password123",
        "is13OrOlder": True, "parentEmail": ""
    }, timeout=5)
    check("Grade 4 + 'I'm 13+' checked (no parent email) is REJECTED, not trusted",
          r.status_code == 400, f"got {r.status_code}: {r.text[:150]}")

    # ---- Full under-13 consent flow ----
    kid_name = f"TestKid{int(time.time())}"
    r = requests.post(f"{BASE}/api/signup", json={
        "name": kid_name, "grade": "4", "password": "password123",
        "is13OrOlder": False, "parentEmail": "parent@example.com"
    }, timeout=5)
    signup_json = r.json() if r.ok else {}
    check("Under-13 signup succeeds but is NOT immediately logged in",
          r.ok and signup_json.get("pendingConsent") is True)

    r = requests.post(f"{BASE}/api/login", json={"name": kid_name, "password": "password123"}, timeout=5)
    check("Login is BLOCKED before parental consent", r.status_code == 403)

    # Read the token directly from disk (isolated test copy only)
    try:
        with open("consent_tokens.json") as f:
            tokens = json.load(f)
        key = kid_name.strip().lower()
        token = next((t for t, v in tokens.items() if v.get("user_key") == key), None)
    except FileNotFoundError:
        token = None
    check("A consent token was generated for the pending account", token is not None)

    if token:
        # Visiting the link (GET) must NOT grant consent by itself — only
        # show a confirmation screen. This matters because email security
        # scanners auto-visit links to check for malware before a human
        # ever opens the message; if GET alone granted consent, a scanner
        # bot could silently "approve" an account with no parent involved.
        r = requests.get(f"{BASE}/api/consent/{token}", timeout=5)
        check("Visiting the consent link (GET) shows a confirm page, doesn't grant yet",
              r.status_code == 200 and "approve" in r.text.lower())

        r = requests.post(f"{BASE}/api/login", json={"name": kid_name, "password": "password123"}, timeout=5)
        check("Login still BLOCKED after only viewing the link (before clicking approve)",
              r.status_code == 403)

        # Now actually click "approve" (the POST the confirm page's button submits)
        r = requests.post(f"{BASE}/api/consent/{token}", timeout=5)
        check("Clicking 'approve' (POST) actually activates the account", r.status_code == 200)

        r = requests.post(f"{BASE}/api/consent/{token}", timeout=5)
        # Apostrophe is HTML-escaped (isn&#x27;t) by design — that's the
        # correct, safe behavior, so check for "already been used" instead
        # of relying on a literal apostrophe in the string.
        check("Reusing the same consent link fails (can't approve twice)",
              "already been used" in r.text.lower() or "expired" in r.text.lower())

        r = requests.post(f"{BASE}/api/login", json={"name": kid_name, "password": "password123"}, timeout=5)
        check("Login SUCCEEDS after parental consent is granted", r.status_code == 200)

    # ---- XSS: a script-like name must render literally, not execute ----
    xss_name = f"<script>alert(1)</script>Kid{int(time.time())}"
    r = requests.post(f"{BASE}/api/signup", json={
        "name": xss_name, "grade": "3", "password": "password123",
        "is13OrOlder": False, "parentEmail": "parent2@example.com"
    }, timeout=5)
    try:
        with open("consent_tokens.json") as f:
            tokens2 = json.load(f)
        xss_key = " ".join(xss_name.split()).strip().lower()
        xss_token = next((t for t, v in tokens2.items() if v.get("user_key") == xss_key), None)
    except FileNotFoundError:
        xss_token = None
    if xss_token:
        r = requests.get(f"{BASE}/api/consent/{xss_token}", timeout=5)
        check("A <script> tag in a name is escaped on the consent page, not executed",
              "<script>alert(1)</script>" not in r.text and "&lt;script&gt;" in r.text)

    # ---- Name length cap ----
    r = requests.post(f"{BASE}/api/signup", json={
        "name": "X" * 200, "grade": "9", "password": "password123", "is13OrOlder": True
    }, timeout=5)
    check("Signup rejects a name that's way too long", r.status_code == 400)

    # ---- Resend consent for a stuck-pending account (same name, still works) ----
    stuck_name = f"StuckKid{int(time.time())}"
    r1 = requests.post(f"{BASE}/api/signup", json={
        "name": stuck_name, "grade": "5", "password": "password123",
        "is13OrOlder": False, "parentEmail": "firstparent@example.com"
    }, timeout=5)
    r2 = requests.post(f"{BASE}/api/signup", json={
        "name": stuck_name, "grade": "5", "password": "newpassword123",
        "is13OrOlder": False, "parentEmail": "secondparent@example.com"
    }, timeout=5)
    check("Re-signing up with a name still stuck pending consent works (resend), doesn't block",
          r1.ok and r2.ok and r2.json().get("pendingConsent") is True)

    # ---- 13+ signup is unaffected (no gate) ----
    teen_name = f"TestTeen{int(time.time())}"
    r = requests.post(f"{BASE}/api/signup", json={
        "name": teen_name, "grade": "10", "password": "password123", "is13OrOlder": True
    }, timeout=5)
    check("13+ signup logs the user in immediately (no consent gate)",
          r.ok and "pendingConsent" not in r.json())

    # ---- Passwords are never stored in plain text ----
    try:
        with open("users.json") as f:
            users = json.load(f)
        no_plaintext = all(
            "password123" not in json.dumps(u) for u in users.values()
        )
        has_hash = all("password_hash" in u for u in users.values())
        check("No account stores the raw password anywhere", no_plaintext)
        check("Every account stores a password_hash, not a plain password", has_hash)
    except FileNotFoundError:
        check("users.json readable for password-storage check", False, "file not found")

    # ---- Feedback routing ----
    r = requests.post(f"{BASE}/api/feedback", json={
        "name": "Compliance Test", "stars": 5, "review": "Automated test review."
    }, timeout=5)
    check("Feedback submission is accepted", r.ok)
    check("Feedback response reports whether it was actually emailed",
          "emailed" in (r.json() if r.ok else {}))

    with open("app.py") as f:
        app_src = f.read()
    to_email_match = re.search(r'FEEDBACK_TO_EMAIL\s*=\s*"([^"]+)"', app_src)
    check("FEEDBACK_TO_EMAIL is set to a real-looking email address",
          bool(to_email_match) and "@" in (to_email_match.group(1) if to_email_match else ""),
          detail=to_email_match.group(1) if to_email_match else "not found")

    # ---- Terms of Service page ----
    r = requests.get(f"{BASE}/terms", timeout=5)
    check("Terms of Service page loads (200 OK)", r.status_code == 200, f"got {r.status_code}")

    # ---- Password reset: 13+ account WITH a recovery email ----
    teen2_name = f"TestTeenReset{int(time.time())}"
    r = requests.post(f"{BASE}/api/signup", json={
        "name": teen2_name, "grade": "10", "password": "originalpass1",
        "is13OrOlder": True, "recoveryEmail": "teenrecovery@example.com"
    }, timeout=5)
    check("13+ signup with an optional recovery email succeeds", r.ok)
    requests.post(f"{BASE}/api/logout", timeout=5)  # don't stay logged in as this test account

    r = requests.post(f"{BASE}/api/forgot-password", json={"name": teen2_name}, timeout=5)
    check("Forgot-password request for an account WITH a recovery email returns ok", r.ok)

    try:
        with open("reset_tokens.json") as f:
            reset_tokens = json.load(f)
        reset_key = teen2_name.strip().lower()
        reset_token = next((t for t, v in reset_tokens.items() if v.get("user_key") == reset_key), None)
    except FileNotFoundError:
        reset_token = None
    check("A reset token was generated for the account with a recovery email", reset_token is not None)

    if reset_token:
        r = requests.get(f"{BASE}/api/reset-password/{reset_token}", timeout=5)
        check("Visiting the reset link (GET) shows a form, doesn't change the password yet",
              r.status_code == 200 and "new password" in r.text.lower())

        r = requests.post(f"{BASE}/api/reset-password/{reset_token}",
                           data={"password": "brandnewpass2"}, timeout=5)
        check("Submitting the reset form (POST) actually changes the password", r.status_code == 200)

        r = requests.post(f"{BASE}/api/login", json={"name": teen2_name, "password": "originalpass1"}, timeout=5)
        check("Old password no longer works after reset", r.status_code == 401)

        r = requests.post(f"{BASE}/api/login", json={"name": teen2_name, "password": "brandnewpass2"}, timeout=5)
        check("New password works after reset", r.status_code == 200)

        r = requests.post(f"{BASE}/api/reset-password/{reset_token}",
                           data={"password": "shouldnotwork1"}, timeout=5)
        check("Reusing the same reset link fails (token consumed)",
              "isn't valid" in r.text.lower() or "expired" in r.text.lower())

    # ---- Forgot-password for an account with NO recovery email: same generic response (no enumeration leak) ----
    teen3_name = f"TestTeenNoRecovery{int(time.time())}"
    requests.post(f"{BASE}/api/signup", json={
        "name": teen3_name, "grade": "11", "password": "somepassword1", "is13OrOlder": True
    }, timeout=5)
    requests.post(f"{BASE}/api/logout", timeout=5)
    r_no_email = requests.post(f"{BASE}/api/forgot-password", json={"name": teen3_name}, timeout=5)
    r_nonexistent = requests.post(f"{BASE}/api/forgot-password", json={"name": "ThisAccountDoesNotExist"}, timeout=5)
    check("Forgot-password gives the SAME generic response for no-recovery-email and nonexistent accounts "
          "(doesn't leak which accounts exist)",
          r_no_email.ok and r_nonexistent.ok and r_no_email.json().get("message") == r_nonexistent.json().get("message"))

    # ---- Self-service account deletion ----
    del_name = f"TestDeleteMe{int(time.time())}"
    s = requests.Session()
    s.post(f"{BASE}/api/signup", json={
        "name": del_name, "grade": "12", "password": "deletepass1", "is13OrOlder": True
    }, timeout=5)
    r = s.post(f"{BASE}/api/delete-account", json={"password": "wrongpassword"}, timeout=5)
    check("Deleting an account with the WRONG password is rejected", r.status_code == 401)

    r = s.post(f"{BASE}/api/delete-account", json={"password": "deletepass1"}, timeout=5)
    check("Deleting an account with the correct password succeeds", r.ok)

    r = requests.post(f"{BASE}/api/login", json={"name": del_name, "password": "deletepass1"}, timeout=5)
    check("Logging in to a deleted account fails (it's really gone)", r.status_code == 401)

    # ---- Login brute-force lockout ----
    lock_name = f"TestLockout{int(time.time())}"
    requests.post(f"{BASE}/api/signup", json={
        "name": lock_name, "grade": "9", "password": "correctpass1", "is13OrOlder": True
    }, timeout=5)
    requests.post(f"{BASE}/api/logout", timeout=5)
    last_status = None
    for _ in range(8):
        last_status = requests.post(f"{BASE}/api/login", json={
            "name": lock_name, "password": "wrongpassword"
        }, timeout=5).status_code
    check("Repeated wrong passwords eventually lock the account out (429)", last_status == 429)

    r = requests.post(f"{BASE}/api/login", json={"name": lock_name, "password": "correctpass1"}, timeout=5)
    check("Account stays locked out even with the CORRECT password once locked", r.status_code == 429)

    # ---- Admin "who's online" view ----
    r = requests.get(f"{BASE}/api/admin/online", timeout=5)
    check("Admin online-status API is BLOCKED without logging in", r.status_code == 401)

    admin_s = requests.Session()
    r = admin_s.post(f"{BASE}/api/admin/login", json={"password": "definitely-wrong-password"}, timeout=5)
    check("Admin login rejects the wrong password", r.status_code == 401)

    r = admin_s.get(f"{BASE}/api/admin/online", timeout=5)
    check("Still blocked after a failed admin login attempt", r.status_code == 401)

    r = admin_s.post(f"{BASE}/api/admin/login", json={"password": ADMIN_PASSWORD}, timeout=5)
    check("Admin login succeeds with the correct password", r.ok, f"got {r.status_code}: {r.text[:150]}")

    r = admin_s.get(f"{BASE}/api/admin/online", timeout=5)
    check("Admin can view online status once logged in", r.ok)
    online_before = r.json() if r.ok else {}
    check("Online status response has the expected shape",
          all(k in online_before for k in ("totalOnline", "accounts", "guestCount")))

    # Someone (a guest) sends a heartbeat — should now show up in the guest count.
    guest_s = requests.Session()
    guest_s.post(f"{BASE}/api/heartbeat", timeout=5)
    r = admin_s.get(f"{BASE}/api/admin/online", timeout=5)
    online_after_guest = r.json() if r.ok else {}
    check("A guest heartbeat increases the guest online count",
          online_after_guest.get("guestCount", 0) >= online_before.get("guestCount", 0) + 1)

    # A logged-in student sends a heartbeat — should show up by name in the accounts list.
    heartbeat_name = f"TestOnline{int(time.time())}"
    acct_s = requests.Session()
    acct_s.post(f"{BASE}/api/signup", json={
        "name": heartbeat_name, "grade": "6", "password": "onlinepass1",
        "is13OrOlder": False, "parentEmail": "onlineparent@example.com"
    }, timeout=5)
    # Under-13 needs consent before login/heartbeat would reflect a real
    # session, so use a 13+ account instead for this check — simpler and
    # still proves the same mechanism.
    heartbeat_name2 = f"TestOnlineTeen{int(time.time())}"
    acct_s.post(f"{BASE}/api/signup", json={
        "name": heartbeat_name2, "grade": "11", "password": "onlinepass1", "is13OrOlder": True
    }, timeout=5)
    acct_s.post(f"{BASE}/api/heartbeat", timeout=5)
    r = admin_s.get(f"{BASE}/api/admin/online", timeout=5)
    online_after_acct = r.json() if r.ok else {}
    check("A signed-in student's heartbeat makes them show up by name in the admin view",
          any(a["name"] == heartbeat_name2 for a in online_after_acct.get("accounts", [])))

    # ---- Debug mode / production safety ----
    with open("app.py") as f:
        app_src2 = f.read()
    check("app.py does not hardcode debug=True (would enable the RCE-risk Werkzeug debugger)",
          "debug=True" not in app_src2)

    # ---- Safety-flag alerting ----
    concerning_msg = "I want to kill myself and I don't know what to do"
    r = requests.post(f"{BASE}/api/chat", json={
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": concerning_msg}],
    }, timeout=5)
    # Just draining the streamed response is enough to let the safety check run.
    try:
        for _ in r.iter_content(chunk_size=1024):
            pass
    except Exception:
        pass
    time.sleep(0.5)
    try:
        with open("safety_flags.json") as f:
            flags = json.load(f)
        flagged = any(concerning_msg in fl.get("message", "") for fl in flags)
    except FileNotFoundError:
        flagged = False
    check("A message with concerning language gets logged as a safety flag", flagged)

    r = admin_s.get(f"{BASE}/api/admin/safety-flags", timeout=5)
    check("Admin can view recent safety flags", r.ok and "flags" in r.json())

    r = requests.get(f"{BASE}/api/admin/safety-flags", timeout=5)
    check("Safety flags endpoint is blocked for non-admins", r.status_code == 401)

    # ---- Admin stats ----
    r = admin_s.get(f"{BASE}/api/admin/stats", timeout=5)
    check("Admin stats endpoint works", r.ok)
    stats = r.json() if r.ok else {}
    check("Admin stats includes total accounts and feedback counts",
          "totalAccounts" in stats and "totalFeedback" in stats)

    # ---- Favicon / OG assets exist ----
    for asset in ("/static/favicon.svg", "/static/favicon.png", "/static/og-image.png"):
        r = requests.get(f"{BASE}{asset}", timeout=5)
        check(f"{asset} is served (200 OK)", r.status_code == 200)

    r = requests.get(f"{BASE}/", timeout=5)
    check("Home page includes Open Graph tags for link previews", "og:title" in r.text and "og:image" in r.text)

    # ---- Admin "all accounts" view (online AND offline) ----
    r = requests.get(f"{BASE}/api/admin/accounts", timeout=5)
    check("All-accounts admin API is BLOCKED without logging in", r.status_code == 401)

    r = admin_s.get(f"{BASE}/api/admin/accounts", timeout=5)
    check("Admin can list all accounts", r.ok)
    all_accounts = r.json().get("accounts", []) if r.ok else []
    check("The signed-in student who just sent a heartbeat shows up as online with a last_seen time",
          any(a["name"] == heartbeat_name2 and a["online"] and a["last_seen"] for a in all_accounts))
    check("An under-13 account still awaiting consent (never logged in) also shows up, just offline",
          any(a["name"] == heartbeat_name and not a["online"] and a.get("consentStatus") == "pending"
              for a in all_accounts))

    # ---- Admin login brute-force lockout ----
    bad_s = requests.Session()
    last_admin_status = None
    for _ in range(9):
        last_admin_status = bad_s.post(f"{BASE}/api/admin/login", json={"password": "nope"}, timeout=5).status_code
    check("Repeated wrong admin passwords eventually lock out further attempts (429)", last_admin_status == 429)

    r = admin_s.get(f"{BASE}/api/admin/online", timeout=5)
    check("A different, already-logged-in admin session is unaffected by another session's lockout", r.ok)

    admin_s.post(f"{BASE}/api/admin/logout", timeout=5)
    r = admin_s.get(f"{BASE}/api/admin/online", timeout=5)
    check("Admin logout actually revokes access", r.status_code == 401)

    # ---- No-cache headers (so stale pages don't hide real state) ----
    r = requests.get(f"{BASE}/api/guest-status", timeout=5)
    check("API responses are marked no-store (no stale-data risk)",
          "no-store" in r.headers.get("Cache-Control", ""))

    # ---- Summary ----
    print("\n" + "=" * 50)
    passed = sum(1 for s, _, _ in results if s == "PASS")
    failed = sum(1 for s, _, _ in results if s == "FAIL")
    print(f"RESULT: {passed} passed, {failed} failed out of {len(results)} checks")
    if failed:
        print("\nFailed checks:")
        for s, name, detail in results:
            if s == "FAIL":
                print(f"  - {name}" + (f" ({detail})" if detail else ""))


if __name__ == "__main__":
    main()
