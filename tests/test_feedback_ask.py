"""Now and then the chat asks how ThinkStep is going (stars + optional
comment). It uses the normal feedback route, so feedback lands in the same
place as the Feedback page."""


def test_chat_page_asks_for_feedback_now_and_then(client):
    html = client.get("/app").get_data(as_text=True)
    assert "function maybeAskForFeedback" in html
    assert "maybeAskForFeedback();" in html          # runs after each answer
    assert "FB_GAP_MS" in html and "FB_AFTER_SENT_MS" in html  # never nags


def test_quick_rating_without_a_comment_is_accepted(client):
    res = client.post("/api/feedback", json={
        "name": "A ThinkStep student", "stars": 4,
        "review": "(Quick rating from the chat — no comment)"})
    assert res.status_code == 200 and res.get_json()["ok"]
