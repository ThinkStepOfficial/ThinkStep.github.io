"""Settings for running ThinkStep with gunicorn — a proper web server for
real visitors, instead of Flask's built-in test server.

On Render, set the Start Command to:
    gunicorn app:app -c gunicorn.conf.py

(`python3 app.py` still works too — handy on your own computer.)
"""
import os

# Render tells the app which port to listen on.
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# ONE process, many threads. It must stay one process: ThinkStep keeps
# some things in memory (who's online, spam limits, login lockouts) and
# protects account saves with an in-process lock. Two processes would each
# have their own copy and could overwrite each other's account changes.
workers = 1
worker_class = "gthread"
threads = 16  # up to 16 chats streaming at the same time

# A hard think plus a web search can take a while before the answer
# starts, so give requests plenty of time before gunicorn gives up.
timeout = 180
graceful_timeout = 30
keepalive = 5

accesslog = "-"   # request log -> Render's Logs tab
errorlog = "-"
loglevel = "info"


def post_worker_init(worker):
    # Start the backup thread and print setup warnings once the app is
    # loaded inside the worker (gunicorn never runs app.py's __main__ part).
    from app import on_server_start
    on_server_start()
