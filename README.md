# ThinkStep — local setup

A gentle AI tutor for kids that runs fully on your own machine via Ollama.

## 1. Make sure Ollama is running with your models pulled

```
ollama pull llama3.1:8b
ollama pull gemma3:4b
ollama pull qwen2.5:7b
ollama pull qwen3:latest
```

Ollama should be running in the background (it usually starts automatically,
or run `ollama serve`). It listens on `http://localhost:11434`.

## 2. Install Python dependencies

```
pip install -r requirements.txt
```

## 3. Run ThinkStep

```
python app.py
```

Then open **http://localhost:5000** in your browser.

## How it works

- `app.py` is a small Flask server. It serves the chat page and has one
  API route, `/api/chat`, which forwards your conversation to Ollama's
  local `/api/chat` endpoint and streams the response back.
- The system prompt (in `app.py`) instructs the model to guide kids with
  hints and questions instead of just handing over answers.
- The model dropdown in the top right lets you switch between
  `llama3.1:8b`, `gemma3:4b`, `qwen2.5:7b`, and `qwen3:latest` per message.
- Nothing leaves your computer — all requests go to `localhost:11434`.

## Customizing

- Edit `SYSTEM_PROMPT` in `app.py` to change ThinkStep's tutoring style
  (e.g. stricter Socratic-only, or allow more direct hints).
- Edit `ALLOWED_MODELS` in `app.py` if you add/remove Ollama models.
- Edit `static/index.html` for colors, wording, or layout.
