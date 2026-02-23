"""
HuggingFace Spaces entry point.
Starts FastAPI (uvicorn) in a background thread, then launches Gradio on port 7860.
HF Spaces runs this file automatically when sdk: gradio is set in README.md.
"""
import os
import threading
import time

import uvicorn


def start_fastapi():
    """Run FastAPI in a background thread on port 8000."""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="warning")


# ── Start FastAPI in background ───────────────────────────────────────────────
api_thread = threading.Thread(target=start_fastapi, daemon=True)
api_thread.start()

# Give FastAPI a moment to start before Gradio begins accepting requests
time.sleep(3)

# ── Point Gradio at the local FastAPI ─────────────────────────────────────────
os.environ.setdefault("BACKEND_URL", "http://localhost:8000")

# ── Launch Gradio ─────────────────────────────────────────────────────────────
from app.ui.gradio_app import create_app

demo = create_app()
demo.launch(server_name="0.0.0.0", server_port=7860)
