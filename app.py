"""AI Anime Studio — local-first FastAPI app: premise -> episode script -> animated storyboard MP4."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from characters import save_portraits
from story import generate_script, ollama_status
from video import render_episode

BASE = Path(__file__).resolve().parent
OUTPUTS = BASE / "outputs"
STATIC = BASE / "static"
OUTPUTS.mkdir(exist_ok=True)

app = FastAPI(title="AI Anime Studio", version="0.2.0")
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_render_lock = threading.Lock()  # video encoding is CPU-heavy; render one episode at a time
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


class CreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    premise: str = Field(..., min_length=5, max_length=1200)
    genre: str = Field("Demon-hunting swordsman", max_length=60)
    scenes: int = Field(6, ge=3, le=12)
    style: str = Field("Taisho ink action (Demon Slayer-inspired)", max_length=60)
    scene_length: Literal["short", "standard", "long"] = "long"
    use_ai: bool = True


def _project_id(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "episode"
    return f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _run(req: CreateRequest, update=lambda **kw: None) -> dict:
    started = time.perf_counter()
    update(stage="Writing the script…", progress=0.02)
    script, source, note = generate_script(req)
    project_id = _project_id(req.title)
    script_path = OUTPUTS / f"{project_id}_script.json"
    video_path = OUTPUTS / f"{project_id}_episode.mp4"

    context = f"{req.genre} {req.style}"
    for char, fname in zip(script["characters"], save_portraits(script["characters"], OUTPUTS, project_id, context)):
        char["portrait_url"] = f"/outputs/{fname}"

    script_doc = {"project_id": project_id, "source": source, "request": req.model_dump(), **script}
    script_path.write_text(json.dumps(script_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    update(stage="Waiting for the renderer…", progress=0.05)
    with _render_lock:
        update(stage="Drawing characters and rendering scenes…", progress=0.06)
        try:
            duration = render_episode(script, video_path,
                                      progress=lambda p: update(stage="Rendering video…", progress=0.06 + 0.94 * p))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Video rendering failed: {exc}") from exc

    return {
        "project_id": project_id,
        "source": source,
        "note": note,
        "script": script,
        "script_url": f"/outputs/{script_path.name}",
        "video_url": f"/outputs/{video_path.name}",
        "duration_seconds": round(duration, 1),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "ollama": ollama_status()}


@app.post("/api/create")
def create(req: CreateRequest):
    """Synchronous: returns when the video is finished."""
    return _run(req)


@app.post("/api/jobs")
def create_job(req: CreateRequest):
    """Start a render in the background; poll GET /api/jobs/{id} for progress."""
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "status": "running", "stage": "Queued…", "progress": 0.0, "result": None, "error": None}
    with _jobs_lock:
        _jobs[job_id] = job

    def update(**kw):
        with _jobs_lock:
            job.update(kw)

    def worker():
        try:
            result = _run(req, update)
            update(status="done", progress=1.0, stage="Done", result=result)
        except HTTPException as exc:
            update(status="error", error=exc.detail)
        except Exception as exc:  # noqa: BLE001
            update(status="error", error=f"{type(exc).__name__}: {exc}")

    threading.Thread(target=worker, daemon=True).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return dict(job)
