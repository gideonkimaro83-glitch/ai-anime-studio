# AI Anime Studio — free local MVP

A local-first web prototype that turns an anime premise into a structured episode script and a downloadable MP4 storyboard preview. It can use a local Ollama language model to write the script. If Ollama is not installed, it uses a built-in fallback story so the app remains testable.

## What works today

- Browser-based episode setup form with a live progress bar
- Local LLM integration through Ollama's HTTP API (falls back to a built-in story template)
- Two signature looks, used as *style inspiration only* (all characters, names and techniques are original):
  - **Taisho ink action** (Demon Slayer-inspired): swordsmen in patterned haori (wave, hemp-leaf, tortoiseshell,
    checkered, striped), katanas with element-tinted blades, ukiyo-e style water/flame/thunder technique trails,
    misty woodblock-style skies with paper grain, mountain forests, bamboo groves, shrine gates, wisteria
  - **Bright medieval** (Seven Deadly Sins-inspired): knights in plate armor with capes, mages in robes with
    glowing staffs, spear users, saturated skies with fluffy clouds, castles, taverns, villages
- Full-body procedural characters designed from each character's appearance text, consistent across scenes
- Movement: characters walk on stage, idle-sway, blink, step forward and gesture while speaking (mouth moves),
  and listeners dim; the camera eases toward whoever is talking
- Fight shots: characters charge, wind up, slash with an elemental trail, block, get knocked back; impact flashes,
  speed lines, screen shake, and a named-technique callout (e.g. "TIDE STYLE · THIRD FORM — Rising Current")
- Scene length option: **Long** (~40–60 s per scene, up to 3 fights), **Standard** (~20–30 s), **Short** (~10–15 s)
- Synthesized ambient score per scene mood, plus swoosh and impact sound effects in fights
- Cast introduction card and a full-body portrait PNG per character
- Downloadable MP4 and JSON script; no paid API key required; FFmpeg is bundled via `imageio-ffmpeg`

A 6-scene Long episode is about 4–5 minutes of video and takes roughly as long to render on a laptop CPU.

## Important limitations

Characters are simple procedural drawings with pose-based (cut-out) animation, not hand-drawn or AI-generated
anime footage. There is no voice acting: dialogue is shown as subtitles. Do not represent the output as a
finished studio-quality anime episode.

## Project layout

- `app.py` — FastAPI server and API routes
- `story.py` — Ollama script generation, output normalization, built-in fallback story
- `characters.py` — character designs, heads, full-body posed figures, technique names
- `fx.py` — technique trails, impact flashes, speed lines, callouts
- `textutil.py` — fonts and text wrapping
- `video.py` — frame rendering (Pillow), ambient audio synthesis (NumPy), MP4 encoding (FFmpeg)
- `static/index.html` — browser UI
- `outputs/` — generated scripts and videos

## Run on Windows

Open PowerShell in the `ai-anime-studio` folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 in your browser.

If PowerShell blocks activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Optional: enable local AI script writing with Ollama

1. Install Ollama from https://ollama.com/download
2. Download a small model: `ollama pull qwen2.5:3b`
3. Make sure Ollama is running. The badge in the top-right of the app turns green when it is ready.

Environment variables (set before starting the server):

| Variable | Default |
|---|---|
| `OLLAMA_MODEL` | `qwen2.5:3b` |
| `OLLAMA_URL` | `http://127.0.0.1:11434` |
| `OLLAMA_TIMEOUT` | `300` (seconds) |

If generation fails or the model is missing, the app automatically falls back to the built-in template and says so in the result.

## API

- `GET /api/health` — checks app and Ollama availability
- `POST /api/create` — generates a script and exports an MP4 (waits until done)
- `POST /api/jobs` — same, but runs in the background and returns a job id
- `GET /api/jobs/{id}` — job progress (`stage`, `progress` 0–1) and the result when done

```json
{
  "title": "The Starbound Promise",
  "premise": "A shy student discovers a power that protects their town.",
  "genre": "Demon-hunting swordsman",
  "scenes": 6,
  "style": "Taisho ink action (Demon Slayer-inspired)",
  "scene_length": "long",
  "use_ai": true
}
```

## Suggested next milestones

1. Optional image generation via a local ComfyUI server.
2. Character reference sheets reused across scenes for consistency.
3. Local text-to-speech (e.g. Piper) with per-character voices.
4. Short video-clip generation, assembled with FFmpeg.
5. Background jobs with progress, auth, storage quotas, and safety checks before public deployment.
