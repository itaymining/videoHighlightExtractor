# 🎬 Video Highlighter

AI-powered tool that scans a library of videos, picks the most interesting moments, and stitches them into a combined highlight reel. Optionally boosts segments that contain a specific person's face.

The repo also ships a few **utility scripts** for getting videos into your library in the first place (Google Photos, Telegram) and for fixing slow-motion footage.

---

## How it works

For every video in your input directories, the highlighter:

1. **Detects scenes** via frame-difference analysis.
2. **Scores each segment** by motion (optical flow) + visual interestingness (ResNet50) + optional known-face presence.
3. **Saves the top highlights** per video into `individual_highlights/`.
4. **Concatenates** the best-scoring clips up to a target duration into `combined_highlight_reel.mp4`.

---

## Install

Requires **Python 3.9+** and **FFmpeg** installed on your system.

```bash
# (recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

> **Face recognition** (`face-recognition`, `dlib`) needs `cmake` and a C++ build toolchain.
> If you don't need that feature, you can skip those two lines in `requirements.txt`.

---

## Usage — highlight generator

The main entry point is `highlight.py`. Point it at one or more folders of videos:

```bash
python highlight.py --input-dir ./my_videos --output-dir ./out
```

Multiple input folders, custom reel length:

```bash
python highlight.py -i ./trip2023 -i ./trip2024 --target-duration 300
```

With face recognition (put reference `*.jpg` images in a folder):

```bash
python highlight.py -i ./videos --use-face-recognition --face-images-dir ./faces
```

See every option:

```bash
python highlight.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --input-dir` | *(required)* | Input video folder (repeat for several) |
| `-o, --output-dir` | `video_highlights` | Where outputs are written |
| `--target-duration` | `300` | Combined reel length (seconds) |
| `--min-clip-duration` / `--max-clip-duration` | `2` / `3` | Per-clip bounds (seconds) |
| `--highlights-per-video` | `2` | Top clips saved per source video |
| `--max-video-duration` | `120` | Skip videos longer than this (seconds) |
| `--processing-order` | `asc` | `asc` or `desc` |
| `--shuffle` | off | Shuffle clips instead of keeping order |
| `--use-face-recognition` | off | Boost segments with known faces |
| `--face-images-dir` | `faces` | Reference face images (`*.jpg`) |
| `--has-face-only` | off | Keep only segments with a known face |
| `--from-existing-clips` | off | Build reel from clips already in output dir |

---

## Utility scripts

These live in `scripts/` and are optional helpers for sourcing videos.

### Google Photos downloader

Downloads videos from a Google Photos album within a date range.

```bash
python scripts/download_google_photos.py \
    --client-secret credentials/sa1.json \
    --token credentials/token.json \
    --output-dir ./downloads \
    --start-date 2024-01-01 --end-date 2024-12-31
```

Omit `--album-id` / dates to be prompted interactively. On first run a browser opens for OAuth; the token is cached to `credentials/token.json`.

### Telegram channel downloader

```bash
python scripts/download_telegram.py --channel @somechannel \
    --start-date 2023-10-07 --end-date 2024-11-01 \
    --max-duration 30 --output-dir ./downloads
```

### Slow-motion → normal speed

```bash
python scripts/convert_slowmo.py -i ./slowmo -o ./normal --target-fps 30
```

---

## Credentials & configuration

**No secrets are committed.** Copy the template and fill in your own values:

```bash
cp .env.example .env
```

| Variable | Used by | Notes |
|----------|---------|-------|
| `GOOGLE_CLIENT_SECRET` | Google Photos | Path to OAuth client secret JSON |
| `GOOGLE_TOKEN_FILE` | Google Photos | Cached token path (auto-created) |
| `TELEGRAM_API_ID` | Telegram | From <https://my.telegram.org> |
| `TELEGRAM_API_HASH` | Telegram | From <https://my.telegram.org> |

- `.env`, the `credentials/` folder, Telegram `*.session` files, and all generated media are **gitignored**.
- Put your Google OAuth JSON files in `credentials/` (e.g. `credentials/sa1.json`).
- CLI flags always override `.env` values.

---

## Project layout

```
video-highlighter/
├── highlight.py                  # Runnable entry point for the highlighter
├── src/video_highlighter/
│   ├── highlighter.py            # VideoHighlightGenerator (core logic)
│   ├── face_recognizer.py        # Known-face matching
│   └── cli.py                    # argparse CLI
├── scripts/
│   ├── download_google_photos.py
│   ├── download_telegram.py
│   └── convert_slowmo.py
├── credentials/                  # OAuth JSON + tokens (gitignored)
├── .env.example
├── requirements.txt
└── pyproject.toml
```

Install as a package to get a `video-highlighter` command:

```bash
pip install -e .
video-highlighter --input-dir ./videos
```

---

## License

MIT
