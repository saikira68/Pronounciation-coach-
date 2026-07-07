---
title: Livo Pronunciation Coach
emoji: 🗣️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Livo Pronunciation Coach

Upload **30–45 seconds** of English speech and get an overall pronunciation score plus the
specific words and sounds that need work. Built for the Livo AI SWE assessment.

- **Live demo:** deployed on Hugging Face Spaces (Docker). See the repo/PR description for the URL.
- **Architecture & DPDP compliance:** see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## How it works

1. `faster-whisper` transcribes the clip into words with timestamps.
2. A `wav2vec2` CTC **phoneme recogniser** produces the phonemes the speaker actually said.
3. `espeak-ng` provides the **canonical** phonemes for each recognised word.
4. A global phoneme alignment (Needleman–Wunsch, with allophonic equivalence classes) compares
   the two and produces a per-word accuracy score, which aggregates into the overall score.

No paid speech APIs; everything runs in-process. Audio is processed in memory and never stored.

## Run locally

```bash
pip install -r requirements.txt
sudo apt-get install -y espeak-ng ffmpeg   # system deps
uvicorn app.main:app --reload --port 7860
# open http://localhost:7860
```

Or with Docker:

```bash
docker build -t pronunciation-coach .
docker run -p 7860:7860 pronunciation-coach
```

## Configuration

| Env var          | Default                                     | Purpose                         |
| ---------------- | ------------------------------------------- | ------------------------------- |
| `WHISPER_MODEL`  | `base.en`                                   | faster-whisper model size       |
| `PHONEME_MODEL`  | `facebook/wav2vec2-lv-60-espeak-cv-ft`      | acoustic phoneme recogniser     |
| `MIN_DURATION`   | `30`                                        | min clip length (seconds)       |
| `MAX_DURATION`   | `45`                                        | max clip length (seconds)       |
