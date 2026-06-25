# Wave Vision

Wave Vision is a Streamlit demo app for fixed-camera surf video analysis. It uses
classical OpenCV, draggable calibration overlays, Open-Meteo weather data, and an
optional OpenAI-written report grounded in computed metrics.

## What it does

- Upload a `.mp4`, `.mov`, or `.avi` beach video.
- Mark the horizon, shoreline, surf-zone ROI, and scale reference.
- Detect whitewater-like wave candidates with OpenCV.
- Estimate wave count, height, velocity, confidence, and set intervals.
- Fetch current wind and weather from Open-Meteo.
- Generate a surf report with template text or OpenAI when `OPENAI_API_KEY` is set.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

OpenAI is optional:

```bash
export OPENAI_API_KEY="your-key"
```

For local tests, install the development requirements:

```bash
pip install -r requirements-dev.txt
pytest
```

## Deploy

Live demo: https://ssmitty-wave-tracker-app-ch6ihf.streamlit.app/

The app is deployed on Streamlit Community Cloud.

- Repository: `ssmitty/wave-tracker`
- Branch: `main`
- Entry point: `app.py`
- Python version: `3.11`
- Optional secret: `OPENAI_API_KEY`

Streamlit Cloud installs production dependencies from `requirements.txt`. Keep
test and lint tools in `requirements-dev.txt` so the public app build stays
small.

## MVP notes

This version is built for one fixed camera angle and should be tuned once real
beach footage is available. Until then, the CV pipeline is intentionally
conservative and labels outputs as estimates.
