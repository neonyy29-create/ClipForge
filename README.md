# ClipForge

ClipForge is a self-hosted video clip generator with transcription, highlight selection, captioning, and clip rendering.

## Deploy on Render

This repository includes a Dockerfile for Render. The container installs FFmpeg and Python dependencies, serves the Flask app with Gunicorn, and binds to Render's `PORT`.

### Required service environment variables

Set both of these in the Render service settings before making the service public:

- `CLIPFORGE_AUTH_USERNAME` — login username
- `CLIPFORGE_AUTH_PASSWORD` — a strong, private login password

The app refuses to start on Render if either is missing. Keep these values private and do not commit them to Git.

### Notes

- The service is configured to process one job at a time to reduce memory pressure.
- The web UI defaults to the Whisper `tiny` model to reduce memory use. Larger models need more memory and may fail on small instances.
- The free Render service has an ephemeral filesystem and may sleep when idle. Generated clips and projects can be lost on restarts or redeploys.
- Video processing is CPU- and memory-intensive. A small/free instance may run slowly or run out of memory for longer or higher-resolution videos.
- YouTube upload requires separate Google OAuth credentials and is not preconfigured.
- Only upload videos you have permission to process.

## Local use

Install Python 3.11+, FFmpeg, and the dependencies:

```bash
pip install -r requirements.txt
python clipforge.py --serve
```

Then open `http://localhost:5000`.
