## Deploy Free on Hugging Face Spaces (Docker)

This app can run on Hugging Face Spaces without a card using the provided `Dockerfile`.

### Steps
1) Create a new Space on Hugging Face:
   - Type: **Docker**
   - Visibility: **Public** (recommended for free tier)
2) Push the repo contents (including `Dockerfile`, `app.py`, `requirements.txt`, and the `artifacts/` directory with `model.pkl` and `proprocessor.pkl`) to the Space:
   - If using Git: add the Space as a remote and push `main`
   - Or upload via the web UI
3) Hugging Face will build and run the container automatically.
   - The app listens on `$PORT` (default 7860) and uses `gunicorn app:app`.
4) Once the build succeeds, click “Visit Space” to use the app.

### Notes
- The build installs system deps (`gcc`, `libgomp1`) for `xgboost/catboost`.
- Keep `artifacts/` in the repo so the model loads at runtime.
- If you retrain, commit updated artifacts before pushing.

