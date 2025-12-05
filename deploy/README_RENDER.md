## Render Deployment (Option A)

This project is set up to run as a Flask app with Gunicorn. The saved model artifacts are bundled in `artifacts/` and are loaded at runtime for predictions.

### What’s included
- `Procfile` with the Gunicorn command: `web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
- `runtime.txt` specifying Python 3.10.14
- `requirements.txt` with all runtime dependencies
- Model artifacts in `artifacts/` (`model.pkl`, `proprocessor.pkl`, etc.)

### Quick deploy on Render
1) Push the latest code to your repository.
2) Create a new **Web Service** on Render and connect the repo.
3) Set **Environment** to `Python 3`.
4) Render will auto-detect the **Build Command** (`pip install -r requirements.txt`).
5) Render will auto-detect the **Start Command** from `Procfile`. If needed, set it explicitly to:
   ```
   gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
   ```
6) Ensure the **Root Directory** is the repo root (where `Procfile` lives).
7) Deploy. The service will listen on the port Render provides via `$PORT`.

### Notes
- If you retrain the model, commit updated files in `artifacts/` so the service uses the latest artifacts.
- The Flask app serves `/` and `/predictdata`; templates are in `templates/`.

