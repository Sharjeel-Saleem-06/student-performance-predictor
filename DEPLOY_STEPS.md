# Deployment Guide (Hugging Face API + Netlify UI)

This project is split into:
- **Backend API**: Flask + model artifacts on **Hugging Face Spaces (Docker)**
- **Frontend UI**: Static HTML/CSS/JS on **Netlify**

Use the same approach for any similar project by following the steps below.

---

## Part A: Hugging Face (Backend API)

### Files you need to upload
- `app.py` (API-only Flask app)
- `Dockerfile` (container build)
- `requirements.txt`
- `src/` (pipeline code)
- `artifacts/` (model.pkl, proprocessor.pkl, data/test/train if needed)

### One-time setup (CLI)
```bash
# Install HF CLI
curl -LsSf https://hf.co/cli/install.sh | bash
# Login (will open browser)
hf login
```

### Create or clone the Space
```bash
# Option 1: Create via web, then clone
# https://huggingface.co/new-space -> Type: Docker, Visibility: Public
git clone https://huggingface.co/spaces/<username>/<space-name>

# Option 2: If already created, just clone
git clone https://huggingface.co/spaces/<username>/<space-name>
cd <space-name>
```

### Copy your backend files into the Space repo
```bash
cp /path/to/your/project/app.py .
cp /path/to/your/project/Dockerfile .
cp /path/to/your/project/requirements.txt .
cp -r /path/to/your/project/src .
cp -r /path/to/your/project/artifacts .
```

### Commit and push
```bash
git add -A
git commit -m "Add backend API"
git push
```

### What happens
- Pushing to the Space triggers a build
- Hugging Face builds the Docker image and runs `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2`
- Your API becomes available at `https://<username>-<space-name>.hf.space`

### API endpoints in this project
- `GET /` → health/info
- `POST /api/predict` (JSON body with gender, ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score)

---

## Part B: Netlify (Frontend UI)

### Files you need to upload
- `frontend/index.html`
- `frontend/predict.html`
- (Optional) `frontend/README.md`

### Configure the API URL in the frontend
Edit `frontend/predict.html` and set:
```javascript
const API_URL = 'https://<username>-<space-name>.hf.space';
```

### Deploy via drag-and-drop (easiest)
1) Go to https://app.netlify.com/ → Add new site → Deploy manually  
2) Drag the `frontend/` folder onto Netlify  
3) Done. Netlify hosts static files; no build command needed.

### Deploy via Git connect
1) Push the repo to GitHub/GitLab/Bitbucket  
2) In Netlify: Add new site → Import existing project  
3) Base directory: `frontend`  
4) Publish directory: `frontend`  
5) Build command: _leave empty_ (static)  
6) Deploy

---

## How I deployed this project (recap)
1) Cloned the HF Space you provided:  
   `git clone https://huggingface.co/spaces/sharry121/student-performance /tmp/hf-space`
2) Copied backend files (app.py, Dockerfile, requirements.txt, src/, artifacts/) into the Space repo.
3) Committed and pushed:  
   `git add -A && git commit -m "Add ML backend API" && git push`
4) Updated frontend API URL in `frontend/predict.html` to `https://sharry121-student-performance.hf.space`.
5) (Pending for you) Deploy `frontend/` to Netlify via drag-and-drop or Git connect.

---

## Checklist for another project
- Backend: app.py, Dockerfile, requirements.txt, src/, artifacts/ → push to HF Space
- Frontend: static HTML/JS → set API_URL to HF Space URL → deploy to Netlify
- Ensure CORS enabled on backend (flask-cors is included)
- Hugging Face Space must be Public for free tier
- Netlify: static deploy, no backend code needed


