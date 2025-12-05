# Frontend for Netlify

This folder contains the static frontend files to deploy on Netlify.

## Files
- `index.html` - Landing page
- `predict.html` - Prediction form page

## Deploy to Netlify

### Option 1: Drag & Drop
1. Go to [Netlify](https://app.netlify.com/)
2. Drag this `frontend` folder to the deploy area
3. Done!

### Option 2: Connect Git
1. Push this repo to GitHub
2. In Netlify, click "Add new site" → "Import an existing project"
3. Connect your GitHub repo
4. Set **Base directory**: `frontend`
5. Set **Publish directory**: `frontend`
6. Deploy!

## ⚠️ IMPORTANT: Update API URL

After deploying the backend to Hugging Face Spaces, you MUST update the API URL in `predict.html`:

1. Open `predict.html`
2. Find this line near the bottom:
   ```javascript
   const API_URL = 'https://YOUR_HF_SPACE_URL.hf.space';
   ```
3. Replace `YOUR_HF_SPACE_URL` with your actual Hugging Face Space URL
   - Example: `https://sharjeel-student-performance.hf.space`

## Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│                     │         │                     │
│   Netlify (UI)      │  ──────▶│  Hugging Face (API) │
│                     │  POST   │                     │
│   - index.html      │ /api/   │   - Flask app       │
│   - predict.html    │ predict │   - ML model        │
│                     │         │   - Preprocessor    │
└─────────────────────┘         └─────────────────────┘
     Static Files                  Python Backend
```

The frontend makes API calls to the Hugging Face backend for predictions.

