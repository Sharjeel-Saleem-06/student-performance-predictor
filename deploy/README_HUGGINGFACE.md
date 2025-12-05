# Deploy Backend to Hugging Face Spaces

This guide explains how to deploy the ML backend API to Hugging Face Spaces (free, no credit card required).

## Prerequisites
- Hugging Face account (free): https://huggingface.co/join

## Steps

### 1. Create a New Space
1. Go to https://huggingface.co/new-space
2. Enter a name (e.g., `student-performance`)
3. Select **Docker** as the SDK
4. Choose **Public** visibility (required for free tier)
5. Click **Create Space**

### 2. Clone the Space Repository
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/student-performance
cd student-performance
```

### 3. Copy Backend Files
Copy these files/folders from your project:
- `app.py` (API-only Flask app)
- `Dockerfile`
- `requirements.txt`
- `src/` folder
- `artifacts/` folder (contains model.pkl and proprocessor.pkl)

### 4. Push to Hugging Face
```bash
git add .
git commit -m "Initial deployment"
git push
```

### 5. Wait for Build
- Hugging Face will automatically build and deploy
- Watch the "Building" status in your Space
- Once green, your API is live!

## API Endpoints

Your Space URL will be: `https://YOUR_USERNAME-student-performance.hf.space`

### Health Check
```
GET /
```
Returns API status and available endpoints.

### Prediction
```
POST /api/predict
Content-Type: application/json

{
    "gender": "male",
    "ethnicity": "group B",
    "parental_level_of_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "reading_score": 72,
    "writing_score": 74
}
```

Response:
```json
{
    "success": true,
    "prediction": 73.5,
    "input": { ... }
}
```

## After Deployment

Update the frontend (`frontend/predict.html`) with your Space URL:
```javascript
const API_URL = 'https://YOUR_USERNAME-student-performance.hf.space';
```

Then deploy the frontend to Netlify!
