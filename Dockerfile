FROM python:3.10-slim

# System deps for xgboost/catboost builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only backend files (no frontend needed on HF)
COPY app.py .
COPY src/ ./src/
COPY artifacts/ ./artifacts/

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2"]
