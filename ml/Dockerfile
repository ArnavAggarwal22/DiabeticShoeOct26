# SmartSole ML service. Build context = repository root:
#   docker build -f ml/Dockerfile .
FROM python:3.12-slim

WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SMARTSOLE_DATA_DIR=/data \
    PORT=8000

COPY ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/app ./app
# Pre-trained starter model: activated on first boot, so no training is needed at start-up.
COPY ml/models ./models

# Mount persistent storage at /data (Railway volume / docker volume) so the database and
# trained models survive restarts and redeploys.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
