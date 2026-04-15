# ── Dockerfile for Intel Image Classifier (Flask)
# CPU build by default; swap base image for CUDA builds.

FROM python:3.12

# Configuration pour éviter les invites interactives
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create outputs directory
RUN mkdir -p models outputs

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Render uses the PORT environment variable (defaults to 10000)
ENV PORT 10000
EXPOSE $PORT

# Gunicorn with 2 workers (increase for more RAM)
CMD gunicorn \
     --bind 0.0.0.0:$PORT \
     --workers 2 \
     --threads 4 \
     --timeout 120 \
     --preload \
     app:app
