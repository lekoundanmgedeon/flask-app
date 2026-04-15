# ── Dockerfile for Intel Image Classifier (Flask)
FROM python:3.12-slim

# Configuration pour éviter les invites interactives
ENV DEBIAN_FRONTEND=noninteractive

# Correction : libgl1 au lieu de libgl1-mesa-glx
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY . .

# Création des dossiers nécessaires
RUN mkdir -p models outputs

# Sécurité : Utilisateur non-root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Configuration du Port pour Render
ENV PORT=10000
EXPOSE $PORT

# Commande de lancement (Format Shell pour que $PORT soit interprété)
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 --preload app:app
