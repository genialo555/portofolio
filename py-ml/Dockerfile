FROM python:3.10-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configuration du répertoire de travail
WORKDIR /app

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Variables d'environnement
ENV PYTHONPATH=/app
ENV PORT=8000
ENV MODEL_CACHE_DIR=/app/model_cache
ENV EXPERIMENT_TRACKING_URI=sqlite:///mlflow.db

# Exposition du port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "src.ml_service.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 