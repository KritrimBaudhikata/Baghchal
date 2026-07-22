# Baghchal web app: API + static frontend. Run with a trained .keras model.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY baghchal.py .
COPY game_actions.py .
COPY config_loader.py .
COPY config.yaml .
COPY models/ models/
COPY training/ training/
COPY api/ api/

# Non-root user for production
RUN useradd --create-home --shell /bin/bash baghchal \
    && chown -R baghchal:baghchal /app
USER baghchal

# Default: expect model at /app/models/final_bagh_chal_model.keras (mount or copy)
ENV MODEL_PATH=/app/models/final_bagh_chal_model.keras

EXPOSE 8080

# Run from project root so imports work. Copy your trained model into models/ before build or mount at run.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]
