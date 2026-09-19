FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# Install system dependencies for numerical packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend source into /app/backend/ so 'from backend.xxx' imports work
COPY . ./backend/

# Ensure /app/ACV2 is populated and on PYTHONPATH for the ACV leak localisation module
RUN if [ -d "/app/backend/ACV2" ]; then cp -r /app/backend/ACV2 /app/ACV2; fi
ENV PYTHONPATH="/app:/app/ACV2:/app/backend/ACV2:${PYTHONPATH}"

EXPOSE 8080

# Run from /app so ROOT_DIR = /app and 'backend' is a valid package
CMD exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
