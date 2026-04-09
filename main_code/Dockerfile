FROM python:3.11-slim

WORKDIR /app

# Keep the SQLite database on a writable path that can be bind-mounted.
ENV COSMIKAI_DB_PATH=/data/stars_cache.db

# System packages commonly needed by scientific Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker build caching
COPY requirements.txt /app/requirements.txt

# Install dependencies required by main_code inference + API server
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only main_code server pipeline
COPY . /app/

RUN mkdir -p /data

EXPOSE 8000

CMD ["uvicorn", "server_implementation:app", "--host", "0.0.0.0", "--port", "8000"]
