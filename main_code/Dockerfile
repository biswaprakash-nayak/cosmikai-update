FROM python:3.11-slim

WORKDIR /app

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

EXPOSE 8000

CMD ["uvicorn", "server_implementation:app", "--host", "0.0.0.0", "--port", "8000"]
