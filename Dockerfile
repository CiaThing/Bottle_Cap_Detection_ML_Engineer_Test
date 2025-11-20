# Use Python 3.10 Slim (Lightweight, Stable, & Recommended)
FROM python:3.10-slim

# Prevent Python from writing .pyc files (saves space)
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure Python output is sent straight to terminal (important for debugging Docker)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install System Dependencies
# OpenCV requires 'libgl1' and 'libglib2.0-0' on slim versions
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
# Copy project definition file first to leverage Docker cache
COPY pyproject.toml .

# Install libraries. Add --no-cache-dir to keep the image size SMALL
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ultralytics opencv-python pyyaml

# 3. Copy Source Code & Assets
# Do this last because code changes most frequently
COPY bsort/ ./bsort/
COPY models/ ./models/
COPY settings.yaml .

# 4. Entrypoint
ENTRYPOINT ["python", "-m", "bsort.main"]
