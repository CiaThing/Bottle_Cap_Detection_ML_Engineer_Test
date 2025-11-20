# Use Python 3.10 Slim for a lightweight base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install System Dependencies
# 'libgl1' is required for OpenCV on Debian-based images
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
COPY pyproject.toml .

RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU-only version first to significantly reduce image size
# This prevents downloading large CUDA/GPU binaries
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (Ultralytics will skip PyTorch installation as it detects the CPU version)
RUN pip install --no-cache-dir ultralytics opencv-python pyyaml

# 3. Copy Source Code & Assets
# Copying source code last leverages Docker cache for faster rebuilds
COPY bsort/ ./bsort/
COPY models/ ./models/
COPY settings.yaml .

# 4. Set Entrypoint
ENTRYPOINT ["python", "-m", "bsort.main"]