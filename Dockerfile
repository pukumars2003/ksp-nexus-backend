FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (required for some ML packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Pre-install CPU-only PyTorch so we don't download 5GB of useless NVIDIA CUDA drivers
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Zoho Catalyst sets X_ZOHO_CATALYST_LISTEN_PORT
# Our main.py reads this environment variable and binds uvicorn to it!
CMD ["python", "main.py"]
