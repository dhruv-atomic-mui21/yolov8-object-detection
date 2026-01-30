# YOLOv8 Object Detection Docker Image
# Multi-stage build for production deployment

# Stage 1: Base image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS base

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Builder
FROM base AS builder

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (including ultralytics)
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Production
FROM base AS production

# Copy installed packages from builder
COPY --from=builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Create directories for data and models
RUN mkdir -p data/raw data/processed models/weights results logs

# Download YOLOv8 nano weights on build (optional)
# RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: Run API server
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative commands:
# Run detection on image:
# docker run -v $(pwd)/images:/app/images yolov8 python main.py --source images/test.jpg
#
# Run with GPU:
# docker run --gpus all -v $(pwd)/images:/app/images yolov8 python main.py --source images/test.jpg
