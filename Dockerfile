# Use an official PyTorch runtime as a parent image (with CUDA 12.1 for GPU support)
ARG PYTORCH_VERSION=2.1.2
ARG CUDA_VERSION=12.1
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-devel

# Set environment variables for better reproducibility
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies (required for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    make \
    cmake \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the entire repository (excluding unnecessary files)
COPY . .

# Set up the PYTHONPATH to include the 'tptt' package and 'scripts' directory
ENV PYTHONPATH=/app/tptt:/app/tptt/scripts:${PYTHONPATH}

# Default command: help message (override with `docker run <image> <command>`)
CMD ["echo", "To start training, run:", \
    "docker run -it --gpus all -v $(pwd)/data:/data tptt", \
    "python -m train --model_name meta-llama/Llama-3.2-1B --method delta_rule --mag_weight 0.5"]

# Label for better maintainability
LABEL maintainer="Fabien Furfaro <fabien.furfaro@gmail.com>" \
    description="TPTT: Transforming Pretrained Transformers into Titans" \
    version="0.1.0"
