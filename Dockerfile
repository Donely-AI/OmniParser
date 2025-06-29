FROM runpod/base:0.6.3-cuda11.8.0

# Update package lists and install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 from deadsnakes PPA
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-pip \
    python3.12-venv \
    python3.12-dev \
    python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install huggingface-cli for downloading models
RUN python -m pip install --upgrade pip && \
    python -m pip install huggingface_hub[cli]

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN python -m pip install --upgrade -r requirements.txt --no-cache-dir

# Copy all necessary project files
COPY . /app/

# Create weights directory and download model checkpoints
RUN mkdir -p weights && \
    for f in icon_detect/train_args.yaml icon_detect/model.pt icon_detect/model.yaml icon_caption/config.json icon_caption/generation_config.json icon_caption/model.safetensors; do \
        huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; \
    done

# Move icon_caption to icon_caption_florence as required
RUN mv weights/icon_caption weights/icon_caption_florence

# Expose port for gradio demo (if needed)
EXPOSE 7860

# Run the handler
CMD python -u /app/handler.py