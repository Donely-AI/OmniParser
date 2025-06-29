FROM runpod/base:0.6.3-cuda11.8.0

# Install Python 3.12 and set it as default
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-pip python3.12-venv python3.12-dev && \
    ln -sf $(which python3.12) /usr/local/bin/python && \
    ln -sf $(which python3.12) /usr/local/bin/python3

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