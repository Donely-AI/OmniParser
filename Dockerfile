FROM runpod/base:0.6.3-cuda11.8.0

# Update package lists and install basic dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Use existing Python version and ensure pip is up to date
RUN python3 -m pip install --upgrade pip

# Install huggingface-cli for downloading models
RUN python3 -m pip install huggingface_hub[cli]

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN python3 -m pip install --ignore-installed -r requirements.txt --no-cache-dir

# Copy all necessary project files
COPY . /app/

# Create weights directory and download model checkpoints
RUN mkdir -p weights && \
    for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do \
        huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; \
    done

# Move icon_caption to icon_caption_florence as required
RUN mv weights/icon_caption weights/icon_caption_florence

# Expose port for gradio demo (if needed)
EXPOSE 7860

# Run the handler
CMD python3 -u /app/handler.py 