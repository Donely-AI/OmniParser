# OmniParser RunPod Serverless Handler

This handler implements the OmniParser functionality for RunPod serverless inference, allowing you to parse GUI screenshots into structured, actionable elements.

## Features

- **Icon Detection**: Automatically detects UI elements and icons in screenshots
- **OCR Processing**: Extracts text from images using PaddleOCR or EasyOCR
- **Element Captioning**: Generates descriptions for detected UI elements using Florence2
- **Structured Output**: Returns parsed content in JSON format with bounding boxes and metadata
- **Flexible Input**: Supports multiple image input formats (base64, file paths, data URLs)

## Setup

### 1. Install Dependencies

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Download Model Weights

Download the OmniParser V2 model weights:
```bash
# Download the model checkpoints to weights/ directory
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do 
    huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
done

# Rename the caption model directory
mv weights/icon_caption weights/icon_caption_florence
```

### 3. Directory Structure

Ensure your directory structure matches:
```
OmniParser/
├── handler.py
├── weights/
│   ├── icon_detect/
│   │   ├── model.pt
│   │   ├── model.yaml
│   │   └── train_args.yaml
│   └── icon_caption_florence/
│       ├── config.json
│       ├── generation_config.json
│       └── model.safetensors
└── util/
    └── utils.py
```

## Usage

### Input Format

The handler expects a job with the following input structure:

```json
{
  "input": {
    "image": "base64_encoded_image_or_path",
    "box_threshold": 0.05,
    "output_format": "detailed",
    "use_ocr": true,
    "ocr_engine": "paddleocr"
  }
}
```

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | string | **required** | Image data (base64, data URL, or file path) |
| `box_threshold` | float | 0.05 | Confidence threshold for element detection |
| `output_format` | string | "detailed" | Output format: "detailed" or "simple" |
| `use_ocr` | boolean | true | Whether to perform OCR text extraction |
| `ocr_engine` | string | "paddleocr" | OCR engine: "paddleocr" or "easyocr" |

### Image Input Formats

The handler supports multiple image input formats:

1. **Base64 encoded image**:
   ```json
   {"image": "/9j/4AAQSkZJRgABAQEAAAAA..."}
   ```

2. **Data URL**:
   ```json
   {"image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."}
   ```

3. **File path**:
   ```json
   {"image": "/path/to/image.png"}
   ```

### Output Format

#### Detailed Output (default)
```json
{
  "success": true,
  "annotated_image": "base64_encoded_annotated_image",
  "label_coordinates": {
    "0": [x, y, width, height],
    "1": [x, y, width, height]
  },
  "parsed_content": [
    {
      "type": "text",
      "content": "Button text",
      "bbox": [x1, y1, x2, y2],
      "interactivity": true,
      "source": "ocr"
    },
    {
      "type": "icon", 
      "content": "Search icon",
      "bbox": [x1, y1, x2, y2],
      "interactivity": true,
      "source": "icon_detection"
    }
  ],
  "structured_data": [...],
  "image_size": [width, height],
  "total_elements": 25
}
```

#### Simple Output
```json
{
  "success": true,
  "parsed_content": [
    {
      "type": "text",
      "content": "Button text", 
      "bbox": [x1, y1, x2, y2],
      "interactivity": true
    }
  ],
  "image_size": [width, height]
}
```

### Error Handling

If processing fails, the handler returns:
```json
{
  "success": false,
  "error": "Error description"
}
```

## Testing Locally

Use the provided test script to test the handler locally:

```bash
python test_handler.py
```

This will:
- Process a sample image
- Save the annotated result as `test_output_annotated.png`
- Save the JSON results as `test_output_results.json`

## RunPod Deployment

### 1. Build Docker Image

```bash
docker build -t omniparser-handler .
```

### 2. Deploy to RunPod

1. Upload your Docker image to a registry
2. Create a new RunPod serverless endpoint
3. Configure the endpoint with your image
4. Set appropriate GPU requirements (recommended: RTX 4090 or better)

### 3. API Usage

Once deployed, you can call your endpoint:

```python
import requests
import base64

# Load and encode image
with open("screenshot.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Make API request
response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "input": {
            "image": image_data,
            "box_threshold": 0.05,
            "output_format": "detailed"
        }
    }
)

result = response.json()
```

## Performance Notes

- **GPU Requirements**: CUDA-compatible GPU recommended (RTX 4090 or better)
- **Memory Usage**: ~4GB GPU memory for default batch size
- **Processing Time**: Varies by image size and complexity (typically 2-10 seconds)
- **Batch Size**: Adjust `batch_size` parameter in `get_som_labeled_img` call to optimize for your GPU

## Troubleshooting

### Common Issues

1. **Model weights not found**: Ensure weights are downloaded to correct paths
2. **CUDA out of memory**: Reduce batch size or use smaller images
3. **OCR errors**: Try switching between "paddleocr" and "easyocr" engines
4. **Import errors**: Verify all dependencies are installed

### Debug Mode

For debugging, add print statements in the handler or run with verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This implementation follows the same license terms as the original OmniParser project:
- Icon detection model: AGPL license
- Caption models: MIT license

See the original [OmniParser repository](https://github.com/microsoft/OmniParser) for full license details. 