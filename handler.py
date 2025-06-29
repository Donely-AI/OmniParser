"""OmniParser handler for runpod serverless inference."""

import base64
import io
import os

import pandas as pd
import runpod
import torch
from PIL import Image

from util.utils import (
    check_ocr_box,
    get_caption_model_processor,
    get_som_labeled_img,
    get_yolo_model,
)

# Global variables for models
som_model = None
caption_model_processor = None


def initialize_models():
    """Initialize the OmniParser models."""
    global som_model, caption_model_processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load YOLO model for icon detection
    model_path = "weights/icon_detect/model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    som_model = get_yolo_model(model_path)
    som_model.to(device)
    print(f"SOM model loaded to {device}")

    # Load Florence2 caption model
    caption_model_processor = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path="weights/icon_caption_florence",
        device=device,
    )
    print("Caption model loaded")


def handler(job):
    """Handler function that processes OmniParser jobs."""
    global som_model, caption_model_processor

    # Initialize models on first run
    if som_model is None or caption_model_processor is None:
        initialize_models()

    job_input = job["input"]

    # Get input parameters
    image_input = job_input.get("image")
    if not image_input:
        return {"error": "No image provided"}

    # Optional parameters with defaults
    box_threshold = job_input.get("box_threshold", 0.05)
    output_format = job_input.get("output_format", "detailed")  # "detailed" or "simple"
    use_ocr = job_input.get("use_ocr", True)
    ocr_engine = job_input.get("ocr_engine", "paddleocr")  # "paddleocr" or "easyocr"

    try:
        # Handle different image input formats
        if isinstance(image_input, str):
            if image_input.startswith("data:image"):
                # Handle base64 data URL
                image_data = image_input.split(",")[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif image_input.startswith("/") or os.path.exists(image_input):
                # Handle file path
                image = Image.open(image_input)
            else:
                # Assume it's base64 encoded
                image_bytes = base64.b64decode(image_input)
                image = Image.open(io.BytesIO(image_bytes))
        else:
            return {"error": "Invalid image format"}

        image_rgb = image.convert("RGB")
        print(f"Processing image of size: {image.size}")

        # Configure drawing parameters based on image size
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            "text_scale": 0.8 * box_overlay_ratio,
            "text_thickness": max(int(2 * box_overlay_ratio), 1),
            "text_padding": max(int(3 * box_overlay_ratio), 1),
            "thickness": max(int(3 * box_overlay_ratio), 1),
        }

        # Perform OCR if enabled
        ocr_bbox = None
        ocr_text = []
        if use_ocr:
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
                image,
                display_img=False,
                output_bb_format="xyxy",
                goal_filtering=None,
                easyocr_args={"paragraph": False, "text_threshold": 0.9},
                use_paddleocr=(ocr_engine == "paddleocr"),
            )
            ocr_text, ocr_bbox = ocr_bbox_rslt

        # Process image with OmniParser
        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image,
            som_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=ocr_text,
            use_local_semantics=True,
            iou_threshold=0.7,
            scale_img=False,
            batch_size=128,
        )

        # Prepare response based on output format
        if output_format == "simple":
            # Simple format: just the parsed content
            simple_content = []
            for item in parsed_content_list:
                simple_content.append(
                    {
                        "type": item["type"],
                        "content": item["content"],
                        "bbox": item["bbox"],
                        "interactivity": item["interactivity"],
                    }
                )

            return {
                "success": True,
                "parsed_content": simple_content,
                "image_size": list(image.size),
            }

        else:
            # Detailed format: include annotated image and structured data
            df = pd.DataFrame(parsed_content_list)
            df["ID"] = range(len(df))

            return {
                "success": True,
                "annotated_image": dino_labeled_img,  # base64 encoded
                "label_coordinates": label_coordinates,
                "parsed_content": parsed_content_list,
                "structured_data": df.to_dict("records"),
                "image_size": list(image.size),
                "total_elements": len(parsed_content_list),
            }

    except Exception as e:
        return {"error": f"Processing failed: {str(e)}", "success": False}


runpod.serverless.start({"handler": handler})
