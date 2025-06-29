#!/usr/bin/env python3
"""
Test script for the OmniParser handler.
This script simulates how the handler would be called in a runpod serverless environment.
"""

import base64
import json

from handler import handler


def load_image_as_base64(image_path):
    """Load an image file and convert it to base64."""
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        return base64.b64encode(image_data).decode("utf-8")


def test_handler():
    """Test the handler with a sample image."""

    # Test with one of the demo images
    image_path = "imgs/windows_home.png"  # You can change this to any image path

    try:
        # Load image as base64
        image_base64 = load_image_as_base64(image_path)

        # Create a mock job similar to what runpod would send
        test_job = {
            "input": {
                "image": image_base64,
                "box_threshold": 0.05,
                "output_format": "detailed",  # or "simple"
                "use_ocr": True,
                "ocr_engine": "paddleocr",  # or "easyocr"
            }
        }

        print("Testing OmniParser handler...")
        print(f"Processing image: {image_path}")

        # Call the handler
        result = handler(test_job)

        if result.get("success"):
            print("✅ Handler executed successfully!")
            print(f"Total elements detected: {result.get('total_elements', 0)}")
            print(f"Image size: {result.get('image_size')}")

            # Print some sample parsed content
            parsed_content = result.get("parsed_content", [])
            print(f"\nFirst 5 detected elements:")
            for i, item in enumerate(parsed_content[:5]):
                print(
                    f"  {i+1}. Type: {item['type']}, Content: {item['content'][:50]}..."
                )

            # Save annotated image if available
            if "annotated_image" in result:
                with open("test_output_annotated.png", "wb") as f:
                    f.write(base64.b64decode(result["annotated_image"]))
                print("\n📸 Annotated image saved as 'test_output_annotated.png'")

            # Save full results as JSON
            with open("test_output_results.json", "w") as f:
                # Remove the base64 image data for cleaner JSON
                clean_result = result.copy()
                if "annotated_image" in clean_result:
                    clean_result["annotated_image"] = "[base64 image data removed]"
                json.dump(clean_result, f, indent=2)
            print("📄 Full results saved as 'test_output_results.json'")

        else:
            print("❌ Handler failed:")
            print(result.get("error", "Unknown error"))

    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_handler()
