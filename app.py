import os
import gradio as gr
from pathlib import Path
from src.object_removal import remove_objects_from_image
from ultralytics import YOLO
import logging
from rich.logging import RichHandler
import cv2
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Get available classes from YOLO model
AVAILABLE_OBJECTS = sorted(model.names.values())


def post_process_image(image_path, cleanup_level="medium"):
    """
    Apply post-processing to clean up artifacts.

    Args:
        image_path: Path to the image
        cleanup_level: Level of cleanup (light, medium, heavy)

    Returns:
        str: Path to the processed image
    """
    # Read the image
    image = cv2.imread(image_path)

    if cleanup_level == "light":
        # Light cleanup - subtle smoothing
        processed = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)
    elif cleanup_level == "medium":
        # Medium cleanup - balanced approach
        processed = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
        # Apply subtle denoising
        processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
    else:  # heavy
        # Heavy cleanup - aggressive smoothing and enhancement
        processed = cv2.detailEnhance(image, sigma_s=15, sigma_r=0.25)
        processed = cv2.fastNlMeansDenoisingColored(processed, None, 15, 15, 7, 21)
        # Additional edge-aware smoothing
        processed = cv2.edgePreservingFilter(
            processed, flags=2, sigma_s=100, sigma_r=0.5
        )

    # Save the processed image
    output_path = image_path.replace(".jpg", "_cleaned.jpg")
    cv2.imwrite(output_path, processed)
    return output_path


def process_image(image, objects_to_remove, cleanup_level="medium"):
    """
    Process the image using the object removal pipeline.

    Args:
        image: Input image (numpy array or filepath)
        objects_to_remove: List of objects to remove
        cleanup_level: Level of post-processing cleanup

    Returns:
        tuple: (processed image path, status message)
    """
    if not objects_to_remove:
        return image, "Please select at least one object to remove."

    # Save input image temporarily
    input_path = "data/input/temp_input.jpg"
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)

    try:
        # Handle different input types
        if isinstance(image, str):  # If image is a file path
            input_path = image
        elif isinstance(image, np.ndarray):  # If image is a numpy array
            cv2.imwrite(input_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            return None, "❌ Error: Invalid image format"

        # Process the image
        result_path = remove_objects_from_image(input_path, objects_to_remove)

        # Apply post-processing cleanup
        final_path = post_process_image(result_path, cleanup_level)

        # Return the processed image path
        return final_path, f"✨ Objects removed and image cleaned successfully!"

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return None, f"❌ Error: {str(e)}"
    finally:
        # Cleanup temporary input file only if we created it
        if input_path == "data/input/temp_input.jpg" and os.path.exists(input_path):
            os.remove(input_path)


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 AI Object Removal Tool

        Upload an image and select objects you want to remove. The AI will detect and remove them automatically!

        Powered by YOLOv8 and OpenCV inpainting with advanced artifact cleanup.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload Image", type="numpy", sources=["upload", "clipboard"]
            )
            objects = gr.Dropdown(
                choices=AVAILABLE_OBJECTS,
                multiselect=True,
                label="Select Objects to Remove",
                info="Choose one or more objects to remove from the image",
            )
            cleanup_level = gr.Radio(
                choices=["light", "medium", "heavy"],
                value="medium",
                label="Cleanup Level",
                info="Choose how aggressively to remove artifacts",
            )
            process_btn = gr.Button("🚀 Remove Objects", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Processed Image", type="filepath")
            status_text = gr.Textbox(label="Status", interactive=False)

    # Add examples
    example_path = str(Path("data/input/pexels-theo-1090064-3414792.jpg"))
    if os.path.exists(example_path):
        gr.Examples(
            examples=[[example_path, ["cup"], "medium"]],
            inputs=[input_image, objects, cleanup_level],
            outputs=[output_image, status_text],
            fn=process_image,
            cache_examples=True,
        )

    # Set up event handler
    process_btn.click(
        fn=process_image,
        inputs=[input_image, objects, cleanup_level],
        outputs=[output_image, status_text],
    )

    gr.Markdown(
        """
        ### 📝 Instructions
        1. Upload an image or use the example
        2. Select objects you want to remove from the dropdown
        3. Choose cleanup level:
           - Light: Subtle artifact removal
           - Medium: Balanced cleanup (recommended)
           - Heavy: Aggressive smoothing
        4. Click "Remove Objects" and wait for processing
        5. Download the result using the download button

        ### ℹ️ Note
        - The AI can detect and remove common objects like people, cars, animals, furniture, etc.
        - The quality of removal depends on the complexity of the scene and the size of objects
        - Higher cleanup levels may reduce detail but remove more artifacts
        """
    )

if __name__ == "__main__":
    demo.launch(
        share=True,  # Create a public link
        server_name="0.0.0.0",  # Make accessible from other devices
        server_port=7860,  # Default Gradio port
    )
