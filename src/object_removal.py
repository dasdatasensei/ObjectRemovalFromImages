import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load YOLOv8 model (Runs Locally)
yolo_model = YOLO("yolov8n.pt")  # Smallest, fastest version

def detect_objects(image_path: str):
    """
    Runs YOLO locally to detect objects in the image.

    Parameters:
    - image_path (str): Path to the input image.

    Returns:
    - list of detected objects with bounding boxes
    """
    results = yolo_model(image_path)  # Perform object detection
    detections = []

    for result in results:
        for box in result.boxes.data.tolist():  # Extract bounding boxes
            x1, y1, x2, y2, confidence, class_id = box
            label = result.names[int(class_id)]  # Get object label
            detections.append({"label": label, "bbox": [x1, y1, x2, y2]})

    return detections

def generate_object_mask(image_path: str, objects: list[str], detections: list) -> np.ndarray:
    """
    Creates a binary mask for specified objects.

    Parameters:
    - image_path (str): Path to the input image.
    - objects (list[str]): List of object names to remove.
    - detections (list): YOLO detection results.

    Returns:
    - np.ndarray: Binary mask of objects to remove.
    """
    image = cv2.imread(image_path)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Empty mask

    for detection in detections:
        label = detection["label"]
        if label in objects:
            x1, y1, x2, y2 = detection["bbox"]
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=-1)

    return mask

def inpaint_image(image_path: str, mask: np.ndarray, output_path: str) -> str:
    """
    Uses OpenCV inpainting to remove objects.

    Parameters:
    - image_path (str): Path to the input image.
    - mask (np.ndarray): Binary mask for object removal.
    - output_path (str): Path to save modified image.

    Returns:
    - str: Path to modified image.
    """
    image = cv2.imread(image_path)
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    cv2.imwrite(output_path, inpainted_image)
    logging.info(f"Saved modified image at {output_path}")

    return output_path

def remove_objects_from_image(image_path: str, objects: list[str]) -> str:
    """
    Detects and removes specified objects from an image.

    Parameters:
    - image_path (str): Path to input image.
    - objects (list[str]): List of object names to remove.

    Returns:
    - str: Filepath to modified image.
    """
    logging.info(f"Processing image: {image_path}")

    # Step 1: Detect objects using YOLO (Local)
    detections = detect_objects(image_path)

    # Step 2: Create object mask
    mask = generate_object_mask(image_path, objects, detections)

    # Step 3: Define output path
    filename = os.path.basename(image_path).replace(".jpg", "_cleaned.jpg")
    output_path = os.path.join("data", "output", filename)

    # Step 4: Inpaint and save result
    return inpaint_image(image_path, mask, output_path)