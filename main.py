import os
from src.object_removal import remove_objects_from_image

if __name__ == "__main__":
    input_dir = "data/input"
    output_dir = "data/output"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Image to process
    image_filename = "pexels-theo-1090064-3414792.jpg"
    image_path = os.path.join(input_dir, image_filename)

    # Objects to remove
    objects_to_remove = ["cup", "book"]

    # Process image
    result_path = remove_objects_from_image(image_path, objects_to_remove)
    print(f"Modified image saved at: {result_path}")
