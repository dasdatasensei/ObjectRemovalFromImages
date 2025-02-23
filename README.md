# **Object Removal Using YOLOv8 🚀**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23white.svg?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Gradio](https://img.shields.io/badge/Gradio-4.19+-orange.svg)](https://gradio.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository provides an **AI-powered object removal tool** using **YOLOv8** for object detection and OpenCV for inpainting. It allows users to automatically detect and remove unwanted objects from images.

## **🖥️ Interface**

![AI Object Removal Tool Interface](assets/AIObjectRemovalTool.png)

The tool provides an intuitive web interface where you can:

- Upload or paste images directly
- Select objects to remove from a comprehensive list
- Choose cleanup level (light, medium, heavy)
- See before/after comparison
- Download the processed image

---

## **✨ Features**

- **Object Detection**: Uses **YOLOv8** to identify objects in an image.
- **Automated Object Removal**: Removes detected objects and fills the missing areas using **OpenCV inpainting**.
- **Efficient Processing**: Fast and lightweight pipeline for real-time object removal.

---

## **🛠️ Installation**

Clone the repository and install dependencies:

```sh
git clone https://github.com/yourusername/ObjectRemoval.git
cd ObjectRemoval
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

---

## **🚀 Usage**

Run the script with an input image:

```sh
python main.py --input data/input/image.jpg --output data/output/cleaned.jpg
```

The script will:

1. Detect objects in the image.
2. Create masks for detected objects.
3. Remove objects and inpaint the missing area.

---

## **📂 Project Structure**

```
📦 ObjectRemoval
 ┣ 📂 data
 ┃ ┣ 📂 input        # Original images
 ┃ ┣ 📂 output       # Processed images
 ┣ 📂 src
 ┃ ┣ object_removal.py  # Core processing logic
 ┣ .venv                # Virtual environment (ignored)
 ┣ requirements.txt      # Dependencies
 ┣ main.py              # Entry point
 ┗ README.md            # You're reading this!
```

---

## **🔧 Dependencies**

```sh
pip install -r requirements.txt
```

- `opencv-python`
- `torch`, `torchvision`
- `ultralytics` (YOLOv8)
- `numpy`, `Pillow`

---

## **🖼️ Example**

In the provided example, the script successfully removes a cup from an image sourced from Pexels. The input image shows the cup, and the output image displays the same scene without the cup.

### **Input**

<img src="data/input/pexels-theo-1090064-3414792.jpg" width="400">

### **Output (Object Removed)**

<img src="data/output/pexels-theo-1090064-3414792_cleaned.jpg" width="400">

---

## **📜 License**

MIT License. Free to use and modify.

---

Made with love by The Data Sensei! 🚀
