# **Object Removal Using YOLOv8 🚀**

This repository provides an **AI-powered object removal tool** using **YOLOv8** for object detection and OpenCV for inpainting. It allows users to automatically detect and remove unwanted objects from images.

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
### **Input**
<img src="data/input/image.jpg" width="400">

### **Output (Object Removed)**
<img src="data/output/cleaned.jpg" width="400">

---

## **📜 License**
MIT License. Free to use and modify.

---

Let me know if you need any refinements! 🚀
