# 🐦 Bird Detection using YOLOv8 + ResNet-50 Refinement

This project implements a deep learning–based real-time bird detection system designed for agricultural field surveillance. It uses a two-stage pipeline where YOLOv8 performs fast object detection and a ResNet-50 classifier refines detections to remove false positives such as leaves, insects, shadows, and branches.

It provides a complete pipeline—from dataset preparation and model training to inference and deployment via a Flask web application.

---

## 📚 Table of Contents

- 📦 Dataset
- 🧠 Model Architecture
- 📁 Project Structure
- ⚙️ Installation
- 🚀 Usage
- 🌐 Web Application
- 💾 Saved Models
- 🛠 Features
- 🧩 Dependencies
- 🧪 Examples
- 🐞 Troubleshooting
- ✍️ Author
- 📜 License

---

## 📦 Dataset

The datasets used in this project are not included in the repository and are ignored using `.gitignore` to keep the repository lightweight.

### Dataset Structure

data/
- yolo_dataset        (YOLOv8 dataset with bird and no-bird classes)
- yolo_dataset-2      (Additional YOLO dataset)
- crop_dataset        (Cropped patches for ResNet-50 training)
- crop_dataset-2

### Dataset Description

- YOLO datasets contain full images with bounding-box annotations
- Classes include bird and no-bird
- Crop datasets contain small image patches extracted from YOLO detections
- Crop datasets are used to train the ResNet-50 classifier

### Dataset Preparation

Generate crops for classifier training using:

python classifier/prepare_crops.py

---

## 🧠 Model Architecture

This system combines fast object detection with classification-based refinement.

YOLOv8 (Stage 1 – Detection):
- Detects bird-like objects in images and video frames
- Outputs bounding boxes with confidence scores
- Optimized for real-time performance

ResNet-50 (Stage 2 – Refinement):
- Takes cropped YOLO detections as input
- Classifies each crop as bird or no-bird
- Filters out false positives before final output

Task Type: Real-time object detection with refinement  
Domain: Agricultural field and crop protection monitoring

---

## 📁 Project Structure

bird_detection_project/
- app.py (Flask web application)
- inference_combined.py (CLI inference for image, video, webcam)
- requirements.txt
- README.md
- templates/
  - index.html
  - result.html
- static/
- utils/
  - config.py
- pipeline/
  - inference_service.py
- yolo_training/
  - train_yolov8.py
- classifier/
  - train_resnet50.py
  - prepare_crops.py
  - inference_resnet_utils.py
  - weights/
    - resnet50_best_v2.pth
- runs_yolo/
  - yolov8m_bird_no_bird_v2/
    - weights/
      - best.pt
- data/ (ignored)

---

## ⚙️ Installation

Requirements:
- Python 3.10 or higher
- Windows / Linux / macOS
- Dependencies listed in requirements.txt

Install dependencies:
pip install -r requirements.txt

---

## 🚀 Usage

Training:
- Train YOLOv8: python yolo_training/train_yolov8.py
- Train ResNet-50: python classifier/train_resnet50.py

Inference (CLI):
- Webcam: python inference_combined.py --source 0
- Image: python inference_combined.py --source test_images/sample.jpg --save
- Video: python inference_combined.py --source test_images/sample.mp4 --save

---

## 🌐 Web Application

Start the Flask server:
python app.py

Open browser:
http://127.0.0.1:5000/

---

## 💾 Saved Models

- YOLOv8 model: runs_yolo/yolov8m_bird_no_bird_v2/weights/best.pt
- ResNet-50 classifier: classifier/weights/resnet50_best_v2.pth

---

## 🛠 Features

- Two-stage detection pipeline for higher precision
- Removes false positives using classifier refinement
- Supports image, video, and webcam input
- Flask-based web interface
- Modular and extensible codebase

---

## 🧩 Dependencies

- Ultralytics YOLOv8
- PyTorch
- OpenCV
- Flask
- NumPy
- Matplotlib
- Pillow

---

## 🧪 Examples

- Upload farm surveillance videos to detect only real birds
- Run webcam inference for live agricultural monitoring

---

## 🐞 Troubleshooting

- Flask app not starting: ensure Flask is installed and port 5000 is free
- Model not found: verify paths to model weight files
- Slow inference: use GPU if available

---

## ✍️ Author

Aamer Khan  
Developed for real-time bird detection in agricultural environments.

---



