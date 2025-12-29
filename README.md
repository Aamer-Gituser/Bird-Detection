🐦 Bird Detection using YOLOv8 + ResNet-50 Refinement

This project implements a deep learning–based real-time bird detection system designed for agricultural field surveillance. It uses a two-stage pipeline where YOLOv8 performs fast object detection and a ResNet-50 classifier refines detections to remove false positives such as leaves, insects, shadows, and branches.

The system supports image, video, and webcam inference and includes training, evaluation, inference, and deployment via a Flask web application.

📚 Table of Contents

📦 Dataset
🧠 Model Architecture
📁 Project Structure
⚙️ Installation
🚀 Usage
🌐 Web Application
💾 Saved Models
🛠 Features
🧩 Dependencies
🧪 Examples
🐞 Troubleshooting
✍️ Author
📜 License

📦 Dataset

The datasets used in this project are not included in the repository and are ignored using .gitignore to keep the repository lightweight.

Dataset structure:

data

yolo_dataset (YOLOv8 dataset with bird and no-bird classes)

yolo_dataset-2 (additional YOLO dataset)

crop_dataset (cropped patches for ResNet-50 training)

crop_dataset-2

Dataset description:

YOLO datasets contain full images with bounding-box annotations

Crop datasets contain small image patches extracted from YOLO detections

Crop datasets are used to train the ResNet-50 classifier

Dataset preparation:

Use classifier/prepare_crops.py to generate crop datasets

Run this step whenever the YOLO dataset is updated

Note: Large datasets are intentionally excluded from GitHub.

🧠 Model Architecture

This system combines fast object detection with classification-based refinement.

YOLOv8 (Stage 1 – Detection):

Detects bird-like objects in images and video frames

Produces bounding boxes with confidence scores

Optimized for real-time performance

ResNet-50 (Stage 2 – Refinement):

Takes cropped YOLO detections as input

Classifies each crop as bird or no-bird

Filters out false positives before final output

Task Type: Real-time object detection with refinement
Domain: Agricultural field and crop protection monitoring

📁 Project Structure

bird_detection_project

app.py (Flask web application)

inference_combined.py (CLI inference for image, video, webcam)

requirements.txt (Python dependencies)

README.md

templates

index.html (Upload page)

result.html (Results display page)

static (Optional CSS and JS files)

utils

config.py (Global constants and settings)

pipeline

inference_service.py (Combined YOLO + ResNet inference logic)

yolo_training

train_yolov8.py (YOLOv8 training script)

classifier

train_resnet50.py (ResNet-50 training script)

prepare_crops.py (Crop dataset generator)

inference_resnet_utils.py (ResNet inference utilities)

weights

resnet50_best_v2.pth (Trained classifier weights)

runs_yolo

yolov8m_bird_no_bird_v2

weights

best.pt (Trained YOLOv8 model)

data (Ignored datasets)

⚙️ Installation

Requirements:

Python 3.10 or higher

Windows, Linux, or macOS

Dependencies listed in requirements.txt

Steps:

Clone the repository and move into the project directory

Create and activate a virtual environment (recommended)

Install all required dependencies using requirements.txt

🚀 Usage

Training:

Train YOLOv8 using yolo_training/train_yolov8.py

Train ResNet-50 classifier using classifier/train_resnet50.py

Inference (CLI):

Webcam inference using inference_combined.py with source 0

Image inference by passing an image path

Video inference by passing a video path

🌐 Web Application

Start the Flask server using app.py

Open the browser at http://127.0.0.1:5000/

Upload an image or video to view refined bird detections

💾 Saved Models

The repository includes pretrained models for direct inference:

YOLOv8 model
Location: runs_yolo/yolov8m_bird_no_bird_v2/weights/best.pt

ResNet-50 classifier
Location: classifier/weights/resnet50_best_v2.pth

These models allow inference without retraining.

🛠 Features

Two-stage detection pipeline for higher precision

Removes false positives using classifier refinement

Supports image, video, and webcam input

Flask-based web interface

Modular and extensible codebase

Near real-time performance on GPU

🧩 Dependencies

All dependencies are listed in requirements.txt.

Major libraries include:

Ultralytics YOLOv8

PyTorch

OpenCV

Flask

NumPy

Matplotlib

Pillow

🧪 Examples

Upload farm surveillance videos to detect only real birds

Run webcam inference for live agricultural monitoring

Compare YOLO-only detection versus refined detection accuracy

🐞 Troubleshooting

Flask app not starting:

Ensure Flask is installed and port 5000 is free

Model not found:

Verify paths to best.pt and resnet50_best_v2.pth

No detections:

Check lighting conditions and input resolution

Slow inference:

Use GPU if available

✍️ Author

Aamer Khan
Developed for real-time bird detection in agricultural environments.

📜 License

This project is intended solely for academic and research purposes.
For commercial usage, please contact the author.