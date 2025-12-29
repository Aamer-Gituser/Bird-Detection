🐦 Bird Detection using YOLOv8 + ResNet-50 Refinement

This project implements a deep learning–based real-time bird detection system designed for agricultural field surveillance. It uses a two-stage pipeline where YOLOv8 performs fast object detection and a ResNet-50 classifier refines detections to remove false positives such as leaves, insects, shadows, and branches.

It provides a complete pipeline—from dataset preparation and model training to inference and deployment via a Flask web application.

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

Dataset Structure
data/
├── yolo_dataset/        # YOLOv8 dataset (bird, no-bird)
├── yolo_dataset-2/      # Additional YOLO dataset
├── crop_dataset/        # Cropped patches for ResNet-50 training
└── crop_dataset-2/

Dataset Description

YOLO datasets contain full images with bounding-box annotations

Classes: bird, no-bird

Crop datasets contain small image patches extracted from YOLO detections

Crop datasets are used to train the ResNet-50 classifier

Dataset Preparation

Generate crops for classifier training using:

python classifier/prepare_crops.py


⚠️ Note: Large datasets are intentionally excluded from GitHub.

🧠 Model Architecture

This system combines fast object detection with classification-based refinement.

YOLOv8 (Stage 1 – Detection)

Detects bird-like objects in images and video frames

Outputs bounding boxes with confidence scores

Optimized for real-time performance

ResNet-50 (Stage 2 – Refinement)

Takes cropped YOLO detections as input

Classifies each crop as bird or no-bird

Filters out false positives before final output

Task Type: Real-time object detection with refinement
Domain: Agricultural field and crop protection monitoring

📁 Project Structure
bird_detection_project/
├── app.py                         # Flask web application
├── inference_combined.py          # CLI inference (image/video/webcam)
├── requirements.txt               # Python dependencies
├── README.md
│
├── templates/
│   ├── index.html                 # Upload page
│   └── result.html                # Results display page
│
├── static/                        # Optional CSS / JS
│
├── utils/
│   └── config.py                  # Global constants
│
├── pipeline/
│   └── inference_service.py       # Combined YOLO + ResNet inference
│
├── yolo_training/
│   └── train_yolov8.py            # YOLOv8 training
│
├── classifier/
│   ├── train_resnet50.py          # ResNet-50 training
│   ├── prepare_crops.py           # Crop dataset generator
│   ├── inference_resnet_utils.py  # ResNet inference
│   └── weights/
│       └── resnet50_best_v2.pth
│
├── runs_yolo/
│   └── yolov8m_bird_no_bird_v2/
│       └── weights/
│           └── best.pt
│
└── data/                          # Ignored datasets

⚙️ Installation
Requirements

Python 3.10 or higher

Windows / Linux / macOS

Dependencies listed in requirements.txt

Setup Steps

Create a virtual environment and install dependencies:

pip install -r requirements.txt

🚀 Usage
Training

Train YOLOv8:

python yolo_training/train_yolov8.py


Train ResNet-50:

python classifier/train_resnet50.py

Inference (CLI)

Webcam:

python inference_combined.py --source 0


Image:

python inference_combined.py --source test_images/sample.jpg --save


Video:

python inference_combined.py --source test_images/sample.mp4 --save

🌐 Web Application

Start the Flask server:

python app.py


Open in browser:

http://127.0.0.1:5000/


Upload an image or video to view refined bird detections.

💾 Saved Models

Pretrained models included for direct inference:

YOLOv8
runs_yolo/yolov8m_bird_no_bird_v2/weights/best.pt

ResNet-50
classifier/weights/resnet50_best_v2.pth

No retraining is required for inference.

🛠 Features

Two-stage detection pipeline for higher precision

Eliminates false positives using classifier refinement

Supports image, video, and webcam input

Flask-based web interface

Modular and extensible codebase

Near real-time performance on GPU

🧩 Dependencies

Major libraries used:

Ultralytics YOLOv8

PyTorch

OpenCV

Flask

NumPy

Matplotlib

Pillow

🧪 Examples

Upload farm surveillance videos and detect only real birds

Run webcam inference for live agricultural monitoring

Compare YOLO-only vs refined detection accuracy

🐞 Troubleshooting

Flask app not starting
Ensure Flask is installed and port 5000 is free

Model not found
Verify paths to best.pt and resnet50_best_v2.pth

No detections
Check lighting conditions and image resolution

Slow inference
Use GPU if available

✍️ Author

Aamer Khan
Developed for real-time bird detection in agricultural environments.

📜 License

This project is intended solely for academic and research purposes.
For commercial usage, please contact the author.