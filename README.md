🐦 Bird Detection using YOLOv8 + ResNet-50 Refinement

This project implements a deep learning–based real-time bird detection system for agricultural field surveillance using a two-stage pipeline consisting of YOLOv8 for object detection and ResNet-50 for refinement. The refiner removes false positives such as leaves, insects, shadows, and branches, resulting in higher detection precision. The project includes training, evaluation, inference, visualization, and deployment via a Flask web application.

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

The training and testing datasets are organized in the following structure (not included in the repository):

data/
├── yolo_dataset/ # Used to train YOLOv8 (2 classes: bird, no-bird)
├── yolo_dataset-2/
├── crop_dataset/ # Crops for training ResNet-50 classifier
└── crop_dataset-2/

⚠️ Note: These datasets are excluded using .gitignore to keep the repository lightweight.

Dataset Preparation:

YOLO datasets contain labeled bounding boxes for bird detection.

Crop datasets contain image patches extracted from YOLO detections.

Use classifier/prepare_crops.py to generate crops for classifier training.

🧠 Model Architecture

This system combines fast object detection with classification-based refinement:

YOLOv8:

Detects candidate birds in full images or video frames.

Outputs bounding boxes with confidence scores.

Trained on two classes: bird and no-bird.

ResNet-50:

Takes cropped YOLO detections as input.

Classifies each crop as bird or no-bird.

Filters out false detections.

Task Type: Real-time object detection with refinement
Domain: Agricultural field monitoring

📁 Project Structure

bird_detection_project/
├── app.py # Flask web application
├── inference_combined.py # Unified inference script
├── requirements.txt # Dependencies
├── README.md
│
├── templates/
│ ├── index.html
│ └── result.html
│
├── static/ # Optional CSS / JS files
│
├── utils/
│ ├── config.py
│ └── visualization.py
│
├── pipeline/
│ ├── inference_service.py
│ └── detect_with_refiner.py
│
├── yolo_training/
│ ├── train_yolov8.py
│ └── infer_yolov8.py
│
├── classifier/
│ ├── train_resnet50.py
│ ├── prepare_crops.py
│ ├── inference_resnet_utils.py
│ ├── infer_resnet50.py
│ └── weights/
│ └── resnet50_best_v2.pth
│
├── runs_yolo/
│ └── yolov8m_bird_no_bird_v2/
│ └── weights/
│ └── best.pt
│
└── data/ # Ignored datasets

⚙️ Installation

Requirements:

Python 3.10+

Windows / Linux / macOS

Dependencies listed in requirements.txt

Steps:

1️⃣ Clone the Repository

git clone <repository-url>
cd bird_detection_project

2️⃣ Create Virtual Environment (Recommended)

Windows:
python -m venv venv
venv\Scripts\activate

Linux / macOS:
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies

pip install -r requirements.txt

🚀 Usage

🔧 Train YOLOv8
python yolo_training/train_yolov8.py

🔧 Train ResNet-50 Classifier
python classifier/train_resnet50.py

🧪 Run Inference (CLI)

Webcam:
python inference_combined.py --source 0

Image:
python inference_combined.py --source test_images/sample.jpg --save

Video:
python inference_combined.py --source test_images/sample.mp4 --save

🌐 Web Application

Start the Flask server:
python app.py

Then open your browser at:
http://127.0.0.1:5000/

Upload an image or video to view refined bird detections.

💾 Saved Models

The repository includes pretrained weights:

YOLOv8 model
runs_yolo/yolov8m_bird_no_bird_v2/weights/best.pt

ResNet-50 classifier
classifier/weights/resnet50_best_v2.pth

These allow inference without retraining.

🛠 Features

Two-stage detection pipeline for higher precision

Filters false positives using classifier refinement

Supports image, video, and webcam input

Flask-based web interface

Modular and extensible codebase

Real-time performance

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

Install them using:

pip install -r requirements.txt

🧪 Examples

Upload a farm surveillance video and detect only confirmed birds.

Run webcam inference for real-time monitoring.

Compare YOLO-only vs refined detection accuracy.

🐞 Troubleshooting

Flask app not running? ➜ Ensure Flask is installed and port 5000 is free.

Model not found? ➜ Check paths to best.pt and resnet50_best_v2.pth.

No detections? ➜ Verify input image size and lighting conditions.

Slow inference? ➜ Use GPU if available.

✍️ Author

Aamer Khan
Developed for real-time bird detection in agricultural environments.

📜 License

This project is intended for academic and research purposes only.
For commercial usage, please contact the author.