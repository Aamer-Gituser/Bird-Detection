# yolo_training/train_yolov8.py

import os
from ultralytics import YOLO
import torch


def main():
    # -------- Device selection --------
    if torch.cuda.is_available():
        device = 0  # first GPU
        print("[INFO] CUDA GPU detected. Using GPU 0.")
    else:
        device = "cpu"
        print("[INFO] No CUDA GPU detected. Using CPU.")

    # -------- Paths --------
    # This script is in BIRD_DETECTION_PROJECT/yolo_training/
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_YAML = os.path.join(ROOT_DIR, "data", "yolo_dataset", "data.yaml")

    print(f"[INFO] Project root: {ROOT_DIR}")
    print(f"[INFO] Using data config: {DATA_YAML}")

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"Data config not found at: {DATA_YAML}")

    # -------- Hyperparameters (tunable) --------
    model_name = "yolov8m"      # base architecture
    img_size = 640              # input size (your dataset is 640x640)
    epochs = 100                # max epochs (early stopping via patience)
    batch_size = 16             # adjust if you hit GPU OOM
    patience = 20               # stop if no val improvement after 20 epochs

    # Training output folders
    project_name = os.path.join(ROOT_DIR, "runs_yolo")
    exp_name = "yolov8m_bird_no_bird"

    # -------- Load pretrained YOLOv8m model --------
    print(f"[INFO] Loading model: {model_name}.pt")
    model = YOLO(f"{model_name}.pt")  # downloads weights if not present

    # -------- Train --------
    print("[INFO] Starting training ...")
    results = model.train(
        data=DATA_YAML,
        imgsz=img_size,
        epochs=epochs,
        batch=batch_size,
        project=project_name,
        name=exp_name,
        device=device,
        patience=patience,
        workers=4,          # you can increase if you have many CPU cores
        verbose=True        # detailed logs in terminal
    )

    print("[INFO] Training finished.")
    print(f"[INFO] Best results saved in: {results.save_dir}")


if __name__ == "__main__":
    main()
