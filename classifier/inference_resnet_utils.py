# classifier/inference_resnet_utils.py
"""
Helper utilities to load the trained ResNet-50 bird/no-bird classifier
and run predictions on cropped image patches (BGR, from OpenCV).
"""

from pathlib import Path
from typing import List, Dict, Optional

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from utils.config import CLASS_NAMES, CLF_IMAGE_SIZE


def build_resnet50_model(num_classes: int) -> nn.Module:
    """
    Build the same ResNet-50 architecture used during training:
    - pretrained on ImageNet
    - final FC layer replaced with num_classes outputs
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_transform(image_size: int) -> transforms.Compose:
    """
    Preprocessing pipeline for classifier:
    - BGR (OpenCV) -> RGB
    - Resize to (image_size, image_size)
    - ToTensor
    - Normalize with ImageNet stats
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class ResNetBirdClassifier:
    """
    Wrapper around the trained ResNet-50 bird/no-bird classifier.

    Usage:
        clf = ResNetBirdClassifier(
            weights_path="classifier/resnet50_best.pth",
            class_names=["bird", "no-bird"],
            image_size=224,
        )

        result = clf.predict(crop_bgr)
        print(result["label"], result["conf"])
    """

    def __init__(
        self,
        weights_path: str | Path,
        class_names: Optional[List[str]] = None,
        image_size: int = CLF_IMAGE_SIZE,
        device: Optional[torch.device] = None,
    ):
        self.weights_path = Path(weights_path)
        if class_names is None:
            self.class_names = CLASS_NAMES
        else:
            self.class_names = class_names

        self.num_classes = len(self.class_names)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Build model and load weights
        self.model = build_resnet50_model(self.num_classes)
        state = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing
        self.transform = get_transform(image_size)

    @torch.inference_mode()
    def predict_raw(self, crop_bgr) -> torch.Tensor:
        """
        Run forward pass on a single BGR crop (numpy array).
        Returns logits tensor of shape [num_classes].
        """
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Empty crop provided to classifier.")

        # BGR -> RGB
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        x = self.transform(crop_rgb)  # [C, H, W]
        x = x.unsqueeze(0).to(self.device)  # [1, C, H, W]

        logits = self.model(x)[0]  # [num_classes]
        return logits

    @torch.inference_mode()
    def predict(self, crop_bgr, return_proba: bool = True) -> Dict:
        """
        Predict class for a BGR crop.

        Returns dict:
            {
                "label": "bird" or "no-bird",
                "index": int,
                "conf": float (softmax prob of predicted class),
                "probs": { "bird": p0, "no-bird": p1 }  # if return_proba=True
            }
        """
        logits = self.predict_raw(crop_bgr)
        probs = F.softmax(logits, dim=0)

        conf, idx = torch.max(probs, dim=0)
        idx = int(idx.item())
        conf = float(conf.item())

        label = self.class_names[idx]

        out = {
            "label": label,
            "index": idx,
            "conf": conf,
        }

        if return_proba:
            out["probs"] = {
                self.class_names[i]: float(probs[i].item())
                for i in range(self.num_classes)
            }

        return out
