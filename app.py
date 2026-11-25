
from pathlib import Path
import io

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "model" / "best_finetune_layer4.pt"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

device = torch.device("cpu")


def load_saved_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    # Only load checkpoints you trust. This file is your own saved model.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint["model_config"]
    norm = checkpoint["target_normalization"]
    preprocessing = checkpoint["preprocessing"]

    if config["arch"] != "resnet50":
        raise ValueError(f"Unexpected architecture: {config['arch']}")

    # Match the trained ResNet-50 head.
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()

    class HeightWeightModel(nn.Module):
        def __init__(self, backbone, dropout=0.2):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Dropout(float(dropout)),
                nn.Linear(2048, 2),
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

    model = HeightWeightModel(
        backbone,
        dropout=config.get("dropout", 0.2),
    )     

    # Load the saved parameters without training.
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(
            (
                int(preprocessing["canvas_h"]),
                int(preprocessing["canvas_w"]),
            ),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.Normalize(
            mean=preprocessing["normalize_mean"],
            std=preprocessing["normalize_std"],
        ),
    ])

    # The training notebook used aspect-ratio-preserving resize,
    # centred on a fixed canvas with this padding colour.
    canvas_h = int(preprocessing["canvas_h"])
    canvas_w = int(preprocessing["canvas_w"])
    pad_rgb = tuple(preprocessing["pad_rgb"])

    def preprocess(image):
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        scale = min(canvas_w / image.width, canvas_h / image.height)
        new_w = max(1, round(image.width * scale))
        new_h = max(1, round(image.height * scale))

        image = image.resize(
            (new_w, new_h),
            Image.Resampling.BICUBIC,
        )

        canvas = Image.new("RGB", (canvas_w, canvas_h), pad_rgb)
        left = (canvas_w - new_w) // 2
        top = (canvas_h - new_h) // 2
        canvas.paste(image, (left, top))

        tensor = transforms.ToTensor()(canvas)
        tensor = transforms.Normalize(
            mean=preprocessing["normalize_mean"],
            std=preprocessing["normalize_std"],
        )(tensor)
        return tensor.unsqueeze(0)

    return model, norm, preprocess


model, target_norm, preprocess_image = load_saved_model()

# Load YOLO person detection model
person_detector = YOLO("yolov8n-pose.pt")

target_mean = target_norm["mean"]
target_std = target_norm["std"]


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(error="No image was uploaded."), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify(error="Choose or capture an image first."), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        image = ImageOps.exif_transpose(image).convert("RGB")

        # Detect people using YOLO
        detection_results = person_detector.predict(
            source=image,
            classes=[0],
            conf=0.50,
            verbose=False
        )

        boxes = detection_results[0].boxes
        keypoints = detection_results[0].keypoints

        # Check whether a person was detected
        person_count = 0 if boxes is None else len(boxes)

        if person_count == 0:
            return jsonify(
                error="No person detected. Please upload an image containing a person."
            ), 400

        # Allow only one person
        if person_count > 1:
            return jsonify(
                error="Multiple people detected. Please upload an image containing only one person."
            ), 400

        # Check full-body keypoints
        if keypoints is None or len(keypoints.data) == 0:
            return jsonify(
                error="Body landmarks could not be detected. Please try another image."
            ), 400

        points = keypoints.data[0].tolist()

        # COCO keypoints: shoulders, hips, knees and ankles
        required_points = [5, 6, 11, 12, 13, 14, 15, 16]

        confidence_threshold = 0.35

        for index in required_points:
            if points[index][2] < confidence_threshold:
                return jsonify(
                    error="Full body not visible. Please stand farther away and capture your complete body, including your feet."
                ), 400

        # ResNet-50 prediction
        tensor = preprocess_image(image).to(device)

        with torch.inference_mode():
            output = model(tensor).reshape(-1).cpu().tolist()

        height_cm = output[0] * target_std[0] + target_mean[0]
        weight_kg = output[1] * target_std[1] + target_mean[1]

        return jsonify(
            height_cm=round(float(height_cm), 2),
            weight_kg=round(float(weight_kg), 2),
            note="Estimates only; not verified measurements."
        )

    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify(error=f"Could not process this image: {exc}"), 400


if __name__ == "__main__":
    print("Saved model loaded. Starting local web app...")
    app.run(host="0.0.0.0", port=5000, debug=True)