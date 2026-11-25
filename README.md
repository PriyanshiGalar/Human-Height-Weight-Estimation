# AI-Based Human Height and Weight Estimation Using Deep Learning

## 1. Project Overview

This project presents an AI-based system for estimating human height and weight from images using deep learning and computer vision techniques.

The system uses a Flask web application integrated with YOLOv8 Pose for person detection and pose-based input screening, along with a fine-tuned ResNet-50 model for height and weight regression.

Users can capture an image through their browser camera and receive estimated height (cm) and weight (kg).

## 2. Objectives

- Estimate human height and weight from images.
- Apply deep learning techniques for anthropometric estimation.
- Use pose detection to screen image inputs.
- Develop a user-friendly web interface for image capture and prediction.

## 3. Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Web Framework | Flask |
| Deep Learning Framework | PyTorch |
| Feature Extraction | ResNet-50 |
| Pose Detection | YOLOv8 Pose |
| Frontend | HTML, CSS, JavaScript |
| Image Processing | Pillow |

## 4. System Features

- Browser-based camera capture
- Person detection and pose-based input screening
- Image preprocessing and normalization
- ResNet-50-based height and weight estimation
- Display of estimated measurements through the web interface

## 5. Project Structure

```text
Human-Height-Weight-Estimation/
│
├── app.py
├── requirements.txt
├── .gitignore
├── yolov8n-pose.pt
├── README.md
│
├── model/
│   └── best_finetune_layer4.pt
│
├── static/
│   ├── camera.js
│   └── style.css
│
└── templates/
    └── index.html
```

**Note:** The trained ResNet-50 checkpoint is distributed separately through GitHub Releases.

## 6. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/PriyanshiGalar/Human-Height-Weight-Estimation.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Human-Height-Weight-Estimation
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Trained Model

Download `best_finetune_layer4.pt` from the project's GitHub Release:

[Download Trained ResNet-50 Model – Release v1.0](https://github.com/PriyanshiGalar/Human-Height-Weight-Estimation/releases/tag/v1.0)

Create a folder named `model` in the project directory and place the downloaded checkpoint inside it.

The final model path should be:

```text
model/best_finetune_layer4.pt
```

### Step 5: Run the Application

```bash
python app.py
```

### Step 6: Open the Web Application

Open the following address in your browser:

```text
http://127.0.0.1:5000/
```

Allow camera access when prompted to capture an image and obtain the estimated measurements.

## 7. Model Information

- Architecture: ResNet-50
- Pretrained weights: ImageNet
- Fine-tuning: Layer 4 and regression head
- Output: Height in centimetres and weight in kilograms
- Task: Multi-output regression

The model predicts continuous height and weight values rather than classification labels.

## 8. Limitations and Disclaimer

The system provides estimated measurements and does not replace actual physical measurements.

Prediction quality may be affected by camera angle, image quality, body visibility, pose, clothing, and dataset limitations. The model has been trained and evaluated using a celebrity image dataset and should not be assumed to provide verified accuracy for all individuals or CCTV environments.

## 9. GitHub Repository

**Project Repository:**  
https://github.com/PriyanshiGalar/Human-Height-Weight-Estimation

**Model Release:**  
https://github.com/PriyanshiGalar/Human-Height-Weight-Estimation/releases/tag/v1.0

---

**Developed as a Final Year Project**  
AI-Based Human Height and Weight Estimation Using Deep Learning
