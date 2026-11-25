# AI-Based Human Height and Weight Estimation Using Deep Learning

## Project Overview
This project estimates human height and weight from an image using deep learning and computer vision techniques.

The system uses a Flask-based web application, YOLOv8 Pose for person visibility screening, and a fine-tuned ResNet-50 model for height and weight regression.

## Technologies Used
- Python
- Flask
- PyTorch
- ResNet-50
- YOLOv8 Pose
- HTML, CSS, JavaScript

## Project Structure
```text
Human-Height-Weight-Estimation/
├── app.py
├── requirements.txt
├── model/
│   └── best_finetune_layer4.pt
├── static/
│   ├── camera.js
│   └── style.css
├── templates/
│   └── index.html
└── yolov8n-pose.pt
```

## Features
- Camera-based image capture
- Person detection and pose-based input screening
- Image preprocessing
- Deep learning-based height and weight estimation
- Browser-based result display

## Installation and Usage
1. Clone or download this repository.
2. Install dependencies using:
   `pip install -r requirements.txt`
3. Ensure the trained ResNet-50 checkpoint is placed inside the `model` folder.
4. Run the application:
   `python app.py`
5. Open `http://127.0.0.1:5000/` in your browser.

## Disclaimer
The displayed height and weight values are estimates, not verified physical measurements. Results may vary with image quality, camera angle, visibility, and other factors.
