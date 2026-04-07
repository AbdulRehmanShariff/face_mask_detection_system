# Face Mask Detection System

A real-time face mask detection system built using Transfer Learning
with MobileNetV2 and OpenCV. Detects whether a person is wearing a
face mask or not — from both uploaded images and live webcam feed.

## What it does

- Detects masked and unmasked faces in uploaded images
- Live webcam detection with bounding boxes and confidence scores
- Supports multiple faces in a single frame
- Web interface built with Flask — runs in any browser
- Detection history with session statistics

## Tech Stack

| Layer | Tool |
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| Model | MobileNetV2 (Transfer Learning) |
| Face Detection | OpenCV DNN (SSD + ResNet-10) |
| Web Backend | Flask |
| Frontend | HTML, CSS, JavaScript |

## Project Structure

face-mask-detection/
├── app.py  
├── Procfile  
├── runtime.txt  
├── requirements.txt  
├── README.md  
├── deploy.prototxt  
├── face_detector.caffemodel  
│ └── best_model.keras  
├── templates/
│ └── index.html  
└── static/
└── uploads/

## How to Run Locally

**Step 1 — Clone or download the project**

```bash
cd face-mask-detection
```

**Step 2 — Create virtual environment**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — Run the app**

```bash
python app.py
```

**Step 5 — Open in browser**

http://127.0.0.1:5000

## Dataset

- **Source:** Kaggle Face Mask Dataset
- **Total images:** 7,553
- **Classes:** `with_mask` / `without_mask`
- **Split:** 80% training / 20% validation

## Model Details

- **Base model:** MobileNetV2 pre-trained on ImageNet
- **Fine-tuned on:** 7,553 face mask images
- **Input size:** 224 x 224 x 3
- **Validation accuracy:** 98.87%
- **Face detector:** OpenCV DNN SSD (more accurate than Haarcascade)

## Features

- Transfer learning for high accuracy with less training data
- DNN-based face detection works at angles and varying distances
- Real-time webcam inference with multi-face support
- Confidence score shown for every detected face
- Clean web UI — no installation needed by the end user

**Important**: These are standard stable versions. If your training used a different version, run this command to get YOUR exact versions and replace the file with that output:

**pip freeze > requirements.txt**

Always prefer pip freeze over manually writing versions — it captures exactly what worked on your machine.

**Built by Abdul Rehman**
**year : 2026**
