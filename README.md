# AI Human Analysis System

A single Deep Learning/Computer Vision project containing:

1. Gender and Age Detection
2. Human Pose Estimation
3. Image Caption Generation

## Setup

Recommended: Python 3.11 or 3.12.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Important: Age/Gender model files

Place these six pretrained files inside `models/`:

- opencv_face_detector_uint8.pb
- opencv_face_detector.pbtxt
- age_net.caffemodel
- age_deploy.prototxt
- gender_net.caffemodel
- gender_deploy.prototxt

The application deliberately does NOT invent age/gender results if these model files are missing. Until they are supplied, it falls back to face detection.

## Captioning

The first time Image Caption Generator is used, Hugging Face downloads the BLIP model. Internet access is required for the first download.

## Run

```bash
streamlit run app.py
```
