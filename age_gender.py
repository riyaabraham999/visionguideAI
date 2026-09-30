import os
import cv2
import numpy as np
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

FACE_MODEL = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
FACE_CONFIG = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_net.caffemodel")
AGE_PROTO = os.path.join(MODEL_DIR, "age_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")
GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")

MODEL_URLS = {
    FACE_MODEL: "https://huggingface.co/AjaySharma/genderDetection/resolve/main/opencv_face_detector_uint8.pb",
    FACE_CONFIG: "https://huggingface.co/AjaySharma/genderDetection/resolve/main/opencv_face_detector.pbtxt",
    AGE_MODEL: "https://huggingface.co/AjaySharma/genderDetection/resolve/main/age_net.caffemodel",
    AGE_PROTO: "https://huggingface.co/AjaySharma/genderDetection/resolve/main/age_deploy.prototxt",
    GENDER_MODEL: "https://huggingface.co/AjaySharma/genderDetection/resolve/main/gender_net.caffemodel",
    GENDER_PROTO: "https://huggingface.co/AjaySharma/genderDetection/resolve/main/gender_deploy.prototxt",
}

AGE_LIST = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_LIST = ["Male", "Female"]

face_net = age_net = gender_net = None

def _download_missing_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for path, url in MODEL_URLS.items():
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            try:
                urllib.request.urlretrieve(url, path)
            except Exception:
                pass

def _load_models():
    global face_net, age_net, gender_net
    missing = [p for p in [FACE_MODEL, FACE_CONFIG, AGE_MODEL, AGE_PROTO, GENDER_MODEL, GENDER_PROTO] if not os.path.exists(p)]
    if missing:
        _download_missing_models()
        missing = [p for p in [FACE_MODEL, FACE_CONFIG, AGE_MODEL, AGE_PROTO, GENDER_MODEL, GENDER_PROTO] if not os.path.exists(p)]
        if missing:
            return False, missing
    if face_net is None:
        face_net = cv2.dnn.readNet(FACE_MODEL, FACE_CONFIG)
        age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    return True, []

def analyze_age_gender(image):
    ok, missing = _load_models()
    details = []
    if not ok:
        # Fallback to Haar Cascade face detection if DNN models cannot be loaded
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        out = image.copy()
        for x, y, w, h in faces:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 100), 2)
            cv2.putText(out, "Face detected", (x, max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 100), 2)
            details.append({"gender": "Unknown", "age": "Unknown", "confidence": 1.0})
        return out, len(faces), details

    img_h, img_w = image.shape[:2]

    # --- Face detection (use a dedicated variable for this blob) ---
    face_blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                      [104, 117, 123], False, False)
    face_net.setInput(face_blob)
    detections = face_net.forward()

    # Collect all candidate face boxes and confidences for NMS
    boxes = []
    confidences = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < 0.5:
            continue
        box = detections[0, 0, i, 3:7] * np.array([img_w, img_h, img_w, img_h])
        x1, y1, x2, y2 = box.astype(int)
        boxes.append([x1, y1, x2 - x1, y2 - y1])   # x, y, w, h for NMS
        confidences.append(conf)

    out = image.copy()
    count = 0

    if not boxes:
        return out, 0, details

    # Apply Non-Maximum Suppression to remove duplicate/overlapping detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5,
                               nms_threshold=0.3)

    # Flatten indices (OpenCV versions return different shapes)
    if len(indices) > 0:
        indices = indices.flatten()

    for idx in indices:
        bx, by, bw, bh = boxes[idx]
        confidence = confidences[idx]
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh

        # Proportional padding (~20% of face size) for better classification.
        # The model was trained on faces with surrounding context, so a tight
        # crop causes misclassification (especially female → male).
        pad_x = max(20, int(bw * 0.20))
        pad_y = max(20, int(bh * 0.20))
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(img_w, x2 + pad_x)
        crop_y2 = min(img_h, y2 + pad_y)

        # .copy() ensures an independent array (not a view into the original)
        face = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if face.size == 0:
            continue

        count += 1

        # Create SEPARATE blobs for each network — OpenCV DNN can modify
        # the input blob in-place during forward(), so sharing a blob
        # between networks causes the second network to get corrupted input.
        mean_values = (78.4263377603, 87.7689143744, 114.895847746)

        # Gender prediction
        gender_blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), mean_values, swapRB=False
        )
        gender_net.setInput(gender_blob)
        gender_preds = gender_net.forward().copy()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        # Age prediction (fresh blob from the same face crop)
        age_blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), mean_values, swapRB=False
        )
        age_net.setInput(age_blob)
        age_preds = age_net.forward().copy()
        age = AGE_LIST[age_preds[0].argmax()]

        details.append({
            "gender": gender,
            "age": age,
            "confidence": float(confidence)
        })

        label = f"{gender}, Age {age}"
        font_scale = max(0.55, min((x2 - x1) / 220.0, 1.2))
        thickness = max(2, int(font_scale * 2.2))
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Draw green bounding box around face (original box, not padded)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 100), 2)

        # Background rectangle for high-contrast legible label
        tag_y1 = max(0, y1 - th - 12)
        tag_y2 = y1
        cv2.rectangle(out, (x1, tag_y1), (x1 + tw + 10, tag_y2), (20, 20, 20), -1)
        cv2.putText(out, label, (x1 + 5, tag_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out, count, details
