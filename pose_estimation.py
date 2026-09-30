import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

def _calculate_angle(a, b, c):
    """
    Calculate 2D angle (in degrees) at point b given points a, b, c.
    Returns a value between 0.0 and 180.0.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def classify_pose_from_landmarks(landmarks):
    """
    Classifies human posture and arm gestures from 33 MediaPipe landmarks.
    """
    if not landmarks or len(landmarks) < 33:
        return {
            "pose_name": "Unknown",
            "base_posture": "Unknown",
            "arm_gesture": "None",
            "torso_angle": "N/A",
            "knee_angle": "N/A",
            "hip_angle": "N/A"
        }

    # Extract key landmarks
    nose = landmarks[0]
    l_sh, r_sh = landmarks[11], landmarks[12]
    l_el, r_el = landmarks[13], landmarks[14]
    l_wr, r_wr = landmarks[15], landmarks[16]
    l_hip, r_hip = landmarks[23], landmarks[24]
    l_knee, r_knee = landmarks[25], landmarks[26]
    l_ank, r_ank = landmarks[27], landmarks[28]

    # Midpoints
    sh_mid = np.array([(l_sh.x + r_sh.x) / 2.0, (l_sh.y + r_sh.y) / 2.0])
    hip_mid = np.array([(l_hip.x + r_hip.x) / 2.0, (l_hip.y + r_hip.y) / 2.0])

    # Torso inclination relative to vertical (upright vertical vector is [0, -1])
    dx = sh_mid[0] - hip_mid[0]
    dy = sh_mid[1] - hip_mid[1]  # In image coords, dy < 0 means shoulders are higher than hips
    torso_angle_vert = float(np.degrees(np.arctan2(abs(dx), -dy)))

    # Visibility threshold
    V = 0.4
    left_leg_vis = (l_knee.visibility > V and l_ank.visibility > V)
    right_leg_vis = (r_knee.visibility > V and r_ank.visibility > V)
    has_legs = left_leg_vis or right_leg_vis

    # Knee angles (hip - knee - ankle)
    knee_angles = []
    if left_leg_vis:
        knee_angles.append(_calculate_angle([l_hip.x, l_hip.y], [l_knee.x, l_knee.y], [l_ank.x, l_ank.y]))
    if right_leg_vis:
        knee_angles.append(_calculate_angle([r_hip.x, r_hip.y], [r_knee.x, r_knee.y], [r_ank.x, r_ank.y]))
    avg_knee_angle = np.mean(knee_angles) if knee_angles else None

    # Hip angles (shoulder - hip - knee)
    hip_angles = []
    if l_knee.visibility > V:
        hip_angles.append(_calculate_angle([l_sh.x, l_sh.y], [l_hip.x, l_hip.y], [l_knee.x, l_knee.y]))
    if r_knee.visibility > V:
        hip_angles.append(_calculate_angle([r_sh.x, r_sh.y], [r_hip.x, r_hip.y], [r_knee.x, r_knee.y]))
    avg_hip_angle = np.mean(hip_angles) if hip_angles else None

    # Base Posture
    if dy > 0.1:  # Inverted: shoulders below hips
        base_posture = "Inverted / Handstand"
    elif torso_angle_vert > 55:
        base_posture = "Lying Down / Reclining"
    elif not has_legs:
        base_posture = "Upper Body (Upright)"
    elif torso_angle_vert > 35 and (avg_knee_angle is not None and avg_knee_angle > 140):
        base_posture = "Bending Forward"
    else:
        # Leg analysis: Squatting vs Sitting vs Standing
        if avg_knee_angle is not None and avg_knee_angle < 100 and avg_hip_angle is not None and avg_hip_angle < 100:
            base_posture = "Squatting"
        elif avg_knee_angle is not None and avg_knee_angle < 135:
            base_posture = "Sitting"
        else:
            base_posture = "Standing"

    # Arm Gestures
    l_arm_raised = (l_wr.visibility > V and l_wr.y < l_sh.y)
    r_arm_raised = (r_wr.visibility > V and r_wr.y < r_sh.y)

    arm_gesture = ""
    if l_arm_raised and r_arm_raised:
        arm_gesture = "Both Arms Raised"
    elif l_arm_raised:
        arm_gesture = "Left Arm Raised"
    elif r_arm_raised:
        arm_gesture = "Right Arm Raised"
    else:
        l_el_angle = _calculate_angle([l_sh.x, l_sh.y], [l_el.x, l_el.y], [l_wr.x, l_wr.y])
        r_el_angle = _calculate_angle([r_sh.x, r_sh.y], [r_el.x, r_el.y], [r_wr.x, r_wr.y])
        sh_dist = abs(l_sh.x - r_sh.x)
        wr_dist = abs(l_wr.x - r_wr.x)
        if (l_el_angle > 135 and r_el_angle > 135 and 
            abs(l_wr.y - l_sh.y) < 0.15 and abs(r_wr.y - r_sh.y) < 0.15 and
            wr_dist > max(sh_dist * 1.8, 0.3)):
            arm_gesture = "Arms Outstretched (T-Pose)"
        elif (l_wr.visibility > V and r_wr.visibility > V and
              abs(l_wr.y - l_hip.y) < 0.15 and abs(r_wr.y - r_hip.y) < 0.15 and
              l_el_angle < 130 and r_el_angle < 130):
            arm_gesture = "Hands on Hips"

    if arm_gesture:
        pose_name = f"{base_posture} ({arm_gesture})"
    else:
        pose_name = base_posture

    return {
        "pose_name": pose_name,
        "base_posture": base_posture,
        "arm_gesture": arm_gesture if arm_gesture else "Arms at Sides / Neutral",
        "torso_angle": round(torso_angle_vert, 1),
        "knee_angle": round(avg_knee_angle, 1) if avg_knee_angle is not None else "N/A",
        "hip_angle": round(avg_hip_angle, 1) if avg_hip_angle is not None else "N/A"
    }

def detect_pose(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    out = image.copy()

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            out, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        pose_info = classify_pose_from_landmarks(results.pose_landmarks.landmark)
        return out, True, pose_info

    return out, False, {
        "pose_name": "No Pose Detected",
        "base_posture": "None",
        "arm_gesture": "None",
        "torso_angle": "N/A",
        "knee_angle": "N/A",
        "hip_angle": "N/A"
    }
