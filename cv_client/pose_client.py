import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets
import json
import time

# ---------------- CONFIG ----------------
USER_ID = 1
WEIGHT = 60
MET = 3

# ---------------- MEDIAPIPE ----------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- PREPROCESSING (CLAHE) ----------------
def preprocess(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl,a,b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return final

# ---------------- ANGLE CALCULATION ----------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle
    return angle

# ---------------- POSE CLASSIFICATION ----------------
def classify_pose(landmarks):

    l_shoulder = [landmarks[11].x, landmarks[11].y]
    r_shoulder = [landmarks[12].x, landmarks[12].y]
    l_elbow = [landmarks[13].x, landmarks[13].y]
    r_elbow = [landmarks[14].x, landmarks[14].y]
    l_wrist = [landmarks[15].x, landmarks[15].y]
    r_wrist = [landmarks[16].x, landmarks[16].y]
    l_hip = [landmarks[23].x, landmarks[23].y]
    r_hip = [landmarks[24].x, landmarks[24].y]
    l_knee = [landmarks[25].x, landmarks[25].y]
    r_knee = [landmarks[26].x, landmarks[26].y]
    l_ankle = [landmarks[27].x, landmarks[27].y]
    r_ankle = [landmarks[28].x, landmarks[28].y]

    left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
    right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
    left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    body_angle = calculate_angle(l_shoulder, l_hip, l_ankle)

    # ---- PLANK ----
    if body_angle > 160 and left_knee_angle > 160 and right_knee_angle > 160:
        return "plank"

    # ---- DOWNDOG ----
    if body_angle < 100:
        return "downdog"

    # ---- WARRIOR2 ----
    if (80 < left_knee_angle < 110 or 80 < right_knee_angle < 110) and \
       left_elbow_angle > 160 and right_elbow_angle > 160:
        return "warrior2"

    # ---- TREE ----
    if (left_knee_angle < 100 and right_knee_angle > 150) or \
       (right_knee_angle < 100 and left_knee_angle > 150):
        return "tree"

    # ---- GODDESS ----
    if 80 < left_knee_angle < 120 and 80 < right_knee_angle < 120:
        return "goddess"

    return "unknown"

# ---------------- WEBSOCKET SENDER ----------------
async def send_data(data):
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(data))
        await websocket.recv()

# ---------------- MAIN LOOP ----------------
cap = cv2.VideoCapture(0)

start_time = time.time()
last_sent = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = preprocess(frame)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=3, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
        )

        landmarks = results.pose_landmarks.landmark

        pose_name = classify_pose(landmarks)

        duration = time.time() - start_time
        calories = MET * WEIGHT * (duration / 3600)

        confidence = np.mean([lm.visibility for lm in landmarks])

        data = {
            "user_id": USER_ID,
            "pose_name": pose_name,
            "duration": round(duration, 2),
            "accuracy": 90,
            "confidence": round(confidence, 2),
            "calories": round(calories, 2)
        }

        # Send every 3 seconds
        if time.time() - last_sent > 3:
            asyncio.run(send_data(data))
            print("Sent:", data)
            last_sent = time.time()

        cv2.putText(frame, f"Pose: {pose_name}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.putText(frame, f"Conf: {confidence:.2f}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,0,0), 2)

    cv2.imshow("Yoga Pose Detection (Improved)", frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()