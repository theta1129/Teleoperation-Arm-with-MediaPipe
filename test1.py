import cv2
import mediapipe as mp
import time
import sys
import numpy as np


def Get_Angles(a):
    arr = []
    # print(len(a))
    for i in range(2, len(a)):
        x1 = a[i].x
        y1 = a[i].y
        z1 = a[i].z
        x2 = a[i - 1].x
        y2 = a[i - 1].y
        z2 = a[i - 1].z
        x3 = a[i - 2].x
        y3 = a[i - 2].y
        z3 = a[i - 2].z
        v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
        v2 = np.array([x3 - x2, y3 - y2, z3 - z2])
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        arr.append(angle_deg)
    return arr

# MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Set HandLandmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)


landmarker = HandLandmarker.create_from_options(options)
mp_drawing = mp.tasks.vision.drawing_utils
mp_hands = mp.tasks.vision.HandLandmarksConnections
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open error")
    exit()

timestamp = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

    timestamp += 1

    result = landmarker.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            for landmark in landmarks:
                h, w, _ = frame.shape

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            for connection in mp_hands.HAND_CONNECTIONS:
                start = landmarks[connection.start]
                end = landmarks[connection.end]
                x1 = int(start.x * w)
                y1 = int(start.y * h)

                x2 = int(end.x * w)
                y2 = int(end.y * h)

                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # print(result.hand_landmarks[0])
        # sys.exit(0)
        # print(type(result.hand_landmarks))
        # print(len(result.hand_landmarks))

        angles = Get_Angles([result.hand_landmarks[0][0]] + result.hand_landmarks[0][5:9])
        print(angles)

    cv2.imshow("Hand Landmarker", frame)


    if cv2.waitKey(1) & 0xFF == 27:
        break



cap.release()
cv2.destroyAllWindows()
landmarker.close()