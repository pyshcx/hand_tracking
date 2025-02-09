import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import math

# Streamlit App Setup
st.title("Hand Tracking & Volume Control")
run = st.checkbox('Start Webcam', value=True)

# Initialize MediaPipe for Hand Detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Setup webcam
cap = cv2.VideoCapture(0)
pTime = 0

# Smoothing and Volume variables
smoothing = 5
vol_levels = []

def process_frame(frame):
    global vol_levels
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if lmList:
                x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
                x2, y2 = lmList[8][1], lmList[8][2]  # Index

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Draw circles and line
                cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                # Distance between thumb and index finger
                length = math.hypot(x2 - x1, y2 - y1)

                # Volume Control (mapped between 0 to 100)
                vol = np.interp(length, [50, 300], [0, 100])
                vol_levels.append(vol)
                if len(vol_levels) > smoothing:
                    vol_levels.pop(0)
                vol = sum(vol_levels) / len(vol_levels)

                volPer = int(vol)

                # Volume bar display
                volBar = np.interp(vol, [0, 100], [400, 150])
                cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f'{volPer}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                # Change color when fingers are close
                if length < 50:
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    return frame


# Display video stream in Streamlit
frame_placeholder = st.empty()

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to access the webcam.")
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally for a mirror-like effect
    processed_frame = process_frame(frame)

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(processed_frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

    # Convert to RGB and display in Streamlit
    frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")

cap.release()
