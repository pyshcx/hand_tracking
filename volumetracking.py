import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import osascript
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.HandDetector(detectionCon=0.7, maxHands=1)

def set_volume(volume):
    osascript.osascript(f"set volume output volume {volume}")

def get_output_volume():
    result = osascript.osascript('output volume of (get volume settings)')
    return int(result[1])

vol = 0
volBar = 400
volPer = 0
smoothing = 5
vol_levels = []

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw elements on the image
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # Interpolating the length into volume level
        vol = np.interp(length, [50, 300], [0, 100])
        vol_levels.append(vol)
        if len(vol_levels) > smoothing:
            vol_levels.pop(0)
        vol = sum(vol_levels) / len(vol_levels)

        volBar = np.interp(vol, [0, 100], [400, 150])
        volPer = int(vol)

        # Set the system volume every frame
        set_volume(volPer)
        print(f"Volume Level: {volPer}% | Thumb: ({x1}, {y1}) | Index: ({x2}, {y2})")

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # Draw volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{volPer}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
