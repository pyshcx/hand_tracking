import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import osascript
import warnings

# Suppress the specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated.")

wCam, hCam = 720, 720

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
frame_count = 0
smoothing = 5
vol_levels = []
volume_locked = False
thumbs_up_frames = 0
thumbs_up_threshold = 40  # Increased to about 2 seconds at 20 FPS
last_state_change_time = 0
state_change_cooldown = 2  # 2 seconds cooldown

while True:
    success, img = cap.read()
    frame_count += 1

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # Check for thumbs up gesture
        thumb_tip = lmList[4][2]
        index_tip = lmList[8][2]
        middle_tip = lmList[12][2]
        ring_tip = lmList[16][2]
        pinky_tip = lmList[20][2]

        if thumb_tip < index_tip and thumb_tip < middle_tip and thumb_tip < ring_tip and thumb_tip < pinky_tip:
            thumbs_up_frames += 1
            if thumbs_up_frames >= thumbs_up_threshold:
                current_time = time.time()
                if current_time - last_state_change_time >= state_change_cooldown:
                    volume_locked = not volume_locked
                    last_state_change_time = current_time
                thumbs_up_frames = 0
        else:
            thumbs_up_frames = 0

        if not volume_locked and frame_count % 2 == 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            vol = np.interp(length, [50, 300], [0, 100])
            vol_levels.append(vol)
            if len(vol_levels) > smoothing:
                vol_levels.pop(0)
            vol = sum(vol_levels) / len(vol_levels)

            volBar = np.interp(vol, [0, 100], [400, 150])
            volPer = int(vol)

            if frame_count % 10 == 0:
                set_volume(volPer)

            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{volPer}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Display lock status
    lock_status = "Locked" if volume_locked else "Unlocked"
    cv2.putText(img, f'Volume: {lock_status}', (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img,handNo=0,draw=True):

        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                #print(id,lm)
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                #print(id,cx,cy)
                lmList.append([id,cx,cy] )
                if draw:
                #if id==12:
                    cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist=detector.findPosition(img)
        if len(lmlist)!=0:
            print(lmlist[4])


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()


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

