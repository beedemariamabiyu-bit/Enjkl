import cv2
import mediapipe as mp
import pyautogui
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
def main(): 
    base_options = python.BaseOptions(
        model_asset_path="hand_landmarker.task"
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )

    detector = vision.HandLandmarker.create_from_options(options)

    # Open camera
    cap = cv2.VideoCapture(0)

    screen_width, screen_height = pyautogui.size()
    # Gesture control variables
    clicking = False
    previous_scroll_y = None
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("Camera not working")
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb_frame
        )
        results = detector.detect(mp_image)
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                clicking = detect_click(hand_landmarks, clicking)
                previous_scroll_y = detect_scroll(hand_landmarks, previous_scroll_y)
                move_cursor(hand_landmarks, screen_width, screen_height, w, h, frame)
        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
def detect_click(hand_landmarks, clicking):
    index_finger_tip = hand_landmarks[8]
    thumb_tip = hand_landmarks[4]

    distance = np.hypot(
            index_finger_tip.x - thumb_tip.x,
            index_finger_tip.y - thumb_tip.y
            )

    if distance < 0.07 and not clicking:
        pyautogui.click()
        clicking = True

    elif distance > 0.10:
        clicking = False
        
    return clicking
def detect_scroll(hand_landmarks, previous_scroll_y):
    index_finger_tip = hand_landmarks[8]
    middle_finger_tip = hand_landmarks[12]
    #check if index and middle fingers 
    index_up = hand_landmarks[8].y < hand_landmarks[6].y
    middle_up = hand_landmarks[12].y < hand_landmarks[10].y

    if index_up and middle_up:
        
        current_y = (index_finger_tip.y + middle_finger_tip.y) / 2

        if previous_scroll_y is not None:
            movement = previous_scroll_y - current_y

            if abs(movement) > 0.005:
                pyautogui.scroll(int(movement*1500))

        previous_scroll_y = current_y

    else:
        previous_scroll_y = None
    return previous_scroll_y
def move_cursor(hand_landmarks, screen_width, screen_height, w, h, frame):
    index_finger_tip = hand_landmarks[8]
    x = int(index_finger_tip.x * w)
    y = int(index_finger_tip.y * h)

    screen_x = np.interp( x, [100, w - 100], [0,screen_width] )
    screen_y = np.interp( y, [80, h - 80], [0,screen_height])
    screen_x = np.clip( screen_x, 0, screen_width)
    screen_y = np.clip( screen_y, 0, screen_height)

    pyautogui.moveTo(screen_x, screen_y)

    for landmark in hand_landmarks:
        px = int(landmark.x * w)
        py = int(landmark.y * h)
        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
    connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),        
                (0, 5), (5, 6), (6, 7), (7, 8),      
                (5, 9), (9, 10), (10, 11), (11, 12),     
                (9, 13), (13, 14), (14, 15), (15, 16),   
                (13, 17), (17, 18), (18, 19), (19, 20),   
                (0, 17)         
            ]

    for start, end in connections:
        x1 = int(hand_landmarks[start].x * w)
        y1 = int(hand_landmarks[start].y * h)

        x2 = int(hand_landmarks[end].x * w)
        y2 = int(hand_landmarks[end].y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    


if __name__ == "__main__":
    main()