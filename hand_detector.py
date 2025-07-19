import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

class HandDetector:
    def __init__(self, max_hands=2, detection_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        return frame

    def draw_landmarks(self, frame):
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)

    def fingers_up(self, hand_landmarks):
        tips = [4, 8, 12, 16, 20]
        fingers = []

        # Get landmark positions
        landmarks = hand_landmarks.landmark

        # Thumb
        if landmarks[tips[0]].x < landmarks[tips[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for tip_id in tips[1:]:
            if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers