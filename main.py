import cv2
import numpy as np
from hand_detector import HandDetector
from tensorflow.keras.models import load_model

# Load your trained model and labels
model = load_model("model.h5")
labels = ["A", "B", "C", "D", "E"]  # Replace with your labels

# Initialize detector and webcam
detector = HandDetector()
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = detector.find_hands(frame)

    # 1. Hand Landmark Detection
    fingers_info = "No hand detected"
    if detector.results.multi_hand_landmarks:
        for hand_landmarks in detector.results.multi_hand_landmarks:
            detector.draw_landmarks(frame)
            fingers = detector.fingers_up(hand_landmarks)
            fingers_info = f"Fingers: {fingers}"

            # 2. Alphabet Prediction from ROI
            x, y, w, h = 100, 100, 300, 300
            roi = frame[y:y+h, x:x+w]

            roi = cv2.resize(roi, (200, 200))  # match model input
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = roi / 255.0
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi)
            predicted_label = labels[np.argmax(prediction)]

            # 3. Show predicted alphabet
            cv2.putText(frame, f"Sign: {predicted_label}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    # 4. Show finger status
    cv2.putText(frame, fingers_info, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Optional: Show ROI box
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    cv2.imshow("Hand Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
