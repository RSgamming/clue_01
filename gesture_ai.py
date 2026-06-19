import cv2
import mediapipe as mp
import numpy as np
import keras
import os
import pickle
from collections import deque, Counter

DATA_PATH = "gesture_data.pkl"
MODEL_PATH = "gesture_model.h5"
LABELS_PATH = "gesture_labels.pkl"

class Gesture_AI:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils
        self.model = None
        self.guessed_gesture = None

    def extract_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            return landmarks, results
        return None, results

    def collect_data(self):
        gesture_name = input("Enter gesture name: ")
        samples = []
        cap = cv2.VideoCapture(0)
        print("Press 's' to save a sample, 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks, results = self.extract_landmarks(frame)
            if results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Collecting Gesture Data", frame)
            key = cv2.waitKey(1)
            if key == ord('s') and landmarks is not None:
                samples.append(landmarks)
                print(f"Saved sample #{len(samples)}")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "rb") as f:
                data = pickle.load(f)
        else:
            data = {}

        data.setdefault(gesture_name.upper(), []).extend(samples)
        with open(DATA_PATH, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {len(samples)} samples for gesture '{gesture_name.upper()}'")

    def train_model(self):
        if not os.path.exists(DATA_PATH):
            print("No data found.")
            return

        with open(DATA_PATH, "rb") as f:
            data = pickle.load(f)

        X, y = [], []
        labels = sorted(data.keys())
        for idx, gesture in enumerate(labels):
            for sample in data[gesture]:
                X.append(sample)
                y.append(idx)

        X = np.array(X)
        y = keras.utils.to_categorical(y, num_classes=len(labels))

        self.model = keras.Sequential([
            keras.layers.Input(shape=(X.shape[1],)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(len(labels), activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=30, batch_size=16)
        self.model.save(MODEL_PATH)
        with open(LABELS_PATH, "wb") as f:
            pickle.dump(labels, f)
        print("Model trained and saved.")

    def recognize(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            print("Model not found.")
            return

        self.model = keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH, "rb") as f:
            labels = pickle.load(f)

        cap = cv2.VideoCapture(0)
        buffer_size = 10
        prediction_buffer = deque(maxlen=buffer_size)
        gesture = ""

        print("Press 'y' to confirm gesture, 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks, results = self.extract_landmarks(frame)
            if results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)
            if landmarks is not None and self.model is not None:
                pred = self.model.predict(np.expand_dims(landmarks, axis=0))
                predicted_label = labels[np.argmax(pred)]
                prediction_buffer.append(predicted_label)
                most_common = Counter(prediction_buffer).most_common(1)
                gesture = most_common[0][0] if most_common else ""
            
            frame_cv = frame.copy()
            if gesture:
                cv2.rectangle(frame_cv, (0, 0), (300, 50), (0, 0, 0), -1)
                cv2.putText(frame_cv, f"Gesture: {gesture}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Gesture Recognition", frame_cv)
            key = cv2.waitKey(1)
            if key == ord('y') and gesture:
                print(f"Gesture confirmed: {gesture}")
                self.guessed_gesture = gesture
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def main(self):
        print("1. Collect Data\n2. Train Model\n3. Recognize Gesture\n4. Exit")
        while True:
            choice = input("Choose option: ")
            if choice == '1':
                self.collect_data()
            elif choice == '2':
                self.train_model()
            elif choice == '3':
                self.recognize()
            elif choice == '4':
                break
            else:
                print("Invalid choice.")

if __name__ == "__main__":
    Gesture_AI().main()


